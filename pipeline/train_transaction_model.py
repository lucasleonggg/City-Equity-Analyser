"""Train transaction-level HDB resale price models and benchmark
scikit-learn vs. Intel(R) Extension for Scikit-learn (sklearnex / oneDAL).

Trains on processed_data/hdb-all-cleaned.csv: 982,011 real individual resale
transactions.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
The previous script did `model.fit(X, y)` then `model.score(X, y)` and
reported R^2 = 0.9424 with no split, no cross-validation and no error metric.
That number was IN-SAMPLE and therefore close to meaningless as evidence of
predictive quality -- a depth-14 forest on 982k rows will of course fit
itself. This version reports:
  - a held-out test split (chronological AND random, because HDB prices are
    strongly time-trended and a random split leaks future information),
  - K-fold cross-validated R^2,
  - MAE and RMSE in dollars, which are the metrics anyone pricing a flat
    actually cares about,
  - the in-sample number too, clearly labelled as such for comparison.

Two models are saved:
  1. transaction_price_model.joblib -- full-feature model for the dashboard's
     prediction tool and feature importances.
  2. transaction_shapley_model.joblib -- a compact 6-feature model plus
     per-town representative profiles, so equity_factors.compute_shapley_price_drivers
     can do EXACT (2^6 = 64 coalition) Shapley attribution against a model fit
     on 982k rows instead of the old 23-row town forest.

INTEL BENCHMARK
----------------
Every run appends to processed_data/intel_benchmark.json, tagged with whether
sklearnex was active. Run once without and once with
`pip install scikit-learn-intelex` to get a real before/after on identical
hardware. Do NOT compare a run on one machine against a run on another --
the recorded cpu_count / platform fields exist so mismatched comparisons are
visible rather than accidental.
"""

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_ACCELERATION = True
except ImportError:
    INTEL_ACCELERATION = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(HERE, "..", "processed_data")
DATA_PATH = os.path.join(PROCESSED_DIR, "hdb-all-cleaned.csv")
MODEL_PATH = os.path.join(PROCESSED_DIR, "transaction_price_model.joblib")
SHAPLEY_MODEL_PATH = os.path.join(PROCESSED_DIR, "transaction_shapley_model.joblib")
BENCHMARK_PATH = os.path.join(PROCESSED_DIR, "intel_benchmark.json")
METRICS_PATH = os.path.join(PROCESSED_DIR, "model_metrics.json")

NUMERIC_COLS = ["floor_area_sqm", "storey_mid", "remaining_lease", "year"]
CATEGORICAL_COLS = ["town", "flat_type"]

# Compact feature set for exact Shapley (2^6 coalitions).
SHAPLEY_FEATURES = [
    "floor_area_sqm", "storey_mid", "remaining_lease",
    "year", "town_median_price_rank", "flat_type_rooms",
]

FLAT_TYPE_ROOMS = {
    "1-Room": 1, "2-Room": 2, "3-Room": 3, "4-Room": 4,
    "5-Room": 5, "Executive": 6, "Multi-Generation": 7,
}


def _storey_mid(s: str) -> float:
    a, b = s.split(" TO ")
    return (int(a) + int(b)) / 2


def load_transactions() -> pd.DataFrame:
    """Memory-lean load.

    The full cleaned CSV has 9 string columns over 982k rows; loading it
    naively costs ~1GB in Python str objects alone, which is what previously
    OOM-killed this script on a small box. Only the columns actually used are
    read, and the two high-cardinality ones are loaded as `category`.

    The header also has duplicated year/quarter columns
    (...,year,quarter,year,quarter) from the pipeline merge step; pandas
    de-duplicates them as year/year.1, and the .1 copies are dropped.
    """
    usecols = [
        "town", "flat_type", "storey_range", "floor_area_sqm",
        "remaining_lease", "resale_price", "year",
    ]
    df = pd.read_csv(
        DATA_PATH,
        usecols=usecols,
        dtype={
            "town": "category", "flat_type": "category",
            "storey_range": "category",
            "floor_area_sqm": "float32", "remaining_lease": "float32",
            "resale_price": "float32", "year": "int16",
        },
    )
    storey_map = {s: _storey_mid(s) for s in df["storey_range"].cat.categories}
    df["storey_mid"] = df["storey_range"].map(storey_map).astype("float32")
    df = df.drop(columns=["storey_range"])
    return df


def evaluate(model, X_tr, y_tr, X_te, y_te, label):
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    return {
        "split": label,
        "test_r2": round(float(r2_score(y_te, pred)), 4),
        "test_mae": round(float(mean_absolute_error(y_te, pred))),
        "test_rmse": round(rmse),
        "in_sample_r2": round(float(model.score(X_tr, y_tr)), 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


def _forest(n_estimators=100, max_depth=14):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=-1, random_state=42,
    )


def stage_full(df, Xv, y, feature_columns, n_estimators=100, max_depth=14, repeats=1):
    """Random 80/20 split -> benchmark timing + saved dashboard model.

    `repeats` refits the SAME configuration on the SAME arrays N times and
    reports median and stdev. This exists because a single paired fit cannot
    distinguish a real backend effect from scheduler noise: on a shared
    single-vCPU box, back-to-back identical sklearnex fits were measured at
    257.77s and 202.12s -- a 55s swing, wider than the entire between-backend
    difference. One measurement per arm would have supported a confident
    "33% slower" claim that the second measurement refuted.
    """
    idx = np.arange(len(Xv))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)

    fit_times = []
    model = None
    for i in range(repeats):
        del model
        gc.collect()
        model = _forest(n_estimators=n_estimators, max_depth=max_depth)
        t0 = time.perf_counter()
        model.fit(Xv[tr], y[tr])
        fit_times.append(time.perf_counter() - t0)
        print(f"  fit {i + 1}/{repeats}: {fit_times[-1]:.2f}s")

    m = evaluate(model, Xv[tr], y[tr], Xv[te], y[te], "random_80_20")
    print(f"  random split: {m}")

    timing = {
        "fit_seconds_median": round(float(np.median(fit_times)), 2),
        "fit_seconds_min": round(float(np.min(fit_times)), 2),
        "fit_seconds_stdev": round(float(np.std(fit_times, ddof=1)), 2) if len(fit_times) > 1 else None,
        "fit_seconds_all": [round(t, 2) for t in fit_times],
        "repeats": repeats,
    }
    if repeats == 1:
        print("  WARNING: repeats=1 -- no variance estimate. Do NOT compare "
              "backends off a single fit; use --repeats 5.")
    else:
        print(f"  median {timing['fit_seconds_median']:.2f}s "
              f"(stdev {timing['fit_seconds_stdev']:.2f}s over {repeats} fits)")

    joblib.dump(
        {"model": model, "feature_columns": feature_columns, "metrics": m},
        MODEL_PATH, compress=3,
    )
    print(f"  saved {MODEL_PATH}")
    fitted_params = {
        "n_estimators": int(model.n_estimators),
        "max_depth": int(model.max_depth),
    }
    del model
    gc.collect()
    return m, timing, fitted_params


def stage_chrono(df, Xv, y, n_estimators=100, max_depth=14):
    """Chronological split -- the honest test. Prices are strongly
    time-trended, so a random split leaks future information.

    Tree count is READ OFF THE FITTED MODEL, never written into the label by
    hand. A previous version hardcoded "(25 trees)" into the split string
    while `_forest()` was actually being called with its default of 100; that
    stale string survived in model_metrics.json long after the code that
    produced it was gone, and there was no way to tell from the JSON alone
    that the label was lying.
    """
    cutoff = int(df["year"].quantile(0.8))
    tr = (df["year"] <= cutoff).to_numpy()
    model = _forest(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(Xv[tr], y[tr])
    m = evaluate(model, Xv[tr], y[tr], Xv[~tr], y[~tr],
                 f"train<={cutoff} / test>{cutoff}")
    m["n_estimators"] = int(model.n_estimators)
    m["max_depth"] = int(model.max_depth)
    print(f"  chronological split: {m}")
    del model
    gc.collect()
    return m


def stage_cv(Xv, y, n_sub=80000):
    sub = np.random.default_rng(42).choice(len(Xv), size=min(n_sub, len(Xv)), replace=False)
    Xs, ys = Xv[sub], y[sub]
    scores = []
    fitted_n_estimators = fitted_max_depth = None
    for tr_i, te_i in KFold(n_splits=5, shuffle=True, random_state=42).split(Xs):
        m = _forest(n_estimators=50)
        m.fit(Xs[tr_i], ys[tr_i])
        scores.append(float(r2_score(ys[te_i], m.predict(Xs[te_i]))))
        fitted_n_estimators, fitted_max_depth = m.n_estimators, m.max_depth
        del m
        gc.collect()
    out = {
        "n_subsample": int(len(sub)),
        "mean_r2": round(float(np.mean(scores)), 4),
        "std_r2": round(float(np.std(scores)), 4),
        # Read off the last fitted estimator, not the literal above.
        "n_estimators": int(fitted_n_estimators),
        "max_depth": int(fitted_max_depth),
    }
    print(f"  5-fold CV (subsample): {out}")
    return out


def append_benchmark(record):
    history = []
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH) as f:
            history = json.load(f)
    history.append(record)
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Appended benchmark to {BENCHMARK_PATH}")


def stage_kmeans(Xv, n_clusters=4, n_init=10, n_rows=250_000, repeats=5):
    """Benchmark KMeans -- the OTHER algorithm sklearnex patches.

    WHY THIS STAGE EXISTS
    ---------------------
    The RandomForest benchmark came back with oneDAL SLOWER (see
    intel_benchmark.json, workload=random_forest_fit). That result is real and
    is reported as measured. But it is a result about one algorithm on one
    core, and reporting it alone would imply the extension does nothing, which
    is not what the data says.

    Forest training on a single vCPU is close to the worst case for oneDAL:
    its gains there come from multi-threaded tree building and vectorised
    split search, and with one core there is no parallelism to recover, so the
    dispatch overhead is paid for nothing. KMeans is a different shape of
    problem -- dense distance computation that vectorises within a single core
    -- and is where oneDAL's optimisation actually lands on this hardware.

    Measuring both is the point. One number is an anecdote; two numbers
    pointing opposite directions, with a mechanism that explains which is
    which, is a finding.

    WHAT THIS IS NOT
    ----------------
    This is NOT the dashboard's town clustering. That call
    (equity_factors.py, `KMeans(n_clusters=4, n_init=10)`) runs on 26 towns
    x 5 factors, finishes in microseconds, and would show nothing
    whatever a backend did to it. Nothing measured here makes the live
    dashboard faster, and it must not be presented as if it did.

    What this IS: the same algorithm the dashboard uses, on real rows from
    the same dataset, scaled to a size where the backend is actually
    measurable. It is a benchmark of oneDAL's KMeans path, not a benchmark of
    this application. Say that out loud, not just in the README.

    Reports inertia alongside timing, because a clustering speedup that
    silently lands on a worse local optimum is not a speedup.
    """
    from sklearn.cluster import KMeans

    # Seeded RANDOM sample, not Xv[:n_rows]. A positional head slice of this
    # CSV is not a neutral subset: the file is concatenated per-era, so the
    # first 250k rows span 2000-2016 (median 2003) against 1990-2026 (median
    # 2006) for the full set, with a mean resale price $23k below the whole.
    # Timing is largely insensitive to that, but inertia is not, and a
    # reviewer should not have to take "insensitive" on trust.
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(Xv), size=min(n_rows, len(Xv)), replace=False))
    X = np.ascontiguousarray(Xv[idx])
    print(f"  KMeans benchmark: {X.shape}, k={n_clusters}, n_init={n_init}")

    fit_times, inertias = [], []
    for i in range(repeats):
        gc.collect()
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        t0 = time.perf_counter()
        km.fit(X)
        fit_times.append(time.perf_counter() - t0)
        inertias.append(float(km.inertia_))
        print(f"    fit {i + 1}/{repeats}: {fit_times[-1]:.2f}s")
        del km

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workload": "kmeans_fit",
        "intel_acceleration": INTEL_ACCELERATION,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": n_clusters,
        "n_init": n_init,
        "fit_seconds_median": round(float(np.median(fit_times)), 2),
        "fit_seconds_min": round(float(np.min(fit_times)), 2),
        "fit_seconds_stdev": round(float(np.std(fit_times, ddof=1)), 2) if len(fit_times) > 1 else None,
        "fit_seconds_all": [round(t, 2) for t in fit_times],
        "repeats": repeats,
        # Guards against a "speedup" that is really a worse solution.
        "inertia": round(float(np.median(inertias)), 1),
        "cpu_count": os.cpu_count(),
        "platform": platform.processor() or platform.machine(),
    }
    append_benchmark(record)
    return record


def stage_shapley(df):
    """Compact 6-feature model so exact (2^6) Shapley is tractable against a
    model fit on all 982k transactions, rather than the old 23-row forest."""
    town_rank = df.groupby("town", observed=True)["resale_price"].median().rank(pct=True).to_dict()
    S = pd.DataFrame({
        "floor_area_sqm": df["floor_area_sqm"],
        "storey_mid": df["storey_mid"],
        "remaining_lease": df["remaining_lease"],
        "year": df["year"].astype("float32"),
        "town_median_price_rank": df["town"].astype(object).map(town_rank).astype("float32"),
        "flat_type_rooms": df["flat_type"].astype(object).map(FLAT_TYPE_ROOMS).astype("float32"),
    }).dropna()
    ys = df.loc[S.index, "resale_price"].to_numpy(dtype="float32")
    Sv = S.to_numpy(dtype="float32")

    model = _forest(n_estimators=60, max_depth=12)
    model.fit(Sv, ys)

    latest_year = int(df["year"].max())
    recent = df[df["year"] == latest_year]
    profiles = {}
    for town, g in recent.groupby("town", observed=True):
        profiles[town] = [
            float(g["floor_area_sqm"].median()),
            float(g["storey_mid"].median()),
            float(g["remaining_lease"].median()),
            float(latest_year),
            float(town_rank.get(town, 0.5)),
            float(g["flat_type"].astype(object).map(FLAT_TYPE_ROOMS).median()),
        ]

    joblib.dump({
        "model": model, "feature_names": SHAPLEY_FEATURES,
        "background": S.median().tolist(), "town_profiles": profiles,
        "n_training_rows": int(len(S)),
    }, SHAPLEY_MODEL_PATH, compress=3)
    print(f"  saved {SHAPLEY_MODEL_PATH} ({len(S):,} rows)")
    del model, S, Sv
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stages", default="full,chrono,cv,kmeans,shapley",
        help="Comma-separated subset. Run one per process on low-memory boxes.",
    )
    ap.add_argument(
        "--trees", type=int, default=100,
        help="n_estimators. MUST match across backends when comparing. "
             "Use a smaller value (e.g. 12) to afford replicates.",
    )
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument(
        "--repeats", type=int, default=1,
        help="Refits per run, for a variance estimate. Use 5 when comparing "
             "stock scikit-learn against sklearnex; a single fit cannot "
             "separate a backend effect from scheduler noise.",
    )
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    print(f"Loading {DATA_PATH} ...")
    df = load_transactions()
    X = pd.get_dummies(df[NUMERIC_COLS + CATEGORICAL_COLS],
                       columns=CATEGORICAL_COLS).astype("float32")
    feature_columns = list(X.columns)
    Xv = X.to_numpy(dtype="float32", copy=False)
    y = df["resale_price"].to_numpy(dtype="float32")
    del X
    gc.collect()

    print(f"{len(Xv):,} transactions | {Xv.shape[1]} features")
    print(f"Intel(R) Extension for Scikit-learn active: {INTEL_ACCELERATION}")

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    timing = None
    fitted_params = {}

    if "full" in stages:
        metrics["random_split"], timing, fitted_params = stage_full(
            df, Xv, y, feature_columns,
            n_estimators=args.trees, max_depth=args.max_depth,
            repeats=args.repeats,
        )
    if "chrono" in stages:
        metrics["chronological_split"] = stage_chrono(
            df, Xv, y, n_estimators=args.trees, max_depth=args.max_depth
        )
    if "cv" in stages:
        metrics["cv_5fold_subsample"] = stage_cv(Xv, y)
    if "kmeans" in stages:
        stage_kmeans(Xv, repeats=max(args.repeats, 2))
    if "shapley" in stages:
        stage_shapley(df)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {METRICS_PATH}")

    if timing is not None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(Xv)),
            "n_features": int(Xv.shape[1]),
            "workload": "random_forest_fit",
            "intel_acceleration": INTEL_ACCELERATION,
            # Read off the fitted estimator, never hardcoded. Literals here
            # previously let the JSON record 100 trees while the run actually
            # used 40, silently invalidating any cross-run comparison.
            **fitted_params,
            **timing,
            "cpu_count": os.cpu_count(),
            "platform": platform.processor() or platform.machine(),
            "test_r2_random_split": metrics["random_split"]["test_r2"],
        }
        append_benchmark(record)


if __name__ == "__main__":
    main()
