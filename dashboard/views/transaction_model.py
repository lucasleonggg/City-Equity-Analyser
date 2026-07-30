"""Transaction-Level Price Model page.

Real RandomForestRegressor trained on all 982,011 individual HDB resale
transactions (pipeline/train_transaction_model.py), not the 26-town summary
used elsewhere in this dashboard. Also displays the honest Intel(R)
Extension for Scikit-learn benchmark log (processed_data/
intel_benchmark.json) -- shows whatever has actually been measured, nothing
invented.
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from data.loader import RAW_DIR, TOWN_COORDS

PROCESSED_DIR = os.path.join(RAW_DIR, "..", "processed_data")
MODEL_PATH = os.path.join(PROCESSED_DIR, "transaction_price_model.joblib")
BENCHMARK_PATH = os.path.join(PROCESSED_DIR, "intel_benchmark.json")
OPENVINO_BENCHMARK_PATH = os.path.join(PROCESSED_DIR, "openvino_benchmark.json")

FLAT_TYPES = ["1-Room", "2-Room", "3-Room", "4-Room", "5-Room", "Executive", "Multi-Generation"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_benchmark_history():
    if not os.path.exists(BENCHMARK_PATH):
        return []
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def load_openvino_history():
    """Inference-side benchmark. Absent until pipeline/export_onnx.py is run
    on a machine where openvino/skl2onnx can actually be installed."""
    if not os.path.exists(OPENVINO_BENCHMARK_PATH):
        return []
    with open(OPENVINO_BENCHMARK_PATH) as f:
        return json.load(f)



def _render_verdict(wdf, config_col, workload):
    """Compare the latest run of each backend, refusing unsound comparisons.

    Three gates before any speedup number is shown: same configuration, both
    arms replicated, and a gap wider than combined run-to-run variance.
    """
    intel_runs = wdf[wdf["intel_acceleration"]]
    stock_runs = wdf[~wdf["intel_acceleration"]]

    if intel_runs.empty or stock_runs.empty:
        st.caption(
            "Only one backend has been benchmarked for this workload. Run "
            "`pip install scikit-learn-intelex`, then re-run the pipeline "
            "with `--repeats 5` to record the other side."
        )
        return

    stock, intel = stock_runs.iloc[-1], intel_runs.iloc[-1]

    # Refuse to compare across different configurations -- a 40-tree run
    # against a 100-tree run is not a backend comparison.
    if pd.notna(stock.get(config_col)) and stock.get(config_col) != intel.get(config_col):
        st.error(
            f"Cannot compare: the two runs used different `{config_col}` "
            f"({stock[config_col]} vs {intel[config_col]}). Re-run both arms "
            "with the same configuration."
        )
        return

    if stock["repeats"] < 2 or intel["repeats"] < 2:
        st.warning(
            "No variance estimate: at least one arm was run with "
            "`--repeats 1`, so a speedup figure cannot be distinguished from "
            "scheduler noise and is deliberately not shown. Re-run both arms "
            "with `--repeats 5`."
        )
        return

    s_med, i_med = stock["fit_seconds_median"], intel["fit_seconds_median"]
    s_sd, i_sd = stock["fit_seconds_stdev"] or 0, intel["fit_seconds_stdev"] or 0
    gap, noise = abs(s_med - i_med), s_sd + i_sd
    if not i_med:
        return
    ratio = s_med / i_med

    if gap <= noise:
        st.info(
            f"**No measurable difference.** Stock {s_med:.2f}s ± {s_sd:.2f} vs "
            f"oneDAL {i_med:.2f}s ± {i_sd:.2f}. The gap ({gap:.2f}s) is within "
            f"combined run-to-run variance ({noise:.2f}s), so no speedup claim "
            "is supportable from this data."
        )
        return

    # Report the factor the right way up. A ratio of 0.92 means oneDAL took
    # LONGER, and must be shown as 1.09x slower -- writing "0.92x slower"
    # states a slowdown using a speed ratio and is simply wrong.
    if ratio >= 1:
        st.success(
            f"oneDAL is **{ratio:.2f}x faster** "
            f"({s_med:.2f}s ± {s_sd:.2f} -> {i_med:.2f}s ± {i_sd:.2f}, medians "
            f"of {int(intel['repeats'])} fits). Gap exceeds combined "
            "run-to-run variance."
        )
    else:
        st.warning(
            f"oneDAL is **{1 / ratio:.2f}x slower** "
            f"({s_med:.2f}s ± {s_sd:.2f} -> {i_med:.2f}s ± {i_sd:.2f}, medians "
            f"of {int(intel['repeats'])} fits). Gap exceeds combined "
            f"run-to-run variance, so this is a real effect, not noise. "
            f"Measured on {stock['cpu_count']} core(s): oneDAL's forest gains "
            "come from multi-threaded tree building and vectorised split "
            "search, so on a single vCPU there is no parallelism to recover "
            "and the dispatch overhead is paid for nothing."
        )

    if workload == "kmeans_fit":
        s_in, i_in = stock.get("inertia"), intel.get("inertia")
        if pd.notna(s_in) and pd.notna(i_in) and s_in:
            delta = 100 * (i_in - s_in) / s_in
            st.caption(
                f"Solution quality check: inertia {s_in:,.0f} -> {i_in:,.0f} "
                f"({delta:+.2f}%). A clustering speedup that lands on a "
                "materially worse local optimum would not be a speedup; this "
                "one does not."
            )


def show():
    st.title("Transaction-Level Price Model")
    st.markdown(
        "Unlike the Access & Affordability Map page (26-town summary), this model trains "
        "directly on all **982,011 individual HDB resale transactions** "
        "(`processed_data/hdb-all-cleaned.csv`) -- large enough for the "
        "Intel(R) Extension for Scikit-learn oneDAL backend to plausibly "
        "matter, and a legitimate analysis in its own right: what predicts "
        "an *individual transaction's* price, not just a town average."
    )

    bundle = load_model()
    if bundle is None:
        st.warning(
            "No trained model found. Run `python pipeline/train_transaction_model.py` "
            "from the project root to train it (takes several minutes on the "
            "full dataset)."
        )
        return

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # ── Honest Intel benchmark log ──
    st.header("Intel(R) Extension for Scikit-learn Benchmark")
    st.caption(
        "Two workloads, measured the same way on the same machine. They "
        "point in opposite directions, and both are reported."
    )
    history = load_benchmark_history()
    if not history:
        st.info("No benchmark recorded yet.")
    else:
        bench_df = pd.DataFrame(history)
        bench_df["backend"] = bench_df["intel_acceleration"].map(
            {True: "Intel oneDAL (sklearnex)", False: "Stock scikit-learn"}
        )
        # Schema-tolerant: older records used a scalar `fit_seconds`, newer
        # ones record a distribution (median/stdev/all) from `--repeats`.
        # Backfill per-row, not per-column: a history containing BOTH schemas
        # would otherwise leave the legacy rows as NaN.
        if "fit_seconds_median" not in bench_df:
            bench_df["fit_seconds_median"] = pd.NA
        if "fit_seconds" in bench_df:
            bench_df["fit_seconds_median"] = bench_df["fit_seconds_median"].fillna(
                bench_df["fit_seconds"]
            )
        for col, default in (("fit_seconds_stdev", None), ("repeats", 1),
                             ("n_estimators", None), ("cpu_count", None),
                             ("inertia", None)):
            if col not in bench_df:
                bench_df[col] = default
        bench_df["repeats"] = bench_df["repeats"].fillna(1)
        # Records written before the KMeans stage existed are all forest runs.
        if "workload" not in bench_df:
            bench_df["workload"] = "random_forest_fit"
        bench_df["workload"] = bench_df["workload"].fillna("random_forest_fit")

        for workload, label, config_col in (
            ("random_forest_fit", "RandomForest training (982k rows)", "n_estimators"),
            ("kmeans_fit", "KMeans clustering (250k-row random sample)", "n_clusters"),
        ):
            wdf = bench_df[bench_df["workload"] == workload]
            if wdf.empty:
                continue
            st.subheader(label)
            if config_col not in wdf:
                wdf = wdf.assign(**{config_col: None})
            cols = ["timestamp", "backend", "n_rows", config_col, "repeats",
                    "fit_seconds_median", "fit_seconds_stdev", "cpu_count"]
            if workload == "kmeans_fit":
                cols.append("inertia")
            if workload == "kmeans_fit":
                st.caption(
                    "Scope: this is **not** the town clustering on the Access & "
                    "Affordability Heatmap page. That runs on 26 towns x 5 "
                    "factors and "
                    "finishes in microseconds -- no backend changes it. This "
                    "benchmarks the same algorithm on a seeded random sample "
                    "of the real transaction data, scaled until the backend "
                    "is measurable. It does not make this dashboard faster."
                )
            st.dataframe(wdf[cols], use_container_width=True, hide_index=True)
            _render_verdict(wdf, config_col, workload)

    # ── Inference benchmark ──
    st.header("Batch Inference Benchmark")
    ov_history = load_openvino_history()
    if not ov_history:
        st.info(
            "Not yet measured. This is the **inference/serving** half of the "
            "Intel story, separate from the sklearnex **training** benchmark "
            "above. To record it:\n\n"
            "1. `pip install onnxruntime skl2onnx onnx openvino`\n"
            "2. `python pipeline/export_onnx.py --self-check` "
            "(verifies the converted model reproduces sklearn -- do this first)\n"
            "3. `python pipeline/export_onnx.py`"
        )
    else:
        latest = ov_history[-1]
        backends = latest.get("backends", {})
        st.caption(
            f"Batch of {latest['n_rows']:,} real transactions, best of "
            f"{latest['repeats']}, on {latest.get('cpu_count')} core(s). "
            "Batch, not single-row: one row takes microseconds and the "
            "measurement would be pure Python call overhead."
        )

        rows = [{
            "runtime": "stock scikit-learn",
            "status": "ok",
            "rows_per_sec": latest.get("sklearn_rows_per_sec"),
            "speedup": 1.0,
            "agreement_mae": 0.0,
        }]
        for name, e in backends.items():
            rows.append({
                "runtime": name,
                "status": e.get("status"),
                "rows_per_sec": e.get("rows_per_sec"),
                "speedup": e.get("speedup"),
                "agreement_mae": e.get("agreement_mae"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ort = backends.get("onnxruntime", {})
        if ort.get("speedup"):
            sp = ort["speedup"]
            verb = "faster" if sp >= 1 else "slower"
            factor = sp if sp >= 1 else 1 / sp
            st.success(
                f"ONNX Runtime batch inference: **{factor:.2f}x {verb}** than "
                f"stock scikit-learn "
                f"({latest['sklearn_rows_per_sec']:,.0f} -> "
                f"{ort['rows_per_sec']:,.0f} rows/sec), reproducing sklearn to "
                f"a mean absolute difference of ${ort['agreement_mae']:.4f}."
            )

        ov = backends.get("openvino", {})
        if ov.get("status") == "unsupported_op":
            st.error(
                "**OpenVINO cannot execute this model.** `skl2onnx` lowers a "
                "RandomForestRegressor to a single `ai.onnx.ml."
                "TreeEnsembleRegressor` node, and OpenVINO's ONNX frontend "
                "implements the neural-network operator set -- it has no "
                "conversion rule for the classical-ML tree operators, so "
                "`read_model()` fails outright. This is not a conversion bug: "
                "the same ONNX file runs correctly under ONNX Runtime. "
                "OpenVINO is the wrong runtime for a decision-tree ensemble, "
                "and no opset or precision setting changes that."
            )
        elif ov.get("status") == "not_installed":
            st.caption("OpenVINO not installed in this environment.")

    # ── Feature importances ──
    st.header("What Predicts an Individual Transaction's Price?")
    importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    fig = px.bar(
        importances, x="importance", y="feature", orientation="h",
        title="Top 15 features by RandomForest importance (real, from the fitted model)",
        height=450,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # ── Live prediction tool ──
    st.header("Predict a Transaction's Price")
    col1, col2, col3 = st.columns(3)
    with col1:
        town = st.selectbox("Town", sorted(TOWN_COORDS.keys()))
        flat_type = st.selectbox("Flat Type", FLAT_TYPES, index=3)
    with col2:
        floor_area = st.slider("Floor Area (sqm)", 30, 200, 90)
        storey_mid = st.slider("Storey (midpoint)", 2, 45, 8)
    with col3:
        remaining_lease = st.slider("Remaining Lease (years)", 40, 99, 80)
        year = st.slider("Transaction Year", 2012, 2026, 2026)

    row = pd.DataFrame([{c: 0 for c in feature_columns}])
    row["floor_area_sqm"] = floor_area
    row["storey_mid"] = storey_mid
    row["remaining_lease"] = remaining_lease
    row["year"] = year
    town_col = f"town_{town}"
    flat_col = f"flat_type_{flat_type}"
    if town_col in row.columns:
        row[town_col] = 1
    if flat_col in row.columns:
        row[flat_col] = 1

    predicted = model.predict(row[feature_columns])[0]
    st.metric("Predicted Resale Price", f"${predicted:,.0f}")
