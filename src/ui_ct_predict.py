
import os
import sys
from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st

# Ensure we can import ct_infer from the same folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ct_infer import CTRegressor  # wraps chemprop CLI predict

st.set_page_config(
    page_title="Clearing Temperature Predictor",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Clearing Temperature Predictor")
st.caption("Powered by a frozen Chemprop GNN model (ct_regression, v1)")

# ---------------- Sidebar ----------------
st.sidebar.header("Model settings")
base_dir = st.sidebar.text_input("Models base directory", value="models")
st.sidebar.code(rf"{base_dir}\ct_regression\v1\model_0\best.pt")

@st.cache_resource(show_spinner=False)
def load_model(_base_dir: str) -> CTRegressor:
    # Fixed to v1 as requested
    return CTRegressor(base_dir=_base_dir, task_name="ct_regression", version="v1")

# Try to load the model once
try:
    infer = load_model(base_dir)
    st.success("Model loaded: ct_regression / v1")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------- Tabs ----------------
tab_single, tab_batch = st.tabs(["Single SMILES", "Batch CSV"])

# ---- Single SMILES ----
with tab_single:
    st.subheader("Single SMILES Prediction")
    smi = st.text_input("SMILES", value="", placeholder="e.g., N#Cc1ccc(cc1)c2ccc(cc2)CCCCC")
    colA, colB = st.columns([1, 1])
    with colA:
        run_single = st.button("Predict", type="primary")
    with colB:
        if st.button("Fill Example (N#Cc1ccc(cc1)c2ccc(cc2)CCCCC)"):
            smi = "N#Cc1ccc(cc1)c2ccc(cc2)CCCCC"
            st.rerun()

    if run_single:
        if not smi.strip():
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Predicting..."):
                try:
                    res = infer.predict_one(smi.strip())
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    if not res.get("ok"):
                        st.error(res.get("error", "Unknown error"))
                    else:
                        st.success("Done")
                        st.metric(
                            label="Predicted clearing temperature",
                            value=f"{res['prediction']:.2f} {res.get('unit','°C')}",
                            delta=None
                        )

# ---- Batch CSV ----
with tab_batch:
    st.subheader("Batch CSV Prediction")
    st.caption("The CSV must contain a SMILES column (default name: 'SMILES').")
    up = st.file_uploader("Upload CSV", type=["csv"])
    smiles_col = st.text_input("SMILES column name", value="SMILES")

    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            if smiles_col not in df.columns:
                st.error(f"Column '{smiles_col}' not found. Available columns: {list(df.columns)}")
            else:
                st.dataframe(df.head(20), use_container_width=True)

                if st.button("Run batch prediction", type="primary"):
                    with st.spinner("Predicting..."):
                        try:
                            # Write uploaded CSV to a temporary file for ct_infer
                            with tempfile.TemporaryDirectory() as tmpdir:
                                tmp_in = Path(tmpdir) / "input.csv"
                                df.to_csv(tmp_in, index=False)

                                # Let ct_infer produce an output CSV; if output path is None,
                                # ct_infer will place it next to the input with a default name.
                                out_path = infer.predict_csv(str(tmp_in), smiles_col=smiles_col, output_csv=None)
                                out_df = pd.read_csv(out_path)
                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")
                        else:
                            st.success("Done")
                            st.dataframe(out_df.head(50), use_container_width=True)
                            st.download_button(
                                "Download predictions CSV",
                                data=out_df.to_csv(index=False).encode("utf-8"),
                                file_name="ct_predictions.csv",
                                mime="text/csv"
                            )
    else:
        st.info("Upload a CSV to enable batch prediction.")

# ---------------- Footer ----------------
with st.expander("Model meta (optional)"):
    st.json(infer.meta or {"note": "meta.json not found; using best.pt only."})
