# -*- coding: utf-8 -*-
"""
mt_app.py
Streamlit UI for melting temperature (°C) prediction.
Now shows only one number by default (Ensemble). Optional breakdown toggles.
"""

import os
import io
import json
import time
import tempfile
from typing import List

import streamlit as st
import pandas as pd

from mt_infer import MTEnsemble  # uses new --model logic internally


st.set_page_config(page_title="MT Predictor (Chemprop + RF)", page_icon="🧪", layout="wide")

# Sidebar settings
st.sidebar.title("Settings")
base_dir = st.sidebar.text_input("Models base dir", value="models")
version  = st.sidebar.text_input("Model version", value="v1")

@st.cache_resource
def load_model(_base_dir: str, _version: str) -> MTEnsemble:
    return MTEnsemble(base_dir=_base_dir, version=_version)

try:
    model = load_model(base_dir, version)
    st.sidebar.success(f"Loaded mt_regression/{version}")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

st.title("Melting Temperature (°C) — Chemprop + RF Ensemble")
st.caption("Default output: a single Ensemble prediction. Toggle breakdown if needed.")

tab_single, tab_csv = st.tabs(["🔹 Single SMILES", "📄 CSV Batch"])

# ------- Single -------
with tab_single:
    st.subheader("Single SMILES")
    smiles = st.text_input("Enter a SMILES:", value="N#Cc1ccc(cc1)c2ccc(cc2)CCCCC", placeholder="e.g., N#Cc1ccc(cc1)c2ccc(cc2)CCCCC")
    show_breakdown = st.checkbox("Show breakdown (Chemprop & RF)", value=False)
    go = st.button("Predict", type="primary")

    if go:
        with st.spinner("Predicting..."):
            t0 = time.time()
            # Single-number output (ensemble)
            val = model.predict_values([smiles], which="ensemble")[0]
            dt = time.time() - t0
            st.success(f"Done in {dt:.2f}s")
            st.metric("Predicted MT (Ensemble)", f"{val:.2f} {model.unit}")

            if show_breakdown:
                full = model.predict_many_full([smiles])[0]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Chemprop", f"{full['per_model']['chemprop']:.2f} {model.unit}")
                with c2:
                    st.metric("RF", f"{full['per_model']['rf']:.2f} {model.unit}")
                st.caption(f"w_RF = {full['ensemble']['w_RF']:.2f}")

# ------- CSV -------
with tab_csv:
    st.subheader("CSV Batch Prediction")
    st.caption("The CSV must contain a SMILES column (default: `canonical_smiles`).")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    smiles_col = st.text_input("SMILES column name", value="canonical_smiles")
    include_breakdown_cols = st.checkbox("Include per-model columns in the output CSV", value=False)
    run_batch = st.button("Run batch prediction", type="primary", disabled=(uploaded is None))

    if uploaded and run_batch:
        try:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            if smiles_col not in df.columns:
                st.error(f"Column '{smiles_col}' not found. Columns: {list(df.columns)}")
            else:
                with st.spinner("Predicting on CSV..."):
                    smi = df[smiles_col].astype(str).tolist()
                    # Ensemble only
                    ens_vals = model.predict_values(smi, which="ensemble")
                    out_df = pd.DataFrame({"SMILES": smi, "Pred_Ensemble": ens_vals})

                    if include_breakdown_cols:
                        full = model.predict_many_full(smi)
                        out_df["Pred_Chemprop"] = [r["per_model"]["chemprop"] for r in full]
                        out_df["Pred_RF"]       = [r["per_model"]["rf"] for r in full]

                st.success("Done.")
                st.dataframe(out_df, use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="mt_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
