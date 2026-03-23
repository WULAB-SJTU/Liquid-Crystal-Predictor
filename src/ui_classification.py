import streamlit as st
from classification_infer import LCEnsemble

st.set_page_config(page_title="LC Classifier", layout="centered")
st.title("Liquid Crystal Classifier (Ensemble)")

task = st.text_input("Task", "lc_classification")
version = st.text_input("Version", "v1")
base_dir = st.text_input("Base dir", "models")
smiles = st.text_input("SMILES", "CCO")

if "clf" not in st.session_state:
    st.session_state.clf = LCEnsemble(base_dir=base_dir, task_name=task, version=version)

if st.button("Reload model"):
    st.session_state.clf = LCEnsemble(base_dir=base_dir, task_name=task, version=version)
    st.success("Reloaded.")

if st.button("Predict"):
    res = st.session_state.clf.predict_one(smiles)
    if res.get("ok", False):
        st.subheader(f"Ensemble: {res['ensemble']['label_text']}")
        st.json(res)
    else:
        st.error(res.get("error","Unknown error"))
        st.json(res)

st.caption("Tip: invalid/empty SMILES returns ok=False without crashing.")