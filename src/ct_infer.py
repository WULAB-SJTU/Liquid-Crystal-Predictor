

import os
import sys
import json
import uuid
import shutil
import tempfile
import subprocess
from glob import glob
from typing import List, Dict, Any, Optional
import pandas as pd

# ---------------- Utils ----------------
def _chemprop_bin() -> str:
    """
    Resolve chemprop executable inside current conda env on Windows/Linux/Mac.
    """
    candidates = [
        os.path.join(sys.prefix, 'Scripts', 'chemprop.exe'),
        os.path.join(sys.prefix, 'Scripts', 'chemprop'),
        'chemprop'
    ]
    for c in candidates:
        try:
            subprocess.run([c, '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return c
        except Exception:
            continue
    return 'chemprop'  # best effort

def _safe_abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_meta(meta_path: str) -> Dict[str, Any]:
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _find_best_model(model_dir: str) -> str:
    """
    Prefer model_*/best.pt; fallback to best.pt at root.
    """
    paths = sorted(glob(os.path.join(model_dir, 'model_*', 'best.pt')))
    if paths:
        return paths[0]
    single = os.path.join(model_dir, 'best.pt')
    if os.path.exists(single):
        return single
    raise FileNotFoundError(f"best.pt not found under {model_dir}")

def _run_predict(chemprop_bin: str, model_path: str, cp_input_csv: str, out_csv: str):
    """
    Call chemprop predict with the hyphen flags (we validated these on your machine):
        --model-paths <best.pt> -i <input> -o <output>
    """
    cmd = [chemprop_bin, 'predict',
           '--model-paths', model_path,
           '-i', cp_input_csv,
           '-o', out_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "chemprop predict failed.\n"
            f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

def _to_cp_csv_from_smiles(smiles_list: List[str], out_csv: str):
    """
    Build a chemprop-compatible CSV (columns: smiles) for inference-only use.
    If target column is absent (pure inference), chemprop still accepts a single 'smiles' column.
    """
    df = pd.DataFrame({'smiles': [s if s is not None else '' for s in smiles_list]})
    df.to_csv(out_csv, index=False)

def _to_cp_csv_from_raw(input_csv: str, smiles_col: str, out_csv: str):
    """
    Convert a raw CSV with SMILES in smiles_col -> chemprop CSV with column 'smiles'.
    """
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {input_csv}.")
    cp = df[[smiles_col]].copy()
    cp.columns = ['smiles']
    cp.to_csv(out_csv, index=False)

# ---------------- Core ----------------
class CTRegressor:
    def __init__(self, base_dir: str = "models", task_name: str = "ct_regression", version: str = "v2"):
        """
        Try to load version; if not found, fallback to v1 automatically.
        """
        self.base_dir = base_dir
        self.task_name = task_name

        vdir = os.path.join(base_dir, task_name, version)
        if not os.path.exists(vdir):
            # fallback
            alt = os.path.join(base_dir, task_name, "v1")
            if os.path.exists(alt):
                vdir = alt
                version = "v1"
            else:
                raise FileNotFoundError(f"Model directory not found: {vdir}")

        self.version = version
        self.model_dir = vdir
        self.chemprop_bin = _chemprop_bin()

        # meta is optional
        self.meta = _read_meta(os.path.join(self.model_dir, "meta.json"))
        # find best.pt
        self.model_path = _find_best_model(self.model_dir)

        # unit from meta if present (defaults to degC)
        target = self.meta.get("target", {})
        self.unit = target.get("unit", "°C")

    # --------- Public APIs ---------
    def predict_one(self, smiles: str) -> Dict[str, Any]:
        """
        Predict a single SMILES. Returns {"ok": bool, "prediction": float, "unit": str, ...}
        """
        if not isinstance(smiles, str) or smiles.strip() == "":
            return {"ok": False, "error": "Invalid or empty SMILES.", "input": {"smiles": smiles}}

        # temp working dir
        with tempfile.TemporaryDirectory(prefix="ct_infer_") as tmp:
            in_csv = os.path.join(tmp, "input.csv")
            out_csv = os.path.join(tmp, "preds.csv")
            _to_cp_csv_from_smiles([smiles], in_csv)
            _run_predict(self.chemprop_bin, self.model_path, in_csv, out_csv)

            pr = pd.read_csv(out_csv)
            # chemprop v2 predictions column name is usually 'preds'
            if 'preds' in pr.columns:
                yhat = float(pr['preds'].iloc[0])
            else:
                yhat = float(pr.iloc[0, -1])

        return {
            "ok": True,
            "model": {"task": self.task_name, "version": self.version},
            "input": {"smiles": smiles},
            "prediction": yhat,
            "unit": self.unit
        }

    def predict_many(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Batch predict a list of SMILES (keeps order).
        """
        if not smiles_list:
            return []

        with tempfile.TemporaryDirectory(prefix="ct_infer_") as tmp:
            in_csv = os.path.join(tmp, "input.csv")
            out_csv = os.path.join(tmp, "preds.csv")
            _to_cp_csv_from_smiles(smiles_list, in_csv)
            _run_predict(self.chemprop_bin, self.model_path, in_csv, out_csv)

            pr = pd.read_csv(out_csv)
            pred_col = 'preds' if 'preds' in pr.columns else pr.columns[-1]
            preds = pr[pred_col].astype(float).tolist()

        results = []
        for s, y in zip(smiles_list, preds):
            results.append({
                "ok": True,
                "model": {"task": self.task_name, "version": self.version},
                "input": {"smiles": s},
                "prediction": float(y),
                "unit": self.unit
            })
        return results

    def predict_csv(self, input_csv: str, smiles_col: str = "SMILES",
                    output_csv: Optional[str] = None) -> str:
        """
        Predict for a CSV file containing a SMILES column. Writes a CSV with 'SMILES' and 'Predicted'.
        Returns the path to the written CSV.
        """
        input_csv = _safe_abspath(input_csv)
        if output_csv is None:
            base, name = os.path.split(input_csv)
            stem = os.path.splitext(name)[0]
            output_csv = os.path.join(base, f"{stem}_predicted_ct.csv")
        output_csv = _safe_abspath(output_csv)

        with tempfile.TemporaryDirectory(prefix="ct_infer_") as tmp:
            cp_in = os.path.join(tmp, "cp_input.csv")
            cp_out = os.path.join(tmp, "cp_preds.csv")
            _to_cp_csv_from_raw(input_csv, smiles_col, cp_in)
            _run_predict(self.chemprop_bin, self.model_path, cp_in, cp_out)

            pr = pd.read_csv(cp_out)
            pred_col = 'preds' if 'preds' in pr.columns else pr.columns[-1]
            preds = pr[pred_col].astype(float).tolist()

            raw = pd.read_csv(input_csv)
            raw.columns = raw.columns.str.strip()
            if smiles_col not in raw.columns:
                raise ValueError(f"Column '{smiles_col}' not found in {input_csv}.")
            if len(preds) != len(raw):
                raise RuntimeError("Row count mismatch between input and predictions.")

            out_df = pd.DataFrame({
                "SMILES": raw[smiles_col].astype(str).values,
                "Predicted": preds
            })
            out_df.to_csv(output_csv, index=False)

        return output_csv


# ---------------- CLI ----------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Predict clearing temperature using a frozen Chemprop model.")
    ap.add_argument("smiles", nargs="*", help="SMILES strings for quick one-off predictions.")
    ap.add_argument("--csv", type=str, help="CSV file to predict on (contains a SMILES column).")
    ap.add_argument("--smiles-col", type=str, default="SMILES", help="SMILES column name in the CSV.")
    ap.add_argument("--out", type=str, help="Output CSV path for --csv mode.")
    ap.add_argument("--base-dir", type=str, default="models", help="Base models dir.")
    ap.add_argument("--version", type=str, default="v2", help="Model version under models/ct_regression/. Fallback to v1 if missing.")
    args = ap.parse_args()

    infer = CTRegressor(base_dir=args.base_dir, task_name="ct_regression", version=args.version)

    if args.csv:
        out = infer.predict_csv(args.csv, smiles_col=args.smiles_col, output_csv=args.out)
        print(json.dumps({"ok": True, "mode": "csv", "output_csv": out}, ensure_ascii=False))
        return

    if args.smiles:
        res = infer.predict_many(args.smiles)
        print(json.dumps(res, ensure_ascii=False))
        return

    ap.print_help()

if __name__ == "__main__":
    _main()
