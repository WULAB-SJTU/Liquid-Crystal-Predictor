
import os
import sys
import json
import tempfile
import subprocess
from glob import glob
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem


def _chemprop_bin() -> str:
    cands = [
        os.path.join(sys.prefix, 'Scripts', 'chemprop.exe'),
        os.path.join(sys.prefix, 'Scripts', 'chemprop'),
        'chemprop'
    ]
    for c in cands:
        try:
            subprocess.run([c, '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return c
        except Exception:
            continue
    return 'chemprop'


def _find_best_model(model_dir: str) -> str:
    paths = sorted(glob(os.path.join(model_dir, 'model_*', 'best.pt')))
    if paths:
        return paths[0]
    single = os.path.join(model_dir, 'best.pt')
    if os.path.exists(single):
        return single
    raise FileNotFoundError(f"best.pt not found under {model_dir}")


def _to_cp_from_smiles(smiles: List[str], out_csv: str):
    pd.DataFrame({'smiles': [s if s else '' for s in smiles]}).to_csv(out_csv, index=False)


def _run_predict(chemprop_bin: str, model_path: str, cp_input_csv: str, out_csv: str):
    cmd = [chemprop_bin, 'predict', '--model-paths', model_path, '-i', cp_input_csv, '-o', out_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"chemprop predict failed.\nCMD: {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}")


def _smiles_to_fp(smiles: str, radius=3, n_bits=2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for idx in bv.GetOnBits():
        arr[idx] = 1
    return arr


class MTEnsemble:
    """Frozen ensemble wrapper (Chemprop + RF + weighted sum)."""

    def __init__(self, base_dir: str = "models", version: str = "v1"):
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, "mt_regression", version)
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        meta_path = os.path.join(self.model_dir, "meta.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        self.unit = self.meta.get("target", {}).get("unit", "°C")
        fp = self.meta.get("fingerprint", {})
        self.radius = int(fp.get("radius", 3))
        self.n_bits = int(fp.get("n_bits", 2048))
        ens = self.meta.get("ensemble", {})
        self.w_rf = float(ens.get("w_RF", 0.5))

        self.chemprop_bin = _chemprop_bin()
        self.cp_model = _find_best_model(self.model_dir)
        self.rf = load(os.path.join(self.model_dir, "rf.joblib"))

    def _predict_cp_many(self, smiles_list: List[str]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="mt_infer_") as tmp:
            cp_in = os.path.join(tmp, "in.csv")
            cp_out = os.path.join(tmp, "out.csv")
            _to_cp_from_smiles(smiles_list, cp_in)
            _run_predict(self.chemprop_bin, self.cp_model, cp_in, cp_out)
            pr = pd.read_csv(cp_out)
            col = 'preds' if 'preds' in pr.columns else pr.columns[-1]
            return pr[col].astype(float).to_numpy()

    def _predict_rf_many(self, smiles_list: List[str]) -> np.ndarray:
        X = np.vstack([_smiles_to_fp(s, radius=self.radius, n_bits=self.n_bits) for s in smiles_list])
        return self.rf.predict(X)

    def predict_many_full(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Full output with per-model and ensemble (kept for UI / debugging)."""
        if not smiles_list:
            return []
        cp = self._predict_cp_many(smiles_list)
        rf = self._predict_rf_many(smiles_list)
        ens = self.w_rf * rf + (1 - self.w_rf) * cp
        out = []
        for s, ycp, yrf, y in zip(smiles_list, cp, rf, ens):
            out.append({
                "ok": True,
                "input": {"smiles": s},
                "per_model": {"chemprop": float(ycp), "rf": float(yrf)},
                "ensemble": {"w_RF": self.w_rf, "prediction": float(y)},
                "unit": self.unit
            })
        return out

    def predict_values(self, smiles_list: List[str], which: str = "ensemble") -> List[float]:
        """
        Return only one value per SMILES.
        which ∈ {'ensemble','chemprop','rf'}
        """
        res = self.predict_many_full(smiles_list)
        vals = []
        for r in res:
            if which == "chemprop":
                vals.append(float(r["per_model"]["chemprop"]))
            elif which == "rf":
                vals.append(float(r["per_model"]["rf"]))
            else:
                vals.append(float(r["ensemble"]["prediction"]))
        return vals

    def predict_csv(self, input_csv: str, smiles_col: str = "canonical_smiles",
                    output_csv: Optional[str] = None, which: str = "ensemble") -> str:
        """Batch predict and save a single chosen column."""
        df = pd.read_csv(input_csv)
        df.columns = df.columns.str.strip()
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in {input_csv}")
        smi = df[smiles_col].astype(str).tolist()
        vals = self.predict_values(smi, which=which)
        out_df = pd.DataFrame({"SMILES": smi, f"Pred_{which.capitalize()}": vals})
        if output_csv is None:
            base, name = os.path.split(input_csv)
            stem = os.path.splitext(name)[0]
            output_csv = os.path.join(base, f"{stem}_predicted_mt_{which}.csv")
        out_df.to_csv(output_csv, index=False)
        return output_csv


# ---------------- CLI ----------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Predict melting temperature (°C) with Chemprop+RF ensemble.")
    ap.add_argument("smiles", nargs="*", help="SMILES strings.")
    ap.add_argument("--csv", type=str, help="CSV file with a SMILES column.")
    ap.add_argument("--smiles-col", type=str, default="canonical_smiles", help="SMILES column name.")
    ap.add_argument("--out", type=str, help="Output CSV path for --csv mode.")
    ap.add_argument("--base-dir", type=str, default="models", help="Base directory of frozen models.")
    ap.add_argument("--version", type=str, default="v1", help="Model version under models/mt_regression.")
    ap.add_argument("--model", type=str, default="ensemble", choices=["ensemble","chemprop","rf"],
                    help="Which predictor to output. Default: ensemble.")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of plain values.")
    args = ap.parse_args()

    model = MTEnsemble(base_dir=args.base_dir, version=args.version)

    # CSV mode: write a single-column prediction file (plus SMILES)
    if args.csv:
        out = model.predict_csv(args.csv, smiles_col=args.smiles_col, output_csv=args.out, which=args.model)
        print(json.dumps({"ok": True, "mode": "csv", "model": args.model, "output_csv": out}) if args.json else out)
        return

    # SMILES list mode: print only one number per SMILES by chosen model
    if args.smiles:
        if args.json:
            vals = model.predict_values(args.smiles, which=args.model)
            print(json.dumps({
                "ok": True,
                "model": args.model,
                "unit": model.unit,
                "inputs": args.smiles,
                "predictions": vals
            }))
        else:
            # plain text, one per line (SMILES \t value)
            vals = model.predict_values(args.smiles, which=args.model)
            for s, v in zip(args.smiles, vals):
                print(f"{s}\t{v:.6f}")
        return

    ap.print_help()


if __name__ == "__main__":
    _main()
