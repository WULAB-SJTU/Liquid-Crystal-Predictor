import os
import json
import numpy as np
from typing import Dict, Any, List, Optional

# Quiet RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit.Chem import AllChem

from joblib import load as joblib_load
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model as keras_load_model


class ModelRegistry:
    """Resolve model directory paths."""
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir

    def path(self, task_name: str, version: str) -> str:
        return os.path.join(self.base_dir, task_name, version)


class LCEnsemble:


    def __init__(self, base_dir: str = "models",
                 task_name: str = "lc_classification",
                 version: str = "v1"):
        self.task_name = task_name
        self.version = version
        self.registry = ModelRegistry(base_dir)
        self.model_dir = self.registry.path(task_name, version)

        # Defaults before meta is read
        self.radius = 3
        self.n_bits = 2048
        self.threshold_cfg = 1
        self.pos_label_id = 1

        # Containers
        self.dnn = None
        self.svm = None
        self.xgb = None
        self.load_errors: List[str] = []

        # Load meta, then members; never raise from __init__
        self._load_meta_safe()
        self._load_members_safe()

    # --------------------- Loading ---------------------
    def _load_meta_safe(self) -> None:
        """Load meta.json if available; keep defaults otherwise."""
        meta_path = os.path.join(self.model_dir, "meta.json")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            fp = meta.get("fingerprint", {})
            self.radius = int(fp.get("radius", self.radius))
            self.n_bits = int(fp.get("n_bits", self.n_bits))
            self.threshold_cfg = int(meta.get("ensemble", {}).get("voting_threshold", self.threshold_cfg))
            self.pos_label_id = int(meta.get("labels", {}).get("positive_id", self.pos_label_id))
        except FileNotFoundError:
            self.load_errors.append(f"meta.json not found at {meta_path}")
        except Exception as e:
            self.load_errors.append(f"Failed to load meta.json: {e}")

    def _load_members_safe(self) -> None:
        """Load DNN/SVM/XGB individually; if missing, keep None and record error."""
        # DNN
        dnn_path = os.path.join(self.model_dir, "dnn.h5")
        try:
            if os.path.exists(dnn_path):
                self.dnn = keras_load_model(dnn_path)
            else:
                self.load_errors.append(f"DNN missing: {dnn_path}")
        except Exception as e:
            self.load_errors.append(f"DNN load failed: {e}")
            self.dnn = None

        # SVM
        svm_path = os.path.join(self.model_dir, "svm.joblib")
        try:
            if os.path.exists(svm_path):
                self.svm = joblib_load(svm_path)
            else:
                self.load_errors.append(f"SVM missing: {svm_path}")
        except Exception as e:
            self.load_errors.append(f"SVM load failed: {e}")
            self.svm = None

        # XGB
        xgb_path = os.path.join(self.model_dir, "xgb.json")
        try:
            if os.path.exists(xgb_path):
                self.xgb = XGBClassifier()
                self.xgb.load_model(xgb_path)
            else:
                self.load_errors.append(f"XGB missing: {xgb_path}")
        except Exception as e:
            self.load_errors.append(f"XGB load failed: {e}")
            self.xgb = None

    # --------------------- Utils ---------------------
    def _available_members(self) -> List[str]:
        """Return the list of actually loaded members."""
        members = []
        if self.dnn is not None: members.append("dnn")
        if self.svm is not None: members.append("svm")
        if self.xgb is not None: members.append("xgb")
        return members

    def _effective_threshold(self, n_members: int) -> int:
        """Clamp voting threshold to [1, n_members] to avoid impossible conditions."""
        if n_members <= 0:
            return 1
        return max(1, min(self.threshold_cfg, n_members))

    def smiles_to_fp(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to Morgan fingerprint; return None on failure."""
        try:
            s = (smiles or "").strip()
            if not s:
                return None
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None
            # Canonicalize (optional, helps consistency)
            _ = Chem.MolToSmiles(mol, canonical=True)
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.fromiter((bv[i] for i in range(self.n_bits)), dtype=np.int8, count=self.n_bits)
            return arr
        except Exception:
            return None

    # --------------------- Inference ---------------------
    def predict_one(self, smiles: str) -> Dict[str, Any]:
        """Predict one SMILES; never raises."""
        avail = self._available_members()
        if len(avail) == 0:
            return {
                "ok": False,
                "error": "No model loaded. "
                         f"Checked {self.model_dir}. Errors: {self.load_errors}",
                "model": {"task": self.task_name, "version": self.version}
            }

        x = self.smiles_to_fp(smiles)
        if x is None:
            return {"ok": False, "error": "Invalid or empty SMILES.", "input": {"smiles": smiles}}

        x = x.reshape(1, -1)

        per_model: Dict[str, Any] = {}
        probs = []
        votes = []

        # DNN
        if self.dnn is not None:
            try:
                p = float(self.dnn.predict(x, verbose=0).ravel()[0])
                per_model["dnn_prob"] = p
                probs.append(p); votes.append(int(p >= 0.5))
            except Exception as e:
                per_model["dnn_error"] = f"predict failed: {e}"

        # SVM
        if self.svm is not None:
            try:
                p = float(self.svm.predict_proba(x)[0, 1])
                per_model["svm_prob"] = p
                probs.append(p); votes.append(int(p >= 0.5))
            except Exception as e:
                per_model["svm_error"] = f"predict_proba failed: {e}"

        # XGB
        if self.xgb is not None:
            try:
                p = float(self.xgb.predict_proba(x)[0, 1])
                per_model["xgb_prob"] = p
                probs.append(p); votes.append(int(p >= 0.5))
            except Exception as e:
                per_model["xgb_error"] = f"predict_proba failed: {e}"

        n_effective = len([v for v in votes if isinstance(v, (int, np.integer))])
        if n_effective == 0:
            return {"ok": False, "error": "No valid model outputs.", "detail": per_model}

        thresh = self._effective_threshold(n_effective)
        votes_sum = int(sum(votes))
        label = int(votes_sum >= thresh)

        return {
            "ok": True,
            "model": {"task": self.task_name, "version": self.version},
            "input": {"smiles": smiles},
            "fingerprint": {"radius": self.radius, "n_bits": self.n_bits},
            "per_model": per_model,
            "ensemble": {
                "members_used": n_effective,
                "voting_threshold": thresh,
                "votes_sum": votes_sum,
                "label": label,
                "label_text": "LC" if label == self.pos_label_id else "Non-LC"
            },
            "warnings": self.load_errors  # expose any load-time issues
        }

    def predict_many(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction; never raises."""
        out = []
        for s in smiles_list:
            try:
                out.append(self.predict_one(s))
            except Exception as e:
                out.append({"ok": False, "error": f"Unhandled error: {e}", "input": {"smiles": s}})
        return out



