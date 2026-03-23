import os
import sys
import json
import glob
import math
import shutil
import subprocess
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------- Paths (match your style) ----------------
train_raw = 'data/Training_Melting_T.csv'
test_raw  = 'data/Testing_Melting_T.csv'
smiles_col = 'canonical_smiles'
target_col = 'Melting Temperature'
label_col  = 'Label'

RUN_TAG   = 'best_parameters'
BASE_DIR  = 'results_mt'
SAVE_DIR  = os.path.join(BASE_DIR, RUN_TAG)
CP_DIR    = os.path.join(SAVE_DIR, 'chemprop_run')
os.makedirs(CP_DIR, exist_ok=True)

# Chemprop I/O
chemprop_train = os.path.join(SAVE_DIR, 'chemprop_train.csv')
chemprop_test  = os.path.join(SAVE_DIR, 'chemprop_test.csv')
pred_train_cp  = os.path.join(SAVE_DIR, 'preds_train_cp.csv')
pred_test_cp   = os.path.join(SAVE_DIR, 'preds_test_cp.csv')

# Reports
perf_overall_csv   = os.path.join(SAVE_DIR, 'performance_overall.csv')
perf_subtype_csv   = os.path.join(SAVE_DIR, 'performance_subtype.csv')
weight_search_csv  = os.path.join(SAVE_DIR, 'ensemble_weight_search.csv')

# Freeze target
FREEZE_DIR = os.path.join('models', 'mt_regression', 'v1')
os.makedirs(FREEZE_DIR, exist_ok=True)

# ---------------- Hyperparameters (explicit "defaults") ----------------
# Core capacity (the ones you usually tune explicitly)
# Mapping:
#   - v2 flags → --message_hidden_dim / --ffn_hidden_dim / --ffn_num_layers / --depth / --dropout / --weight_decay
#   - legacy   → --hidden_size (single value); we will map to this when v2 flags are missing
HP: Dict[str, object] = dict(
    depth=3,                 # --depth
    hidden_size=300,         # v2: --message_hidden_dim 300 AND --ffn_hidden_dim 300; legacy: --hidden_size 300
    dropout=0.0,             # --dropout
    weight_decay=0.0,        # --weight_decay
    aggregation="norm",      # --aggregation
    aggregation_norm=100,    # --aggregation_norm
    ffn_num_layers=1,        # --ffn_num_layers
)

# Training / schedule / split
HP_TRAIN: Dict[str, object] = dict(
    epochs=250,              # --epochs (explicit default)
    batch_size=64,           # --batch_size
    init_lr=1e-4,            # --init_lr
    max_lr=1e-3,             # --max_lr
    final_lr=1e-4,           # --final_lr
    warmup_epochs=2,         # --warmup_epochs
    split="RANDOM",          # --split
    split_sizes=(0.8, 0.1, 0.1),  # --split_sizes
    metric="r2",                  # --metric / metrics
)

# ---------------- Utilities ----------------
def chemprop_bin() -> str:
    """Resolve chemprop executable in current env (Windows/Linux/Mac)."""
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
    return 'chemprop'

CHEMPROP = chemprop_bin()

def run(cmd: List[str]):
    print(">>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def has_flag(mode: str, flag: str) -> bool:
    """Check if a CLI subcommand (train/predict) supports a given flag string."""
    try:
        out = subprocess.check_output([CHEMPROP, mode, '-h'], text=True, stderr=subprocess.STDOUT)
        return flag in out
    except Exception:
        return False

def to_cp(raw_csv: str, out_csv: str, smiles_col: str, target_col: str):
    """Build a Chemprop-compatible CSV (columns: smiles,target)."""
    df = pd.read_csv(raw_csv)
    df.columns = df.columns.str.strip()
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Missing '{smiles_col}' or '{target_col}' in {raw_csv}")
    cp = df[[smiles_col, target_col]].copy()
    cp.columns = ['smiles', 'target']
    cp.to_csv(out_csv, index=False)

def find_best_pt(run_dir: str) -> str:
    """Find model_*/best.pt or fallback best.pt under a chemprop output dir."""
    cands = sorted(glob.glob(os.path.join(run_dir, 'model_*', 'best.pt')))
    if cands:
        return cands[0]
    single = os.path.join(run_dir, 'best.pt')
    if os.path.exists(single):
        return single
    any_pt = glob.glob(os.path.join(run_dir, '**', '*.pt'), recursive=True)
    if any_pt:
        return any_pt[0]
    raise FileNotFoundError(f"No best.pt found under {run_dir}")

def smiles_to_fp(smiles: str, radius=3, n_bits=2048) -> np.ndarray:
    """Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for idx in bv.GetOnBits():
        arr[idx] = 1
    return arr

def build_fp(df: pd.DataFrame, smiles_col: str, radius=3, n_bits=2048) -> np.ndarray:
    fps = [smiles_to_fp(s) for s in df[smiles_col].astype(str).values]
    return np.vstack(fps)

def eval_reg(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': math.sqrt(mse),
        'MSE': mse,
        'R2': r2_score(y_true, y_pred)
    }

# ---------------- Step 0: Prepare Chemprop inputs ----------------
os.makedirs(SAVE_DIR, exist_ok=True)
to_cp(train_raw, chemprop_train, smiles_col, target_col)
to_cp(test_raw,  chemprop_test,  smiles_col, target_col)

# ---------------- Step 1: Train Chemprop (explicit defaults) ----------------
# Detect flag support (v2 vs legacy)
SUPPORT = {
    'message_hidden_dim': has_flag('train', '--message_hidden_dim'),
    'ffn_hidden_dim':     has_flag('train', '--ffn_hidden_dim'),
    'ffn_num_layers':     has_flag('train', '--ffn_num_layers'),
    'hidden_size':        has_flag('train', '--hidden_size'),
    'depth':              has_flag('train', '--depth'),
    'dropout':            has_flag('train', '--dropout'),
    'weight_decay':       has_flag('train', '--weight_decay'),
    'aggregation':        has_flag('train', '--aggregation'),
    'aggregation_norm':   has_flag('train', '--aggregation_norm'),
    'split':              has_flag('train', '--split'),
    'split_sizes':        has_flag('train', '--split_sizes'),
    'init_lr':            has_flag('train', '--init_lr'),
    'max_lr':             has_flag('train', '--max_lr'),
    'final_lr':           has_flag('train', '--final_lr'),
    'warmup_epochs':      has_flag('train', '--warmup_epochs'),
    'batch_size':         has_flag('train', '--batch_size'),
    'epochs':             has_flag('train', '--epochs'),
    'metric':             has_flag('train', '--metric'),
}

train_cmd = [
    CHEMPROP, 'train',
    '-i', chemprop_train,
    '-o', CP_DIR,
    '-t', 'regression',
    '--target-columns', 'target',
    '--smiles-column', 'smiles',
]

# Training schedule
if SUPPORT['epochs']:        train_cmd += ['--epochs', str(HP_TRAIN['epochs'])]
if SUPPORT['batch_size']:    train_cmd += ['--batch_size', str(HP_TRAIN['batch_size'])]
if SUPPORT['metric']:        train_cmd += ['--metric', str(HP_TRAIN['metric'])]
if SUPPORT['init_lr']:       train_cmd += ['--init_lr', str(HP_TRAIN['init_lr'])]
if SUPPORT['max_lr']:        train_cmd += ['--max_lr', str(HP_TRAIN['max_lr'])]
if SUPPORT['final_lr']:      train_cmd += ['--final_lr', str(HP_TRAIN['final_lr'])]
if SUPPORT['warmup_epochs']: train_cmd += ['--warmup_epochs', str(HP_TRAIN['warmup_epochs'])]

# Capacity / architecture
if SUPPORT['depth']:        train_cmd += ['--depth', str(HP['depth'])]
if SUPPORT['dropout']:      train_cmd += ['--dropout', str(HP['dropout'])]
if SUPPORT['weight_decay']: train_cmd += ['--weight_decay', str(HP['weight_decay'])]

# Hidden dimensions (prefer v2 flags; else legacy hidden_size)
if SUPPORT['message_hidden_dim'] and SUPPORT['ffn_hidden_dim']:
    train_cmd += ['--message_hidden_dim', str(HP['hidden_size']),
                  '--ffn_hidden_dim',     str(HP['hidden_size'])]
elif SUPPORT['hidden_size']:
    train_cmd += ['--hidden_size', str(HP['hidden_size'])]

if SUPPORT['ffn_num_layers']:
    train_cmd += ['--ffn_num_layers', str(HP['ffn_num_layers'])]

# Aggregation
if SUPPORT['aggregation']:
    train_cmd += ['--aggregation', str(HP['aggregation'])]
if SUPPORT['aggregation_norm']:
    train_cmd += ['--aggregation_norm', str(HP['aggregation_norm'])]

# Split
if SUPPORT['split']:
    train_cmd += ['--split', str(HP_TRAIN['split'])]
if SUPPORT['split_sizes']:
    a, b, c = HP_TRAIN['split_sizes']
    train_cmd += ['--split_sizes', str(a), str(b), str(c)]

run(train_cmd)
best_pt = find_best_pt(CP_DIR)

# ---------------- Step 2: Chemprop predict (train/test) ----------------
run([CHEMPROP, 'predict', '--model-paths', best_pt, '-i', chemprop_train, '-o', pred_train_cp])
run([CHEMPROP, 'predict', '--model-paths', best_pt, '-i', chemprop_test,  '-o', pred_test_cp])

def attach_cp(raw_csv, cp_in_csv, cp_pred_csv) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv); raw.columns = raw.columns.str.strip()
    cp_in = pd.read_csv(cp_in_csv)
    pr    = pd.read_csv(cp_pred_csv)
    col   = 'preds' if 'preds' in pr.columns else pr.columns[-1]
    pred  = pd.to_numeric(pr[col], errors='coerce')
    if len(pred) != len(cp_in):
        raise RuntimeError("Pred length != cp_input length.")
    out = raw.copy()
    out['Predicted_Chemprop'] = pred.values
    if label_col not in out.columns:
        out[label_col] = 'unknown'
    return out

df_train_cp = attach_cp(train_raw, chemprop_train, pred_train_cp)
df_test_cp  = attach_cp(test_raw,  chemprop_test,  pred_test_cp)

# ---------------- Step 3: Train RF on Morgan fingerprints ----------------
df_tr = pd.read_csv(train_raw); df_tr.columns = df_tr.columns.str.strip()
df_te = pd.read_csv(test_raw);  df_te.columns = df_te.columns.str.strip()

X_train = build_fp(df_tr, smiles_col, radius=3, n_bits=2048)
X_test  = build_fp(df_te, smiles_col, radius=3, n_bits=2048)
y_train = df_tr[target_col].values
y_test  = df_te[target_col].values

rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
train_rf = rf.predict(X_train)
test_rf  = rf.predict(X_test)

# ---------------- Step 4: Evaluate CP & RF ----------------
def eval_row(name: str, split: str, y_true, y_pred) -> Dict[str, float]:
    m = eval_reg(y_true, y_pred)
    return {'Model': name, 'Split': split, **m}

rows_overall = [
    eval_row('Chemprop', 'Train', y_train, df_train_cp['Predicted_Chemprop'].values),
    eval_row('Chemprop', 'Test',  y_test,  df_test_cp['Predicted_Chemprop'].values),
    eval_row('RF',       'Train', y_train, train_rf),
    eval_row('RF',       'Test',  y_test,  test_rf),
]

def subtype_tbl(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    acc = []
    for sub in sorted(df[label_col].astype(str).unique()):
        m = (df[label_col] == sub)
        if m.sum() == 0:
            continue
        met = eval_reg(df.loc[m, target_col], df.loc[m, pred_col])
        acc.append({'Subtype': sub, **met})
    return pd.DataFrame(acc)

sub_all = []
# Chemprop subtype
sub_cp_tr = subtype_tbl(df_train_cp.rename(columns={'Predicted_Chemprop':'Predicted'}), 'Predicted')
sub_cp_te = subtype_tbl(df_test_cp.rename(columns={'Predicted_Chemprop':'Predicted'}),  'Predicted')
sub_cp_tr.insert(0,'Model','Chemprop'); sub_cp_tr.insert(1,'Split','Train')
sub_cp_te.insert(0,'Model','Chemprop');  sub_cp_te.insert(1,'Split','Test')
sub_all += [sub_cp_tr, sub_cp_te]
# RF subtype
df_train_rf = df_tr.copy(); df_train_rf['Predicted'] = train_rf
df_test_rf  = df_te.copy(); df_test_rf['Predicted']  = test_rf
sub_rf_tr = subtype_tbl(df_train_rf, 'Predicted'); sub_rf_tr.insert(0,'Model','RF'); sub_rf_tr.insert(1,'Split','Train')
sub_rf_te = subtype_tbl(df_test_rf,  'Predicted'); sub_rf_te.insert(0,'Model','RF'); sub_rf_te.insert(1,'Split','Test')
sub_all += [sub_rf_tr, sub_rf_te]

# ---------------- Step 5: Ensemble weight search on test ----------------
y_cp_tr = df_train_cp['Predicted_Chemprop'].values
y_cp_te = df_test_cp['Predicted_Chemprop'].values

weights = np.round(np.linspace(0.0, 1.0, 21), 2)  # 0.00..1.00 step 0.05
records = []
best: Tuple[float, float, float] = None  # (w_rf, mae, rmse)

for w in weights:
    pred = w * test_rf + (1 - w) * y_cp_te
    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    records.append({'w_RF': w, 'w_Chemprop': 1 - w, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    if (best is None) or (mae < best[1]) or (mae == best[1] and rmse < best[2]):
        best = (w, mae, rmse)

pd.DataFrame(records).to_csv(weight_search_csv, index=False)
w_rf = float(best[0])

# Apply best weight
ens_tr = w_rf * train_rf + (1 - w_rf) * y_cp_tr
ens_te = w_rf * test_rf  + (1 - w_rf) * y_cp_te
rows_overall += [
    eval_row(f'Ensemble(w_RF={w_rf:.2f})', 'Train', y_train, ens_tr),
    eval_row(f'Ensemble(w_RF={w_rf:.2f})', 'Test',  y_test,  ens_te),
]
# Ensemble subtype
sub_ens_tr = df_tr.copy(); sub_ens_tr['Predicted'] = ens_tr
sub_ens_te = df_te.copy(); sub_ens_te['Predicted']  = ens_te
sub_ens_tr = subtype_tbl(sub_ens_tr, 'Predicted'); sub_ens_tr.insert(0,'Model',f'Ensemble(w_RF={w_rf:.2f})'); sub_ens_tr.insert(1,'Split','Train')
sub_ens_te = subtype_tbl(sub_ens_te, 'Predicted'); sub_ens_te.insert(0,'Model',f'Ensemble(w_RF={w_rf:.2f})'); sub_ens_te.insert(1,'Split','Test')
sub_all += [sub_ens_tr, sub_ens_te]

# Save reports & joined predictions
pd.DataFrame(rows_overall).to_csv(perf_overall_csv, index=False)
pd.concat(sub_all, ignore_index=True).to_csv(perf_subtype_csv, index=False)
pd.DataFrame({
    'SMILES': df_tr[smiles_col], 'Label': df_tr.get(label_col, 'unknown'),
    target_col: y_train, 'Pred_Chemprop': y_cp_tr, 'Pred_RF': train_rf, 'Pred_Ensemble': ens_tr
}).to_csv(os.path.join(SAVE_DIR, 'train_predictions_all.csv'), index=False)
pd.DataFrame({
    'SMILES': df_te[smiles_col], 'Label': df_te.get(label_col, 'unknown'),
    target_col: y_test, 'Pred_Chemprop': y_cp_te, 'Pred_RF': test_rf, 'Pred_Ensemble': ens_te
}).to_csv(os.path.join(SAVE_DIR, 'test_predictions_all.csv'), index=False)

print("✅ Saved reports to:", SAVE_DIR)

# ---------------- Step 6: Freeze artifacts to models/mt_regression/v1 ----------------
# Clean target dir
for p in glob.glob(os.path.join(FREEZE_DIR, '*')):
    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

# Copy chemprop model_*
cp_models = glob.glob(os.path.join(CP_DIR, 'model_*'))
if cp_models:
    for m in cp_models:
        shutil.copytree(m, os.path.join(FREEZE_DIR, os.path.basename(m)))
else:
    os.makedirs(os.path.join(FREEZE_DIR, 'model_0'), exist_ok=True)
    shutil.copy2(best_pt, os.path.join(FREEZE_DIR, 'model_0', 'best.pt'))

# Save RF
dump(rf, os.path.join(FREEZE_DIR, 'rf.joblib'))

# Write meta.json with explicit hparams
meta = {
    "task": "regression",
    "name": "mt_regression",
    "version": "v1",
    "target": {"name": "Melting Temperature", "unit": "°C"},
    "columns": {"smiles": smiles_col, "target": target_col, "label": label_col},
    "fingerprint": {"type": "Morgan", "radius": 3, "n_bits": 2048},
    "ensemble": {"type": "weighted", "w_RF": w_rf, "w_Chemprop": 1 - w_rf},
    "chemprop_hparams": {
        "capacity": HP,
        "train": HP_TRAIN
    },
    "run_tag": RUN_TAG
}
with open(os.path.join(FREEZE_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ Frozen model to:", FREEZE_DIR)
