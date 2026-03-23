# ct_fixed_locked.py
# Train Chemprop regression with fixed best hyperparams (no sweep),
# using the same CLI capability detection pattern as chemprop_sweep_locked.py.

import os, sys, math, json, shutil, subprocess
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== User config (fixed inputs) ==================
train_raw = 'data/ClearingT_Training.csv'
test_raw  = 'data/ClearingT_Testing.csv'
smiles_col = 'SMILES'
target_col = 'Clearing Temperature'
label_col  = 'Label'  # optional

# Fixed best hyperparameters you chose: d7_h1600_dr0p1_wd1em06
HP = dict(depth=7, hidden_size=1600, dropout=0.1, weight_decay=1e-6)

# Outputs
base_save_dir = 'results_ct'
tag = f"d{HP['depth']}_h{HP['hidden_size']}_dr{str(HP['dropout']).replace('.','p')}_wd{str(HP['weight_decay']).replace('e-','em')}"
save_dir = os.path.join(base_save_dir, tag)
os.makedirs(save_dir, exist_ok=True)

chemprop_train = os.path.join(save_dir, 'chemprop_train.csv')
chemprop_test  = os.path.join(save_dir, 'chemprop_test.csv')
train_preds = os.path.join(save_dir, 'preds_train.csv')
test_preds  = os.path.join(save_dir, 'preds_test.csv')

# Freeze destination
freeze_dir = os.path.join('models', 'ct_regression', 'v1')
os.makedirs(freeze_dir, exist_ok=True)

# ================== Helpers (same style) ==================
def chemprop_bin() -> str:
    exe = os.path.join(sys.prefix, 'Scripts', 'chemprop.exe')
    if os.path.exists(exe): return exe
    exe2 = os.path.join(sys.prefix, 'Scripts', 'chemprop')
    if os.path.exists(exe2): return exe2
    return 'chemprop'
CHEMPROP_BIN = chemprop_bin()

def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def has_flag(flag: str, mode: str) -> bool:
    try:
        out = subprocess.check_output([CHEMPROP_BIN, mode, '-h'], text=True, stderr=subprocess.STDOUT)
        return flag in out
    except Exception:
        return False

def to_cp_csv(raw, out):
    df = pd.read_csv(raw)
    df.columns = df.columns.str.strip()
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Missing columns in {raw}. Need '{smiles_col}' and '{target_col}'.")
    cp = df[[smiles_col, target_col]].copy()
    cp.columns = ['smiles', 'target']
    cp.to_csv(out, index=False)

def merge_preds(cp_csv, preds_csv, raw_csv):
    cp = pd.read_csv(cp_csv)     # smiles,target
    pr = pd.read_csv(preds_csv)  # 'preds' or last column
    pred_col = 'preds' if 'preds' in pr.columns else pr.columns[-1]
    raw = pd.read_csv(raw_csv); raw.columns = raw.columns.str.strip()
    raw_map = raw.set_index(smiles_col)
    rows = []
    for smi, tgt, pred in zip(cp['smiles'], cp['target'], pr[pred_col].astype(float)):
        r = raw_map.loc[smi]
        rows.append({
            'SMILES': smi,
            'Label': r[label_col] if label_col in raw_map.columns else 'unknown',
            'true': r[target_col] if target_col in raw_map.columns else tgt,
            'pred': pred
        })
    return pd.DataFrame(rows)

def evaluate(df):
    y_true, y_pred = df['true'].values, df['pred'].values
    mse = mean_squared_error(y_true, y_pred)
    return {'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': math.sqrt(mse), 'MSE': mse, 'R2': r2_score(y_true, y_pred)}

def plot_train_test_fixed(df_tr, df_te, out_png, title='Chemprop_GNN'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, df, name in zip(axes, [df_tr, df_te], ['Train', 'Test']):
        for cls, color in zip(['rod-like', 'disc-like', 'bend-core'], ['#2ca02c','#ff7f0e','#1f77b4']):
            m = (df.get('Label','') == cls)
            if m.any():
                ax.scatter(df.loc[m,'true'], df.loc[m,'pred'], alpha=0.85, label=cls,
                           color=color, edgecolors='white', linewidths=0.4, s=28)
        mn = min(df_tr['true'].min(), df_te['true'].min(), df_tr['pred'].min(), df_te['pred'].min())
        mx = max(df_tr['true'].max(), df_te['true'].max(), df_tr['pred'].max(), df_te['pred'].max())
        pad = 10
        ax.plot([mn-pad,mx+pad],[mn-pad,mx+pad],'k--',linewidth=1.1)
        ax.set_xlim([mn-pad,mx+pad]); ax.set_ylim([mn-pad,mx+pad])
        ax.set_title(f'{name} Set'); ax.set_xlabel('True (°C)'); ax.set_ylabel('Predicted (°C)')
        ax.legend()
    fig.suptitle(f'{title} Prediction by Class (Fixed Axis)')
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

def freeze_models(src_dir: str, dst_dir: str, meta: dict):
    # clean dst
    if os.path.exists(dst_dir):
        for p in glob(os.path.join(dst_dir, '*')):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(dst_dir, exist_ok=True)
    # copy model_* (or best.pt)
    copied = False
    for mdir in sorted(glob(os.path.join(src_dir, 'model_*'))):
        shutil.copytree(mdir, os.path.join(dst_dir, os.path.basename(mdir))); copied = True
    best_root = os.path.join(src_dir, 'best.pt')
    if not copied and os.path.exists(best_root):
        shutil.copy2(best_root, os.path.join(dst_dir, 'best.pt')); copied = True
    if not copied:
        raise FileNotFoundError("No model artifacts found to freeze.")
    with open(os.path.join(dst_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Frozen to {dst_dir}")

# ================== Capability detection ==================
NEW = {
    'data_path':      has_flag('--data_path', 'train'),
    'target_columns': has_flag('--target_columns', 'train'),
    'smiles_column':  has_flag('--smiles_column', 'train'),
}
SUPPORTS = {
    'checkpoint_dir':        has_flag('--checkpoint_dir', 'predict'),
    'smiles_column_predict': has_flag('--smiles_column', 'predict'),
    'num_workers_train':     has_flag('--num_workers', 'train'),
    'num_workers_predict':   has_flag('--num_workers', 'predict'),
}
CAN = {
    'split_type':         has_flag('--split_type', 'train'),
    'target_scaler':      has_flag('--target_scaler', 'train'),
    'features_generator': has_flag('--features_generator', 'train'),
    'ffn_num_layers':     has_flag('--ffn_num_layers', 'train'),
    'batch_size':         has_flag('--batch_size', 'train'),
    'epochs':             has_flag('--epochs', 'train'),
    'metric':             has_flag('--metric', 'train'),
    'loss_function':      has_flag('--loss_function', 'train'),
    'init_lr':            has_flag('--init_lr', 'train'),
    'max_lr':             has_flag('--max_lr', 'train'),
    'warmup_steps':       has_flag('--warmup_steps', 'train'),
    'depth':              has_flag('--depth', 'train'),
    'hidden_size':        has_flag('--hidden_size', 'train'),
    'dropout':            has_flag('--dropout', 'train'),
    'weight_decay':       has_flag('--weight_decay', 'train'),
}

_cpu = os.cpu_count() or 8
AUTO_NUM_WORKERS = max(4, min(16, _cpu - 2))
use_new_cli = all(NEW.values())
print(f"\n[Info] chemprop: {CHEMPROP_BIN}")
print(f"[Info] New CLI detected? {use_new_cli}")
print(f"[Info] '--num_workers' support: train={SUPPORTS['num_workers_train']}, predict={SUPPORTS['num_workers_predict']}")

# ================== Prepare inputs ==================
to_cp_csv(train_raw, chemprop_train)
to_cp_csv(test_raw,  chemprop_test)

# ================== Train (no sweep; fixed HP) ==================
if use_new_cli:
    train_cmd = [
        CHEMPROP_BIN, 'train',
        '--data_path', chemprop_train,
        '--separate_test_path', chemprop_test,
        '--dataset_type', 'regression',
        '--target_columns', 'target',
        '--smiles_column', 'smiles',
        '--save_dir', save_dir,
    ]
else:
    train_cmd = [CHEMPROP_BIN, 'train', '-i', chemprop_train, '-o', save_dir, '-t', 'regression']

# optional knobs if supported by this build
if CAN['split_type']:         train_cmd += ['--split_type', 'scaffold']
if CAN['target_scaler']:      train_cmd += ['--target_scaler', 'standard']
if CAN['features_generator']: train_cmd += ['--features_generator', 'rdkit_2d_normalized']
if CAN['epochs']:             train_cmd += ['--epochs', '500']
if CAN['batch_size']:         train_cmd += ['--batch_size', '64']
if CAN['ffn_num_layers']:     train_cmd += ['--ffn_num_layers', '3']
if CAN['metric']:             train_cmd += ['--metric', 'r2']
if CAN['loss_function']:      train_cmd += ['--loss_function', 'mae']
if CAN['init_lr']:            train_cmd += ['--init_lr', '1e-4']
if CAN['max_lr']:             train_cmd += ['--max_lr', '1e-3']
if CAN['warmup_steps']:       train_cmd += ['--warmup_steps', '2000']

# fixed best HP (only add if supported)
if CAN['depth']:              train_cmd += ['--depth', str(HP['depth'])]
if CAN['hidden_size']:        train_cmd += ['--hidden_size', str(HP['hidden_size'])]
if CAN['dropout']:            train_cmd += ['--dropout', str(HP['dropout'])]
if CAN['weight_decay']:       train_cmd += ['--weight_decay', str(HP['weight_decay'])]

# workers
if SUPPORTS['num_workers_train']:
    train_cmd += ['--num_workers', str(AUTO_NUM_WORKERS)]

run_cmd(train_cmd)

# ================== Predict (train & test) ==================
if SUPPORTS['checkpoint_dir']:
    pred_train = [CHEMPROP_BIN, 'predict', '--checkpoint_dir', save_dir,
                  '--test_path', chemprop_train, '--preds_path', train_preds]
    pred_test  = [CHEMPROP_BIN, 'predict', '--checkpoint_dir', save_dir,
                  '--test_path', chemprop_test,  '--preds_path', test_preds]
    if SUPPORTS['smiles_column_predict']:
        pred_train += ['--smiles_column', 'smiles']
        pred_test  += ['--smiles_column', 'smiles']
else:
    model0 = os.path.join(save_dir, 'model_0', 'best.pt')
    if not os.path.exists(model0):
        model0 = os.path.join(save_dir, 'best.pt')
    assert os.path.exists(model0), f"best.pt not found under {save_dir}"
    pred_train = [CHEMPROP_BIN, 'predict',
                  '--model-paths', model0,
                  '-i', chemprop_train,
                  '-o', train_preds]

    pred_test = [CHEMPROP_BIN, 'predict',
                 '--model-paths', model0,
                 '-i', chemprop_test,
                 '-o', test_preds]

    if SUPPORTS['smiles_column_predict']:
        pred_train += ['--smiles_column', 'smiles']
        pred_test  += ['--smiles_column', 'smiles']

if SUPPORTS['num_workers_predict']:
    pred_train += ['--num_workers', str(AUTO_NUM_WORKERS)]
    pred_test  += ['--num_workers',  str(AUTO_NUM_WORKERS)]

run_cmd(pred_train)
run_cmd(pred_test)

# ================== Merge & evaluate ==================
df_tr = merge_preds(chemprop_train, train_preds, train_raw)
df_te = merge_preds(chemprop_test,  test_preds,  test_raw)
df_tr.to_csv(os.path.join(save_dir, 'train_predictions.csv'), index=False)
df_te.to_csv(os.path.join(save_dir, 'test_predictions.csv'),  index=False)

mtr = evaluate(df_tr); mte = evaluate(df_te)
pd.DataFrame([{
    'Model': f"Chemprop_GNN({tag})",
    'Train MAE': mtr['MAE'], 'Train RMSE': mtr['RMSE'], 'Train MSE': mtr['MSE'], 'Train R2': mtr['R2'],
    'Test MAE':  mte['MAE'], 'Test RMSE':  mte['RMSE'],  'Test MSE':  mte['MSE'],  'Test R2':  mte['R2'],
}]).to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)

# Plot
plot_train_test_fixed(df_tr, df_te, os.path.join(save_dir, 'Chemprop_GNN_prediction_fixed_axis.png'))

# ================== Freeze to models/ct_regression/v1 ==================
meta = {
    "task": "regression",
    "name": "ct_regression",
    "version": "v1",
    "target": {"name": target_col, "unit": "°C"},
    "chemprop_best_setup": tag,
    "columns": {"smiles": "smiles", "target": "target"},
    "note": "Frozen from fixed-HP training (no sweep)."
}
freeze_models(save_dir, freeze_dir, meta)

print("\nDone. Check:", save_dir, "and", freeze_dir)
