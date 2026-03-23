
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# === Setup ===
train_path = "data/Train_LC_nonLC_2025.csv"
test_path = "data/Test_LC_nonLC_2025.csv"
results_dir = "results_classification-202509"
os.makedirs(results_dir, exist_ok=True)

# === Convert SMILES to fingerprint ===
def smiles_to_fp(smiles, radius=3, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else np.zeros(n_bits)

# === Load data ===
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X_train = np.array([smiles_to_fp(s) for s in df_train["SMILES"]])
y_train = df_train["Label-1"].astype(int).values

X_test = np.array([smiles_to_fp(s) for s in df_test["SMILES"]])
y_test = df_test["Label-1"].astype(int).values

# === Train DNN ===
dnn = Sequential([
    Dense(512, activation='relu', input_dim=X_train.shape[1], kernel_regularizer='l2'),
    BatchNormalization(), Dropout(0.3),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
dnn.compile(loss='binary_crossentropy', optimizer=Adam(0.0005), metrics=['accuracy'])
dnn.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=200, batch_size=64, verbose=1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# === Train SVM and XGBoost ===
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# === Predict function for DNN vs others ===
def get_predictions(model, X):
    if X.shape[0] == 0:
        return np.array([]), np.array([])
    if isinstance(model, Sequential):
        probs = model.predict(X).ravel()
    else:
        probs = model.predict_proba(X)[:, 1]
    return (probs > 0.5).astype(int), probs

# === Predict test set ===
dnn_preds, dnn_probs = get_predictions(dnn, X_test)
svm_preds, svm_probs = get_predictions(svm, X_test)
xgb_preds, xgb_probs = get_predictions(xgb, X_test)

# === Ensemble: any model votes LC => LC ===
ensemble_preds = ((dnn_preds + svm_preds + xgb_preds) >= 1).astype(int)

# === Save ensemble test performance ===
ensemble_metrics = {
    "Accuracy": accuracy_score(y_test, ensemble_preds),
    "Precision": precision_score(y_test, ensemble_preds),
    "Recall": recall_score(y_test, ensemble_preds),
    "F1 Score": f1_score(y_test, ensemble_preds)
}
pd.DataFrame([ensemble_metrics]).to_csv(os.path.join(results_dir, "ensemble_metrics_test.csv"), index=False)

# === Save training set ensemble performance ===
dnn_preds_tr, _ = get_predictions(dnn, X_train)
svm_preds_tr, _ = get_predictions(svm, X_train)
xgb_preds_tr, _ = get_predictions(xgb, X_train)
ensemble_preds_tr = ((dnn_preds_tr + svm_preds_tr + xgb_preds_tr) >= 1).astype(int)
ensemble_metrics_tr = {
    "Accuracy": accuracy_score(y_train, ensemble_preds_tr),
    "Precision": precision_score(y_train, ensemble_preds_tr),
    "Recall": recall_score(y_train, ensemble_preds_tr),
    "F1 Score": f1_score(y_train, ensemble_preds_tr)
}
pd.DataFrame([ensemble_metrics_tr]).to_csv(os.path.join(results_dir, "ensemble_metrics_train.csv"), index=False)

# === Save confusion matrix ===
cm = confusion_matrix(y_test, ensemble_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Non-LC', 'LC'], yticklabels=['Non-LC', 'LC'])
plt.title("Confusion Matrix (Ensemble)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_test.png"))
plt.close()

# === Subtype-wise evaluation ===
conf_matrix_dir = os.path.join(results_dir, "confusion_matrices_subtype")
os.makedirs(conf_matrix_dir, exist_ok=True)

subtype_df = df_test[df_test['Label-2'] != 'NA']
subtype_metrics = []
for subtype in subtype_df['Label-2'].unique():
    sub_data = subtype_df[subtype_df['Label-2'] == subtype]
    if len(sub_data) == 0:
        continue
    X_sub = np.array([smiles_to_fp(s) for s in sub_data['SMILES']])
    y_sub = sub_data['Label-1'].astype(int).values
    if X_sub.shape[0] == 0:
        continue

    dnn_p, _ = get_predictions(dnn, X_sub)
    svm_p, _ = get_predictions(svm, X_sub)
    xgb_p, _ = get_predictions(xgb, X_sub)
    ens_p = ((dnn_p + svm_p + xgb_p) >= 1).astype(int)

    acc = accuracy_score(y_sub, ens_p)
    f1 = f1_score(y_sub, ens_p)
    rec = recall_score(y_sub, ens_p)
    subtype_metrics.append({"Subtype": subtype, "n_samples": len(y_sub), "Accuracy": acc, "F1 Score": f1, "Recall": rec})

    cm_sub = confusion_matrix(y_sub, ens_p)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Purples', xticklabels=['Non-LC', 'LC'], yticklabels=['Non-LC', 'LC'])
    plt.title(f"Confusion Matrix ({subtype})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(conf_matrix_dir, f"confusion_matrix_{subtype.lower().replace(' ', '_')}.png"))
    plt.close()

pd.DataFrame(subtype_metrics).to_csv(os.path.join(results_dir, "subtype_metrics.csv"), index=False)



# --- after training and evaluation ---
import json
from joblib import dump

TASK_DIR = os.path.join("models", "lc_classification", "v1")
os.makedirs(TASK_DIR, exist_ok=True)

# 1) DNN
dnn.save(os.path.join(TASK_DIR, "dnn.h5"))
# 2) SVM
dump(svm, os.path.join(TASK_DIR, "svm.joblib"))
# 3) XGB
xgb.save_model(os.path.join(TASK_DIR, "xgb.json"))

# 4) meta
meta = {
    "task": "classification",
    "name": "lc_classification",
    "version": "v1",
    "fingerprint": {"type": "Morgan", "radius": 3, "n_bits": 2048},
    "labels": {"positive": "LC", "negative": "Non-LC", "positive_id": 1},
    "ensemble": {"voting_threshold": 1, "members": ["dnn", "svm", "xgb"]},
}
with open(os.path.join(TASK_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f" Saved models to {TASK_DIR}")
