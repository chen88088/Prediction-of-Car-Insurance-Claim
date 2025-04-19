# claim_model_optimized.py with GPU support and MLflow-ready outputs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# [0] 建立輸出資料夾（依照時間命名）
project_name = "lightgbmoptimization"
# [0] 建立輸出資料夾（依照時間命名）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("..", "OUTPUT", f"{project_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# [1] 讀取資料
print("\n[1] 讀取資料...")
df = pd.read_csv("../DATA/train.csv")
X = df.drop(columns=["policy_id", "is_claim"])
y = df["is_claim"]

# [2] 特徵工程：Label Encoding + 數值標準化
print("[2] 前處理資料...")
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))

# [3] 切分資料
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# [4] 訓練 LightGBM 最佳模型（使用固定參數模擬最佳模型）
print("[3] 訓練最佳 LightGBM 模型...")
scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
params = {
    'objective': 'binary',
    'metric': ['auc'],
    'device': 'gpu',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 10,
    'min_child_samples': 20,
    'scale_pos_weight': scale_weight,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val)

evals_result = {}
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=200,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.record_evaluation(evals_result)
    ]
)

# [5] 預測與評估
print("[4] 模型評估...")
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_label = (y_pred > 0.5).astype(int)

report = classification_report(y_val, y_label, output_dict=True)
roc_auc = roc_auc_score(y_val, y_pred)
fpr, tpr, _ = roc_curve(y_val, y_pred)

# [6] 儲存評估指標與每輪 AUC（供後續 MLflow 使用）
print("[5] 儲存結果檔...")
metrics = {
    "roc_auc": roc_auc,
    "accuracy": report.get("accuracy"),
    "f1": report["1"]["f1-score"],
    "precision": report["1"]["precision"],
    "recall": report["1"]["recall"]
}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(output_dir, "evals_result.json"), "w") as f:
    json.dump(evals_result, f, indent=2)

with open(os.path.join(output_dir, "optimized_lgbm_report.json"), "w") as f:
    json.dump({
        "report": report,
        "roc_auc": roc_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "best_iteration": model.best_iteration,
        "params": params
    }, f, indent=2)

# [7] 繪圖：特徵重要性與 ROC 曲線
print("[6] 繪圖輸出...")
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importance(), y=X.columns)
plt.title("Optimized LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "optimized_feature_importance.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC AUC: {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "optimized_roc_curve.png"))
plt.close()

print(f"\n✅ 模型訓練與評估完成！所有成果已輸出至：{output_dir}")
