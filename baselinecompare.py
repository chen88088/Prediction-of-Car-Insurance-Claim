# claim_model_demo.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# 建立輸出資料夾
project_name = "baselinecompare"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("..", "OUTPUT", f"{project_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# 讀取 train.csv
print("\n[1] 讀取資料...")
df = pd.read_csv("../DATA/train.csv")
X = df.drop(columns=["policy_id", "is_claim"])
y = df["is_claim"]

# Label Encoding for categorical variables
print("[2] 對類別資料進行 Label Encoding...")
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 分割資料集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression
print("[3] 訓練 Logistic Regression...")
log_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_val)
log_probs = log_model.predict_proba(X_val)[:, 1]

# LightGBM
print("[4] 訓練 LightGBM...")
scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
lgb_model = LGBMClassifier(scale_pos_weight=scale_weight, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)
lgb_probs = lgb_model.predict_proba(X_val)[:, 1]

# 評估
print("[5] 產生分類報告...")
results = {
    "LogisticRegression": {
        "report": classification_report(y_val, log_preds, output_dict=True),
        "roc_auc": roc_auc_score(y_val, log_probs),
        "fpr": roc_curve(y_val, log_probs)[0].tolist(),
        "tpr": roc_curve(y_val, log_probs)[1].tolist()
    },
    "LightGBM": {
        "report": classification_report(y_val, lgb_preds, output_dict=True),
        "roc_auc": roc_auc_score(y_val, lgb_probs),
        "fpr": roc_curve(y_val, lgb_probs)[0].tolist(),
        "tpr": roc_curve(y_val, lgb_probs)[1].tolist(),
        "feature_importances": lgb_model.feature_importances_.tolist(),
        "features": X.columns.tolist()
    }
}

# 儲存 json 結果
print("[6] 儲存結果檔...")
with open(os.path.join(output_dir, "logistic_report.json"), "w") as f:
    json.dump(results["LogisticRegression"], f, indent=2)
with open(os.path.join(output_dir, "lightgbm_report.json"), "w") as f:
    json.dump(results["LightGBM"], f, indent=2)

# 繪圖 - LightGBM 特徵重要性
print("[7] 繪製特徵重要性與 ROC Curve")
plt.figure(figsize=(10, 6))
sns.barplot(x=results["LightGBM"]["feature_importances"],
            y=results["LightGBM"]["features"])
plt.title("LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lightgbm_feature_importance.png"))
plt.close()

# 繪 ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(results["LogisticRegression"]["fpr"], results["LogisticRegression"]["tpr"],
         label=f"Logistic ROC AUC: {results['LogisticRegression']['roc_auc']:.3f}")
plt.plot(results["LightGBM"]["fpr"], results["LightGBM"]["tpr"],
         label=f"LightGBM ROC AUC: {results['LightGBM']['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_comparison.png"))
plt.close()

print("\n✅ 全部完成！已輸出報告與圖檔。")
