import os
import sys
import pandas as pd
import mlflow
import json
import glob


experiment_name = sys.argv[1]
experiment_run_name = sys.argv[2]
# 自動抓最新 baselinecompare_* 資料夾
# ✅ 判斷是否在容器中（未來 K8s）或本地執行
if os.path.exists(f"/mnt/storage/{experiment_run_name}/OUTPUT"):
    working_dir = f"/mnt/storage/{experiment_run_name}/OUTPUT"
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "OUTPUT"))

candidates = sorted(
    glob.glob(os.path.join(working_dir, "baselinecompare_20*")),
    key=os.path.getmtime,
    reverse=True
)
if not candidates:
    raise FileNotFoundError("❌ 找不到任何 baselinecompare_20* 資料夾")

output_dir = candidates[0]
print(f"\n📂 自動使用最新的輸出資料夾：{output_dir}")

# 寫死 MLFLOW 與 MINIO 設定
os.environ["MLFLOW_TRACKING_URI"] = "http://10.52.52.142:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.52.52.142:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=experiment_run_name):
    # Log JSON metrics from logistic_report.json
    logistic_path = os.path.join(output_dir, "logistic_report.json")
    if os.path.exists(logistic_path):
        with open(logistic_path, "r") as f:
            logistic_result = json.load(f)
            mlflow.log_metric("logistic_roc_auc", logistic_result["roc_auc"])
            mlflow.log_metric("logistic_accuracy", logistic_result["report"]["accuracy"])
            mlflow.log_metric("logistic_f1", logistic_result["report"]["1"]["f1-score"])
            mlflow.log_metric("logistic_precision", logistic_result["report"]["1"]["precision"])
            mlflow.log_metric("logistic_recall", logistic_result["report"]["1"]["recall"])
            mlflow.log_artifact(logistic_path)

    # Log JSON metrics from lightgbm_report.json
    lgb_path = os.path.join(output_dir, "lightgbm_report.json")
    if os.path.exists(lgb_path):
        with open(lgb_path, "r") as f:
            lgb_result = json.load(f)
            mlflow.log_metric("lgbm_roc_auc", lgb_result["roc_auc"])
            mlflow.log_metric("lgbm_accuracy", lgb_result["report"]["accuracy"])
            mlflow.log_metric("lgbm_f1", lgb_result["report"]["1"]["f1-score"])
            mlflow.log_metric("lgbm_precision", lgb_result["report"]["1"]["precision"])
            mlflow.log_metric("lgbm_recall", lgb_result["report"]["1"]["recall"])
            mlflow.log_artifact(lgb_path)

    # Log 圖檔
    for plot_file in ["lightgbm_feature_importance.png", "roc_comparison.png"]:
        full_path = os.path.join(output_dir, plot_file)
        if os.path.exists(full_path):
            mlflow.log_artifact(full_path)

    print("\n✅ Logistic vs LightGBM baseline 實驗結果已上傳 MLflow")
