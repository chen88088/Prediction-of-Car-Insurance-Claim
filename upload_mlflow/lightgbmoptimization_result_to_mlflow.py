import os
import sys
import pandas as pd
import mlflow
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
    glob.glob(os.path.join(working_dir, "lightgbmoptimization_20*")),
    key=os.path.getmtime,
    reverse=True
)
if not candidates:
    raise FileNotFoundError("❌ 找不到任何 lightgbmoptimization_20* 資料夾")

output_dir = candidates[0]
print(f"\n📂 自動使用最新的輸出資料夾：{output_dir}")

# os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
# os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "default-key")
# os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "default-secret")

# 寫死 MLFLOW 追蹤與 MinIO 設定
os.environ["MLFLOW_TRACKING_URI"] = "http://10.52.52.142:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.52.52.142:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=experiment_run_name):
    # log metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        metrics = pd.read_json(metrics_path, typ="series")
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

    # log evals_result.json (per iteration AUC)
    eval_path = os.path.join(output_dir, "evals_result.json")
    if os.path.exists(eval_path):
        evals = pd.read_json(eval_path)
        if "valid" in evals and "auc" in evals["valid"]:
            for i, auc in enumerate(evals["valid"]["auc"]):
                mlflow.log_metric("roc_auc_iter", auc, step=i)

    # log report
    report_path = os.path.join(output_dir, "optimized_lgbm_report.json")
    if os.path.exists(report_path):
        mlflow.log_artifact(report_path)

    # log images
    for plot_file in ["optimized_feature_importance.png", "optimized_roc_curve.png"]:
        full_path = os.path.join(output_dir, plot_file)
        if os.path.exists(full_path):
            mlflow.log_artifact(full_path)

    print("\n✅ 已上傳所有結果至 MLflow：", experiment_run_name)
