import os 
import sys
import pandas as pd
import mlflow
import glob
import shutil
import yaml

experiment_name = sys.argv[1]
experiment_run_name = sys.argv[2]

reserved_run_id = None
output_dir = None

if len(sys.argv) >= 4:
    reserved_run_id = sys.argv[3]
if len(sys.argv) >= 5:
    output_dir = os.path.abspath(sys.argv[4])
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"❌ 指定的 output_dir 不存在：{output_dir}")

# 判斷是否在容器中（未來 K8s）或本地執行
# ✅ 取代原本這段
if os.getcwd().startswith("/mnt/storage/") and "CODE" in os.getcwd():
    job_root_dir = os.path.abspath(os.path.join(os.getcwd(),  ".."))  # ✅ 跳出 CODE/upload_mlflow
    working_dir = os.path.join(job_root_dir, "OUTPUT")
    config_path = os.path.join(job_root_dir, "CONFIG", "config.yaml")
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "OUTPUT"))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "CONFIG", "config.yaml"))

# 若有手動指定 output_dir 則使用，否則抓最新
output_dir = None
if len(sys.argv) >= 4:
    output_dir = os.path.abspath(sys.argv[3])
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"❌ 指定的 output_dir 不存在：{output_dir}")
else:
    print(f"🔍 自動模式：列出 {working_dir} 下所有資料夾...")
    all_dirs = [f for f in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, f))]
    lightgbm_dirs = [d for d in all_dirs if d.startswith("lightgbmoptimization_20")]
    print("✅ 可用的資料夾：", lightgbm_dirs)
    if not lightgbm_dirs:
        raise FileNotFoundError("❌ 找不到任何 lightgbmoptimization_20* 資料夾")
    output_dir = os.path.join(working_dir, sorted(lightgbm_dirs, reverse=True)[0])

print(f"\n📂 使用的輸出資料夾：{output_dir}")

# 設定 MLflow 與 MinIO
os.environ["MLFLOW_TRACKING_URI"] = "http://10.52.52.142:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.52.52.142:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name)

# === 啟動 MLflow run ===
if reserved_run_id:
    print(f"使用 reserved run_id: {reserved_run_id}")
    mlflow.start_run(run_id=reserved_run_id)
else:
    print(f"新建一個 run：{experiment_run_name}")
    mlflow.start_run(run_name=experiment_run_name)


    
# 上傳 metrics
metrics_path = os.path.join(output_dir, "metrics.json")
if os.path.exists(metrics_path):
    metrics = pd.read_json(metrics_path, typ="series")
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

# 上傳每輪 AUC（evals_result）
eval_path = os.path.join(output_dir, "evals_result.json")
if os.path.exists(eval_path):
    evals = pd.read_json(eval_path)
    if "valid" in evals and "auc" in evals["valid"]:
        for i, auc in enumerate(evals["valid"]["auc"]):
            mlflow.log_metric("roc_auc_iter", auc, step=i)

# 上傳報告與圖表
for fname in ["optimized_lgbm_report.json", "optimized_feature_importance.png", "optimized_roc_curve.png"]:
    full_path = os.path.join(output_dir, fname)
    if os.path.exists(full_path):
        mlflow.log_artifact(full_path)

# ✅ 上傳當下使用的參數 config.yaml 作為 artifact + MLflow params
if os.path.exists(config_path):
    copied_config_path = os.path.join("/tmp", f"used_config_{experiment_run_name}.yaml")  # ✅ 安全，所有人可寫

    shutil.copyfile(config_path, copied_config_path)
    mlflow.log_artifact(copied_config_path)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        if "params" in config_data:
            for key, value in config_data["params"].items():
                mlflow.log_param(key, value)
# === 完成上傳，結束 run ===
mlflow.end_run()
print(f"\n✅ 已上傳所有結果與 config 至 MLflow：{experiment_run_name}")
