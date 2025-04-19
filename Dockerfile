FROM python:3.10-slim

# 安裝必要系統依賴（lightgbm 需要編譯器）
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# 建立容器中的預設工作目錄（實際會被 PVC 掛載覆蓋）
WORKDIR /mnt/storage

# 複製依賴列表並安裝 Python 套件
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 預設使用 bash（讓 K8s Job 動態覆蓋執行內容）
CMD ["bash"]
