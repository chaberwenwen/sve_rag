# Dockerfile —— 多阶段构建

# ============================================================
# Stage 1: Builder（安装依赖）
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-server.txt

# ============================================================
# Stage 2: Runtime（最小运行镜像）
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY server/ ./server/
COPY v4/ ./v4/
COPY llm/ ./llm/
COPY data/ ./data/
COPY web/index.html ./web/index.html
COPY .env .env

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/health')" || exit 1

# 启动
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8001"]
