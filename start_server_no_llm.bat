@echo off
chcp 65001 >nul
title SVE RAG Web 服务 - 仅检索模式

echo ========================================
echo   SVE RAG Web 服务 - 仅向量检索
echo   (LLM 已关闭，不调用 API，不产生费用)
echo ========================================
echo.

cd /d "%~dp0"

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查并安装服务端依赖
echo [1/4] 检查服务端依赖...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] 正在安装服务端依赖...
    pip install -r requirements-server.txt -q
    if %errorlevel% neq 0 (
        echo [ERROR] 依赖安装失败
        pause
        exit /b 1
    )
    echo [OK] 依赖安装完成
) else (
    echo [OK] 服务端依赖已就绪
)

:: 检查索引
echo.
echo [2/4] 检查向量索引...
if not exist v4\index\card_names.faiss (
    echo [构建] 首次运行，正在构建 FAISS 向量索引...
    python v4\run.py index
    if %errorlevel% neq 0 (
        echo [ERROR] 索引构建失败
        pause
        exit /b 1
    )
    echo [OK] 索引构建完成
) else (
    echo [OK] 索引已存在
)

:: 清理旧进程（端口 8001）
echo.
echo [3/4] 清理旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING" 2^>nul') do (
    echo [清理] 终止旧进程 PID=%%a ...
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

:: 启动服务（LLM 关闭）
echo.
echo [4/4] 启动 Web 服务 (仅向量检索) ...
echo.
echo   前端页面: http://localhost:8001
echo   API 文档: http://localhost:8001/api/docs
echo   健康检查: http://localhost:8001/api/health
echo   (LLM 已全局关闭，不消耗 API 配额)
echo   按 Ctrl+C 停止服务
echo ========================================
echo.

set LLM_ENABLED=false
uvicorn server.main:app --host 0.0.0.0 --port 8001

pause
