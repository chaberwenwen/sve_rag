@echo off
chcp 65001 >nul
title SVE RAG 查询助手 - 一键启动

echo ========================================
echo   SVE RAG 查询助手 v4 - 一键启动
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

:: 检查并安装依赖
echo [1/3] 检查依赖...
pip show gradio >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] 正在安装依赖，请稍候...
    pip install -r v4\requirements.txt -q
    if %errorlevel% neq 0 (
        echo [ERROR] 依赖安装失败
        pause
        exit /b 1
    )
    echo [OK] 依赖安装完成
) else (
    echo [OK] 依赖已就绪
)

:: 检查索引是否已构建
echo.
echo [2/3] 检查向量索引...
if not exist v4\index\ (
    echo [构建] 首次运行，正在构建 FAISS 向量索引...
    python v4\run.py index
    if %errorlevel% neq 0 (
        echo [ERROR] 索引构建失败
        pause
        exit /b 1
    )
    echo [OK] 索引构建完成
) else (
    echo [OK] 索引已存在，跳过构建
    echo       如需重建索引请先删除 v4\index\ 文件夹
)

:: 启动 Web UI
echo.
echo [3/3] 启动 Web UI...
echo.
echo 浏览器打开 http://localhost:7860
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

python v4\run.py ui

pause