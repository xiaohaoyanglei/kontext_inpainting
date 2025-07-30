@echo off
echo 🚀 启动 FLUX Inpainting 训练...
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或不在 PATH 中
    pause
    exit /b 1
)

REM 检查配置文件
if not exist "config\train_flux_inpainting.yaml" (
    echo ❌ 配置文件不存在，请先运行 quick_start_migration.py
    pause
    exit /b 1
)

REM 启动训练
echo 📊 开始训练...
python run.py config/train_flux_inpainting.yaml

pause
