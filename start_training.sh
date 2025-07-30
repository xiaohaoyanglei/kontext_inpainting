#!/bin/bash
echo "🚀 启动 FLUX Inpainting 训练..."
echo

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装或不在 PATH 中"
    exit 1
fi

# 检查配置文件
if [ ! -f "config/train_flux_inpainting.yaml" ]; then
    echo "❌ 配置文件不存在，请先运行 quick_start_migration.py"
    exit 1
fi

# 启动训练
echo "📊 开始训练..."
python run.py config/train_flux_inpainting.yaml
