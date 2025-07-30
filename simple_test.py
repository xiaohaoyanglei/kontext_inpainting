#!/usr/bin/env python3
"""
简化测试脚本，快速验证基础功能
"""

import sys
import os

print("🔍 简化测试开始...")

# 测试1: 基础导入
try:
    import torch
    print("✅ PyTorch 导入成功")
except Exception as e:
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

# 测试2: 配置文件读取
try:
    import yaml
    with open('config/train_kontext_inpaint.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ 配置文件读取成功")
    print(f"   模型架构: {config['config']['process'][0]['model']['arch']}")
    print(f"   模型路径: {config['config']['process'][0]['model']['name_or_path']}")
except Exception as e:
    print(f"❌ 配置文件读取失败: {e}")

# 测试3: 模型路径检查
model_path = "/cloud/cloud-ssd1/FLUX.1-Fill-dev"
if os.path.exists(model_path):
    print(f"✅ 模型路径存在: {model_path}")
    print(f"   包含文件: {os.listdir(model_path)[:5]}...")
else:
    print(f"❌ 模型路径不存在: {model_path}")

# 测试4: 数据目录检查
source_dir = "/cloud/cloud-ssd1/my_dataset/source_images"
target_dir = "/cloud/cloud-ssd1/my_dataset/target_images"
output_dir = "/cloud/cloud-ssd1/training_output"

for path, name in [(source_dir, "源图像目录"), (target_dir, "目标图像目录"), (output_dir, "输出目录")]:
    if os.path.exists(path):
        print(f"✅ {name}存在: {path}")
    else:
        print(f"❌ {name}不存在: {path}")

print("\n📊 简化测试完成!")
print("💡 如果所有测试通过，可以开始准备数据集进行训练")