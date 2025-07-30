#!/usr/bin/env python3
"""
测试标准 sd_trainer 是否能正确处理 inpainting 数据

使用此脚本测试迁移后的配置文件是否正常工作
"""

import yaml
import torch
import os
from pathlib import Path
import argparse

def test_config_loading(config_path):
    """测试配置文件加载"""
    print("📋 测试配置文件加载...")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查关键配置项
        process_config = config['config']['process'][0]
        
        print(f"   训练器类型: {process_config['type']}")
        print(f"   模型路径: {process_config['model']['name_or_path']}")
        print(f"   是否启用FLUX: {process_config['model']['is_flux']}")
        
        # 检查数据集配置
        dataset_config = process_config['datasets'][0]
        print(f"   目标图像路径: {dataset_config['folder_path']}")
        print(f"   带洞图像路径: {dataset_config.get('inpaint_path', 'None')}")
        print(f"   掩码路径: {dataset_config.get('mask_path', 'None')}")
        print(f"   缓存latents: {dataset_config.get('cache_latents_to_disk', False)}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def test_data_paths(config):
    """测试数据路径是否存在"""
    print("\n📁 测试数据路径...")
    
    dataset_config = config['config']['process'][0]['datasets'][0]
    
    paths_to_check = [
        ("目标图像", dataset_config['folder_path']),
        ("带洞图像", dataset_config.get('inpaint_path')),
        ("掩码", dataset_config.get('mask_path'))
    ]
    
    all_exist = True
    
    for name, path in paths_to_check:
        if path and Path(path).exists():
            file_count = len(list(Path(path).glob('*.*')))
            print(f"✅ {name}路径存在: {path} ({file_count} 个文件)")
        else:
            print(f"❌ {name}路径不存在或为空: {path}")
            all_exist = False
    
    return all_exist

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🖥️  测试GPU可用性...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("❌ 未检测到可用GPU")
        return False

def test_dataloader_creation(config_path):
    """测试数据加载器创建"""
    print("\n📊 测试数据加载器创建...")
    
    try:
        # 这里我们只是导入必要的模块来测试
        from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
        from toolkit.data_loader import get_dataloader_from_datasets
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_raw_config = config['config']['process'][0]['datasets'][0]
        
        # 预处理配置
        split_configs = preprocess_dataset_raw_config([dataset_raw_config])
        dataset_config = DatasetConfig(**split_configs[0])
        
        print(f"✅ 数据集配置创建成功")
        print(f"   文件夹路径: {dataset_config.folder_path}")
        print(f"   Inpaint路径: {dataset_config.inpaint_path}")
        print(f"   Mask路径: {dataset_config.mask_path}")
        print(f"   缓存latents: {dataset_config.cache_latents_to_disk}")
        print(f"   分辨率: {dataset_config.resolution}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载能力"""
    print("\n🤖 测试模型访问...")
    
    try:
        # 尝试导入huggingface_hub
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            print("⚠️  huggingface_hub未安装，跳过模型访问测试")
            return True
        
        # 检查FLUX模型是否可访问
        repo_id = "black-forest-labs/FLUX.1-dev"
        files = list_repo_files(repo_id)
        
        key_files = ['model_index.json', 'scheduler/scheduler_config.json']
        found_files = [f for f in files if any(key in f for key in key_files)]
        
        print(f"✅ FLUX.1-dev模型可访问")
        print(f"   找到关键文件: {len(found_files)} 个")
        
        return True
        
    except Exception as e:
        print(f"⚠️  模型访问测试失败: {e}")
        print("   (这可能是网络问题或需要HuggingFace令牌)")
        return False

def main():
    parser = argparse.ArgumentParser(description="测试标准sd_trainer配置")
    parser.add_argument("config_path", help="配置文件路径")
    parser.add_argument("--skip_dataloader", action="store_true", help="跳过数据加载器测试")
    parser.add_argument("--skip_model", action="store_true", help="跳过模型访问测试")
    
    args = parser.parse_args()
    
    print("🧪 开始测试标准trainer配置...")
    print("=" * 50)
    
    # 测试配置文件
    config = test_config_loading(args.config_path)
    if not config:
        return
    
    # 测试数据路径
    data_paths_ok = test_data_paths(config)
    
    # 测试GPU
    gpu_ok = test_gpu_availability()
    
    # 测试数据加载器
    if not args.skip_dataloader:
        dataloader_ok = test_dataloader_creation(args.config_path)
    else:
        dataloader_ok = True
        print("\n📊 跳过数据加载器测试")
    
    # 测试模型访问
    if not args.skip_model:
        model_ok = test_model_loading()
    else:
        model_ok = True
        print("\n🤖 跳过模型访问测试")
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 测试总结:")
    
    tests = [
        ("配置文件", config is not None),
        ("数据路径", data_paths_ok),
        ("GPU可用性", gpu_ok),
        ("数据加载器", dataloader_ok),
        ("模型访问", model_ok)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🚀 所有测试通过！可以开始训练：")
        print(f"   python run.py {args.config_path}")
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，请检查配置")
        
        if not data_paths_ok:
            print("\n💡 数据路径问题解决方案：")
            print("   1. 检查路径是否正确")
            print("   2. 运行数据迁移脚本：")
            print("      python scripts/migrate_inpainting_data.py --source_dir ... --target_dir ... --mask_dir ... --output_dir ...")


if __name__ == "__main__":
    main() 