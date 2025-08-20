#!/usr/bin/env python3
"""
整理新数据集脚本
将raw_data_test中的empty和staged文件分别放入new_dataset的source和target文件夹
并添加统一的prompt文件
"""

import os
import shutil
from pathlib import Path

def prepare_new_dataset():
    """整理新数据集"""
    print("🚀 开始整理新数据集...")
    
    # 源数据目录
    raw_data_dir = "/cloud/cloud-ssd1/raw_data_test"
    
    # 新数据集目录
    new_dataset_dir = "/cloud/cloud-ssd1/new_dataset"
    source_dir = os.path.join(new_dataset_dir, "source_images")
    target_dir = os.path.join(new_dataset_dir, "target_images")
    
    # 创建目录
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有文件
    files = os.listdir(raw_data_dir)
    
    # 分离empty和staged文件
    empty_files = [f for f in files if f.endswith('_empty.png')]
    staged_files = [f for f in files if f.endswith('_staged.png')]
    
    print(f"📊 找到 {len(empty_files)} 个empty文件")
    print(f"📊 找到 {len(staged_files)} 个staged文件")
    
    # 处理empty文件（源图像）
    for empty_file in empty_files:
        # 提取基础名称（去掉_empty.png）
        base_name = empty_file.replace('_empty.png', '')
        
        # 复制到source目录
        source_path = os.path.join(source_dir, f"{base_name}.png")
        shutil.copy2(os.path.join(raw_data_dir, empty_file), source_path)
        
        # 创建对应的prompt文件
        prompt_path = os.path.join(source_dir, f"{base_name}.txt")
        with open(prompt_path, 'w') as f:
            f.write("add furniture")
        
        print(f"  ✅ 处理源图像: {empty_file} -> {base_name}.png")
    
    # 处理staged文件（目标图像）
    for staged_file in staged_files:
        # 提取基础名称（去掉_staged.png）
        base_name = staged_file.replace('_staged.png', '')
        
        # 复制到target目录
        target_path = os.path.join(target_dir, f"{base_name}.png")
        shutil.copy2(os.path.join(raw_data_dir, staged_file), target_path)
        
        print(f"  ✅ 处理目标图像: {staged_file} -> {base_name}.png")
    
    # 验证数据完整性
    print("\n🔍 验证数据完整性...")
    
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    prompt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
    
    print(f"  📁 source_images: {len(source_files)} 个图像文件")
    print(f"  📁 target_images: {len(target_files)} 个图像文件")
    print(f"  📝 prompt文件: {len(prompt_files)} 个文本文件")
    
    # 检查配对完整性
    source_names = {os.path.splitext(f)[0] for f in source_files}
    target_names = {os.path.splitext(f)[0] for f in target_files}
    prompt_names = {os.path.splitext(f)[0] for f in prompt_files}
    
    # 找到完整的配对
    complete_pairs = source_names & target_names & prompt_names
    
    print(f"  ✅ 完整配对: {len(complete_pairs)} 对")
    
    if len(complete_pairs) > 0:
        print(f"  📋 配对示例: {list(complete_pairs)[:5]}")
    
    # 检查缺失的文件
    missing_targets = source_names - target_names
    missing_prompts = source_names - prompt_names
    
    if missing_targets:
        print(f"  ⚠️  缺失目标图像: {len(missing_targets)} 个")
        print(f"     示例: {list(missing_targets)[:3]}")
    
    if missing_prompts:
        print(f"  ⚠️  缺失prompt文件: {len(missing_prompts)} 个")
        print(f"     示例: {list(missing_prompts)[:3]}")
    
    print(f"\n🎉 新数据集整理完成!")
    print(f"   源数据目录: {raw_data_dir}")
    print(f"   新数据集目录: {new_dataset_dir}")
    print(f"   可用训练对: {len(complete_pairs)} 对")
    
    return len(complete_pairs)

def verify_image_differences():
    """验证源图和目标图是否有明显差异"""
    print("\n🔍 验证图像差异...")
    
    from PIL import Image
    import torch
    from torchvision import transforms
    
    source_dir = "/cloud/cloud-ssd1/new_dataset/source_images"
    target_dir = "/cloud/cloud-ssd1/new_dataset/target_images"
    
    # 检查前5对图像
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])[:5]
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    total_mse = 0
    count = 0
    
    for source_file in source_files:
        base_name = os.path.splitext(source_file)[0]
        target_file = f"{base_name}.png"
        target_path = os.path.join(target_dir, target_file)
        
        if os.path.exists(target_path):
            # 加载图像
            source_img = Image.open(os.path.join(source_dir, source_file)).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            # 转换为tensor
            source_tensor = transform(source_img)
            target_tensor = transform(target_img)
            
            # 计算差异
            mse = torch.mean((source_tensor - target_tensor) ** 2).item()
            similarity = 1 - mse
            
            total_mse += mse
            count += 1
            
            print(f"  {base_name}:")
            print(f"    MSE: {mse:.6f}")
            print(f"    相似度: {similarity:.6f}")
            print(f"    差异明显: {'是' if similarity < 0.9 else '否'}")
            print()
    
    if count > 0:
        avg_mse = total_mse / count
        avg_similarity = 1 - avg_mse
        print(f"  平均MSE: {avg_mse:.6f}")
        print(f"  平均相似度: {avg_similarity:.6f}")
        print(f"  整体差异: {'明显' if avg_similarity < 0.9 else '不明显'}")

if __name__ == "__main__":
    # 整理新数据集
    pair_count = prepare_new_dataset()
    
    # 验证图像差异
    verify_image_differences()
    
    print(f"\n✅ 新数据集准备完成，共有 {pair_count} 对训练数据")
    print("现在可以使用这个新数据集重新训练模型了！")
