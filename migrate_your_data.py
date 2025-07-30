#!/usr/bin/env python3
"""
数据迁移示例脚本

将您的三文件夹结构数据迁移到标准格式
"""

import os
import shutil
from pathlib import Path

def migrate_your_data():
    """
    根据您的实际情况修改以下路径
    """
    
    # 🔧 修改为您的实际路径
    SOURCE_DIRS = {
        'target_images': '/path/to/your/target_image_dir',    # 完整图像
        'masked_images': '/path/to/your/source_image_dir',    # 带洞图像  
        'masks': '/path/to/your/mask_dir'                     # 掩码
    }
    
    OUTPUT_DIR = './data'
    
    print("🔄 开始数据迁移...")
    
    for folder_name, source_path in SOURCE_DIRS.items():
        if not os.path.exists(source_path):
            print(f"⚠️  路径不存在: {source_path}")
            continue
            
        output_path = Path(OUTPUT_DIR) / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 复制所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        copied = 0
        
        for file_path in Path(source_path).iterdir():
            if file_path.suffix.lower() in image_extensions:
                shutil.copy2(file_path, output_path / file_path.name)
                copied += 1
        
        print(f"✅ {folder_name}: 复制了 {copied} 个文件")
    
    # 创建默认caption文件
    target_dir = Path(OUTPUT_DIR) / 'target_images'
    if target_dir.exists():
        for img_file in target_dir.glob('*.*'):
            if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                caption_file = img_file.with_suffix('.txt')
                if not caption_file.exists():
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write("a high quality image")
    
    print("✅ 数据迁移完成！")
    print("🚀 下一步运行: python run.py config/train_flux_inpainting.yaml")

if __name__ == "__main__":
    migrate_your_data()
