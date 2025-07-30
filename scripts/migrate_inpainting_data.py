#!/usr/bin/env python3
"""
FLUX Inpainting数据迁移脚本

将您现有的三文件夹结构（source_image_dir, target_image_dir, mask_dir）
适配为AI-toolkit标准格式
"""

import os
import shutil
from pathlib import Path
import argparse

def migrate_inpainting_data(source_dir, target_dir, mask_dir, output_dir, create_captions=True):
    """
    迁移inpainting数据到AI-toolkit标准格式
    
    Args:
        source_dir: 带洞图像文件夹路径
        target_dir: 目标完整图像文件夹路径  
        mask_dir: 掩码文件夹路径
        output_dir: 输出文件夹路径
        create_captions: 是否创建默认caption文件
    """
    
    print("🔄 开始迁移inpainting数据...")
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    target_path = output_path / "target_images"
    source_path = output_path / "masked_images" 
    mask_path = output_path / "masks"
    
    # 创建目录
    target_path.mkdir(parents=True, exist_ok=True)
    source_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 创建输出目录结构：")
    print(f"   目标图像: {target_path}")
    print(f"   带洞图像: {source_path}")
    print(f"   掩码图像: {mask_path}")
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    source_files = []
    for ext in image_extensions:
        source_files.extend(Path(source_dir).glob(f"*{ext}"))
        source_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))
    
    print(f"📊 找到 {len(source_files)} 个源图像文件")
    
    copied_count = 0
    error_count = 0
    
    for source_file in source_files:
        try:
            # 获取文件名（不含扩展名）
            base_name = source_file.stem
            
            # 查找对应的目标图像和掩码
            target_file = None
            mask_file = None
            
            # 在目标文件夹中查找同名文件
            for ext in image_extensions:
                potential_target = Path(target_dir) / f"{base_name}{ext}"
                if potential_target.exists():
                    target_file = potential_target
                    break
                potential_target = Path(target_dir) / f"{base_name}{ext.upper()}"
                if potential_target.exists():
                    target_file = potential_target
                    break
            
            # 在掩码文件夹中查找同名文件
            for ext in image_extensions:
                potential_mask = Path(mask_dir) / f"{base_name}{ext}"
                if potential_mask.exists():
                    mask_file = potential_mask
                    break
                potential_mask = Path(mask_dir) / f"{base_name}{ext.upper()}"
                if potential_mask.exists():
                    mask_file = potential_mask
                    break
            
            if target_file and mask_file:
                # 复制文件，保持原始扩展名
                new_target = target_path / f"{base_name}{target_file.suffix}"
                new_source = source_path / f"{base_name}{source_file.suffix}"
                new_mask = mask_path / f"{base_name}{mask_file.suffix}"
                
                shutil.copy2(target_file, new_target)
                shutil.copy2(source_file, new_source)
                shutil.copy2(mask_file, new_mask)
                
                # 创建caption文件（如果需要）
                if create_captions:
                    caption_file = target_path / f"{base_name}.txt"
                    if not caption_file.exists():
                        with open(caption_file, 'w', encoding='utf-8') as f:
                            f.write("a high quality image")  # 默认描述
                
                copied_count += 1
                if copied_count % 100 == 0:
                    print(f"   已处理 {copied_count} 个文件...")
                    
            else:
                missing = []
                if not target_file:
                    missing.append("target")
                if not mask_file:
                    missing.append("mask")
                print(f"⚠️  跳过 {base_name}: 缺少 {', '.join(missing)} 文件")
                error_count += 1
                
        except Exception as e:
            print(f"❌ 处理文件 {source_file} 时出错: {e}")
            error_count += 1
    
    print(f"\n✅ 数据迁移完成！")
    print(f"   成功复制: {copied_count} 组文件")
    print(f"   错误/跳过: {error_count} 个文件")
    
    # 创建配置文件模板
    config_content = f"""---
job: extension
config:
  name: "flux_inpainting_migrated_v1"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      
      save:
        dtype: bf16
        save_every: 250
        max_step_saves_to_keep: 4
      
      datasets:
        - folder_path: "{target_path.absolute()}"
          inpaint_path: "{source_path.absolute()}"
          mask_path: "{mask_path.absolute()}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
          shuffle_tokens: false
          buckets: true
      
      train:
        batch_size: 1
        gradient_accumulation_steps: 4
        train_unet: true
        train_text_encoder: false
        optimizer: "adamw8bit"
        lr: 1e-4
        steps: 2000
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        timestep_type: "flux_shift"
        dtype: bf16
        ema_config:
          use_ema: true
          ema_decay: 0.99
      
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
      
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - "a beautiful landscape"
          - "a portrait of a person"
        guidance_scale: 4
        sample_steps: 20

meta:
  name: "[name]"
  description: "Migrated FLUX inpainting training"
"""
    
    config_file = output_path / "train_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n📝 已创建配置文件: {config_file}")
    print(f"\n🚀 下一步运行命令:")
    print(f"   python run.py {config_file}")


def main():
    parser = argparse.ArgumentParser(description="迁移FLUX inpainting数据到AI-toolkit标准格式")
    parser.add_argument("--source_dir", required=True, help="带洞图像文件夹路径")
    parser.add_argument("--target_dir", required=True, help="目标完整图像文件夹路径")
    parser.add_argument("--mask_dir", required=True, help="掩码文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出文件夹路径")
    parser.add_argument("--no_captions", action="store_true", help="不创建默认caption文件")
    
    args = parser.parse_args()
    
    # 检查输入目录
    for dir_path, name in [(args.source_dir, "source"), (args.target_dir, "target"), (args.mask_dir, "mask")]:
        if not os.path.exists(dir_path):
            print(f"❌ 错误: {name} 目录不存在: {dir_path}")
            return
    
    migrate_inpainting_data(
        args.source_dir,
        args.target_dir, 
        args.mask_dir,
        args.output_dir,
        create_captions=not args.no_captions
    )


if __name__ == "__main__":
    main() 