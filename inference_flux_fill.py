#!/usr/bin/env python3
"""
FLUX Fill Inpainting Inference Script
使用训练好的模型进行inpainting推理
"""

import torch
import argparse
from PIL import Image
import os
from diffusers import FluxFillPipeline
from pathlib import Path


def load_pipeline(model_path, device="cuda", dtype=torch.bfloat16):
    """加载FLUX Fill pipeline"""
    print(f"加载模型: {model_path}")
    
    # 加载pipeline
    pipeline = FluxFillPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device
    )
    
    # 启用内存高效attention (如果可用)
    try:
        pipeline.enable_model_cpu_offload()
        print("✅ 启用CPU offload")
    except:
        print("⚠️ CPU offload不可用")
    
    try:
        pipeline.enable_attention_slicing()
        print("✅ 启用attention slicing")
    except:
        print("⚠️ Attention slicing不可用")
    
    return pipeline


def prepare_images(source_path, mask_path, target_size=(512, 512)):
    """准备输入图像"""
    # 加载source图像
    source_image = Image.open(source_path)
    if source_image.mode != 'RGB':
        source_image = source_image.convert('RGB')
    
    # 加载mask图像
    mask_image = Image.open(mask_path)
    if mask_image.mode != 'L':
        mask_image = mask_image.convert('L')
    
    # 调整尺寸
    source_image = source_image.resize(target_size, Image.LANCZOS)
    mask_image = mask_image.resize(target_size, Image.LANCZOS)
    
    return source_image, mask_image


def run_inference(pipeline, source_image, mask_image, prompt, 
                 num_inference_steps=20, guidance_scale=4.0, seed=None):
    """运行推理"""
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        print(f"使用随机种子: {seed}")
    
    print(f"开始推理...")
    print(f"  提示词: {prompt}")
    print(f"  推理步数: {num_inference_steps}")
    print(f"  引导强度: {guidance_scale}")
    
    # 运行推理
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image=source_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="FLUX Fill Inpainting推理")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--source_image", required=True, help="源图像路径")
    parser.add_argument("--mask_image", required=True, help="mask图像路径")
    parser.add_argument("--prompt", required=True, help="推理提示词")
    parser.add_argument("--output", default="output.png", help="输出图像路径")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--guidance", type=float, default=4.0, help="引导强度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--size", type=int, default=512, help="图像尺寸")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.source_image):
        print(f"❌ 源图像不存在: {args.source_image}")
        return
    
    if not os.path.exists(args.mask_image):
        print(f"❌ Mask图像不存在: {args.mask_image}")
        return
    
    print("🚀 FLUX Fill Inpainting推理")
    print(f"📁 模型路径: {args.model_path}")
    print(f"🖼️ 源图像: {args.source_image}")
    print(f"🎭 Mask图像: {args.mask_image}")
    print(f"💬 提示词: {args.prompt}")
    
    # 加载pipeline
    try:
        pipeline = load_pipeline(args.model_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    # 准备图像
    try:
        source_image, mask_image = prepare_images(
            args.source_image, 
            args.mask_image, 
            target_size=(args.size, args.size)
        )
        print(f"✅ 图像预处理完成: {args.size}x{args.size}")
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return
    
    # 运行推理
    try:
        result_image = run_inference(
            pipeline=pipeline,
            source_image=source_image,
            mask_image=mask_image,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
        # 保存结果
        result_image.save(args.output)
        print(f"✅ 推理完成! 结果保存到: {args.output}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 