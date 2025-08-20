#!/usr/bin/env python3
"""
Kontext-inpaint 伪 mask-free 推理脚本
实现原图 + 纯白控制图 → 编辑结果的推理流程
支持多轮编辑和一致性保持
"""

import torch
import argparse
from PIL import Image
import os
from diffusers import FluxFillPipeline
from pathlib import Path
import numpy as np


class KontextInpaintInference:
    """Kontext-inpaint 推理器"""
    
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            device: 计算设备
            dtype: 数据类型
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        print(f"🚀 加载 Kontext-inpaint 模型: {model_path}")
        self.pipeline = self._load_pipeline()
        
    def _load_pipeline(self):
        """加载 FluxFillPipeline（基于 FLUX.1-Fill）"""
        try:
            pipeline = FluxFillPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device
            )
            
            # 内存优化
            try:
                pipeline.enable_model_cpu_offload()
                print("✅ 启用 CPU offload")
            except:
                print("⚠️ CPU offload 不可用")
                
            try:
                pipeline.enable_attention_slicing()
                print("✅ 启用 attention slicing")
            except:
                print("⚠️ Attention slicing 不可用")
                
            return pipeline
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def create_white_control_image(self, source_image):
        """
        创建纯白控制图像
        
        Args:
            source_image: PIL.Image，原始图像
            
        Returns:
            PIL.Image: 纯白控制图像 (255,255,255)
        """
        width, height = source_image.size
        white_image = Image.new('RGB', (width, height), (255, 255, 255))
        return white_image
    
    def prepare_images(self, source_path, target_size=(512, 512)):
        """
        准备输入图像：原图 + 纯白控制图
        
        Args:
            source_path: 原图路径
            target_size: 目标尺寸
            
        Returns:
            tuple: (source_image, white_control_image)
        """
        # 加载原图
        source_image = Image.open(source_path)
        if source_image.mode != 'RGB':
            source_image = source_image.convert('RGB')
        
        # 创建纯白控制图
        white_control_image = self.create_white_control_image(source_image)
        
        # 调整尺寸
        source_image = source_image.resize(target_size, Image.LANCZOS)
        white_control_image = white_control_image.resize(target_size, Image.LANCZOS)
        
        return source_image, white_control_image
    
    def run_inference(self, source_image, white_control_image, prompt, 
                     num_inference_steps=20, guidance_scale=4.0, seed=None):
        """
        运行 Kontext-inpaint 推理
        
        Args:
            source_image: PIL.Image，原始图像
            white_control_image: PIL.Image，纯白控制图像
            prompt: str，编辑提示词
            num_inference_steps: int，推理步数
            guidance_scale: float，引导强度
            seed: int，随机种子
            
        Returns:
            PIL.Image: 编辑后的图像
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"🎲 使用随机种子: {seed}")
        
        print(f"🎨 开始 Kontext-inpaint 推理...")
        print(f"   📝 提示词: {prompt}")
        print(f"   🔄 推理步数: {num_inference_steps}")
        print(f"   🎯 引导强度: {guidance_scale}")
        print(f"   🎭 控制模式: 纯白控制图 (伪 mask-free)")
        
        with torch.no_grad():
            # 使用 FluxFillPipeline 进行推理
            # 基于 FLUX.1-Fill，使用纯白图作为 mask
            result = self.pipeline(
                prompt=prompt,
                image=source_image,  # 原图作为输入
                mask_image=white_control_image,  # 纯白控制图作为mask
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=source_image.height,
                width=source_image.width,
            )
        
        return result.images[0]
    
    def multi_round_edit(self, source_path, edit_prompts, output_dir, 
                        num_inference_steps=20, guidance_scale=4.0, seed=42):
        """
        多轮编辑功能：保持一致性的连续编辑
        
        Args:
            source_path: 原图路径
            edit_prompts: list，编辑提示词列表
            output_dir: 输出目录
            num_inference_steps: 推理步数
            guidance_scale: 引导强度  
            seed: 随机种子
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔄 开始多轮编辑，共 {len(edit_prompts)} 轮")
        
        # 第一轮：从原图开始
        current_image_path = source_path
        
        for i, prompt in enumerate(edit_prompts):
            print(f"\n🎯 第 {i+1} 轮编辑: {prompt}")
            
            # 准备图像
            source_image, white_control_image = self.prepare_images(current_image_path)
            
            # 执行推理
            result_image = self.run_inference(
                source_image=source_image,
                white_control_image=white_control_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed + i  # 每轮使用不同种子
            )
            
            # 保存结果
            output_path = os.path.join(output_dir, f"round_{i+1:02d}_{prompt[:30].replace(' ', '_')}.png")
            result_image.save(output_path)
            print(f"✅ 第 {i+1} 轮完成，保存到: {output_path}")
            
            # 更新当前图像为下一轮的输入
            current_image_path = output_path
        
        print(f"\n🎉 多轮编辑完成！所有结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Kontext-inpaint 伪 mask-free 推理")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--source_image", required=True, help="源图像路径")
    parser.add_argument("--prompt", required=True, help="编辑提示词")
    parser.add_argument("--output", default="kontext_inpaint_result.png", help="输出图像路径")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--guidance", type=float, default=4.0, help="引导强度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--size", type=int, default=512, help="图像尺寸")
    
    # 多轮编辑选项
    parser.add_argument("--multi_round", action="store_true", help="启用多轮编辑模式")
    parser.add_argument("--prompts_file", help="多轮编辑提示词文件（每行一个）")
    parser.add_argument("--output_dir", default="multi_round_results", help="多轮编辑输出目录")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.source_image):
        print(f"❌ 源图像不存在: {args.source_image}")
        return
    
    print("🎭 Kontext-inpaint 伪 mask-free 推理")
    print(f"📁 模型路径: {args.model_path}")
    print(f"🖼️ 源图像: {args.source_image}")
    
    # 初始化推理器
    try:
        inferencer = KontextInpaintInference(args.model_path)
    except Exception as e:
        print(f"❌ 推理器初始化失败: {e}")
        return
    
    if args.multi_round and args.prompts_file:
        # 多轮编辑模式
        print(f"🔄 多轮编辑模式")
        print(f"📝 提示词文件: {args.prompts_file}")
        
        if not os.path.exists(args.prompts_file):
            print(f"❌ 提示词文件不存在: {args.prompts_file}")
            return
            
        # 读取提示词
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            edit_prompts = [line.strip() for line in f if line.strip()]
        
        # 执行多轮编辑
        inferencer.multi_round_edit(
            source_path=args.source_image,
            edit_prompts=edit_prompts,
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
    else:
        # 单次推理模式
        print(f"💬 编辑提示: {args.prompt}")
        
        # 准备图像
        try:
            source_image, white_control_image = inferencer.prepare_images(
                args.source_image, 
                target_size=(args.size, args.size)
            )
            print(f"✅ 图像预处理完成: {args.size}x{args.size}")
        except Exception as e:
            print(f"❌ 图像预处理失败: {e}")
            return
        
        # 运行推理
        try:
            result_image = inferencer.run_inference(
                source_image=source_image,
                white_control_image=white_control_image,
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed
            )
            
            # 保存结果
            result_image.save(args.output)
            print(f"✅ 推理完成！结果保存到: {args.output}")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()