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
from diffusers import FluxFillPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tv_transforms


class KontextInpaintInference:
    """Kontext-inpaint 推理器"""
    
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, base_model_path: str | None = None):
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
        self.base_model_path = base_model_path
        self.ckpt_mode = False
        
        print(f"🚀 加载 Kontext-inpaint 模型: {model_path}")
        self.pipeline = self._load_pipeline()
        
    def _load_pipeline(self):
        """加载 FluxFillPipeline（基于 FLUX.1-Fill）"""
        try:
            model_index_path = os.path.join(self.model_path, "model_index.json")
            if os.path.exists(model_index_path):
                # 标准 Diffusers 管线目录
                pipeline = FluxFillPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype
                )
            else:
                # 兼容训练保存的 checkpoint 目录（仅包含组件子文件夹）
                # 需要提供 base_model_path 作为管线模板
                if not self.base_model_path or not os.path.isdir(self.base_model_path):
                    raise ValueError(
                        f"缺少 base_model_path（或无效）。checkpoint 目录不含 model_index.json，需提供基础模型目录作为模板。例如: --base_model_path /cloud/cloud-ssd1/FLUX.1-Fill-dev"
                    )
                print(f"🔧 使用基础模型模板构建管线: {self.base_model_path}")
                pipeline = FluxFillPipeline.from_pretrained(self.base_model_path, torch_dtype=self.dtype)

                # 逐项替换为 checkpoint 中的组件
                ckpt = self.model_path
                if os.path.isdir(os.path.join(ckpt, "transformer")):
                    transformer = FluxTransformer2DModel.from_pretrained(os.path.join(ckpt, "transformer"), torch_dtype=self.dtype)
                    pipeline.transformer = transformer
                if os.path.isdir(os.path.join(ckpt, "vae")):
                    vae = AutoencoderKL.from_pretrained(os.path.join(ckpt, "vae"), torch_dtype=self.dtype)
                    pipeline.vae = vae
                if os.path.isdir(os.path.join(ckpt, "text_encoder")):
                    te = CLIPTextModelWithProjection.from_pretrained(os.path.join(ckpt, "text_encoder"), torch_dtype=self.dtype)
                    pipeline.text_encoder = te
                if os.path.isdir(os.path.join(ckpt, "text_encoder_2")):
                    te2 = T5EncoderModel.from_pretrained(os.path.join(ckpt, "text_encoder_2"), torch_dtype=self.dtype)
                    pipeline.text_encoder_2 = te2
                if os.path.isdir(os.path.join(ckpt, "tokenizer")):
                    pipeline.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ckpt, "tokenizer"))
                if os.path.isdir(os.path.join(ckpt, "tokenizer_2")):
                    pipeline.tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(ckpt, "tokenizer_2"))

                # 启用 checkpoint 兼容模式（自定义前向，绕过 pipeline.__call__）
                self.ckpt_mode = True

            # 将整个 pipeline 移动到指定设备
            pipeline = pipeline.to(self.device)
            
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

    def _encode_prompt_clip(self, prompt: str):
        # 使用 CLIP 编码，并加载投影层权重（来自训练保存的 proj_hidden.pt / proj_pooled.pt）
        assert hasattr(self, 'pipeline'), "pipeline 未初始化"
        pipe = self.pipeline
        device = self.device
        dtype = self.dtype
        # 文本编码（CPU也可，这里直接在GPU上）
        clip_ids = pipe.tokenizer(
            prompt,
            padding='max_length',
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(device)
        clip_out = pipe.text_encoder(clip_ids, output_hidden_states=True, return_dict=True)
        hidden_states = clip_out.last_hidden_state  # [1, 77, hidden]
        pooled_state = clip_out.text_embeds  # [1, 768]

        # 加载投影层（来自checkpoint目录）
        proj_hidden_path = os.path.join(self.model_path, 'proj_hidden.pt')
        proj_pooled_path = os.path.join(self.model_path, 'proj_pooled.pt')
        if not (os.path.exists(proj_hidden_path) and os.path.exists(proj_pooled_path)):
            raise FileNotFoundError("未找到 proj_hidden.pt 或 proj_pooled.pt，请确认 checkpoint 目录包含这两个文件")
        # 依据 transformer 配置创建投影层并加载
        joint_dim = int(pipe.transformer.config.joint_attention_dim)
        proj_hidden = nn.Linear(pipe.text_encoder.config.hidden_size, joint_dim).to(device, dtype=dtype)
        pooled_dim = pipe.transformer.config.get('pooled_projection_dim', 768)
        proj_pooled = nn.Linear(pipe.text_encoder.config.projection_dim, pooled_dim).to(device, dtype=dtype)
        proj_hidden.load_state_dict(torch.load(proj_hidden_path, map_location=device))
        proj_pooled.load_state_dict(torch.load(proj_pooled_path, map_location=device))
        proj_hidden.eval(); proj_pooled.eval()

        with torch.no_grad():
            enc_states = proj_hidden(hidden_states.to(device, dtype=dtype))
            pool_proj = proj_pooled(pooled_state.to(device, dtype=dtype))
            # 稳健化
            enc_states = torch.clamp(enc_states, -5.0, 5.0)
            pool_proj = torch.clamp(pool_proj, -5.0, 5.0)
        return enc_states, pool_proj

    def _run_inference_ckpt(self, source_image: Image.Image, white_control_image: Image.Image, prompt: str,
                             num_inference_steps: int, guidance_scale: float, generator: torch.Generator | None):
        pipe = self.pipeline
        device = self.device
        dtype = self.dtype

        # 1) 图像预处理到 [-1,1]
        transform = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            # normalize to [-1,1]
            tv_transforms.Lambda(lambda x: x * 2 - 1)
        ])
        source_tensor = transform(source_image).unsqueeze(0).to(device, dtype=dtype)
        control_tensor = transform(white_control_image).unsqueeze(0).to(device, dtype=dtype)

        # 2) VAE 编码为 latent
        with torch.no_grad():
            lat_source = pipe.vae.encode(source_tensor).latent_dist.sample()
            lat_control = pipe.vae.encode(control_tensor).latent_dist.sample()

        # 3) 取前16通道，拼接并补零到64
        lat_source = lat_source[:, :16]
        lat_control = lat_control[:, :16]
        model_input = torch.cat([lat_source, lat_control], dim=1)  # [B,32,H,W]
        if model_input.shape[1] < 64:
            pad = torch.zeros(model_input.shape[0], 64 - model_input.shape[1], model_input.shape[2], model_input.shape[3],
                              device=device, dtype=dtype)
            model_input = torch.cat([model_input, pad], dim=1)

        # 4) patchify (patch_size=1) → [B*H*W, 64]
        B, C, H, W = model_input.shape
        patches = model_input.unfold(2, 1, 1).unfold(3, 1, 1)
        patches = patches.contiguous().view(B, C, -1, 1, 1)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, C * 1 * 1)

        # 5) 文本编码 + 投影
        enc_states, pool_proj = self._encode_prompt_clip(prompt)

        # 6) 位置 ID + timestep/guidance
        txt_len = enc_states.shape[1]
        # 2D ids（去掉batch维，符合diffusers新接口要求）
        txt_ids = torch.zeros((txt_len, 2), device=device, dtype=torch.long)
        y_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()
        x_coords = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).flatten()
        img_ids_single = torch.stack([y_coords, x_coords], dim=1)
        img_ids = img_ids_single  # 2D
        timesteps = torch.zeros(B, dtype=torch.float32, device=device)
        guidance = timesteps.clone()

        # 7) Transformer 前向
        with torch.autocast(device_type='cuda', dtype=dtype):
            pred = pipe.transformer(
                hidden_states=patches,
                encoder_hidden_states=enc_states,
                pooled_projections=pool_proj,
                guidance=guidance,
                timestep=timesteps,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False
            )[0]

        # 8) 还原图像 latent 并解码
        img_seq_len = pred.shape[1]
        h = w = int(img_seq_len ** 0.5)
        img_pred_reshaped = pred.permute(0, 2, 1).reshape(B, pred.shape[2], h, w)
        img_pred_matched = img_pred_reshaped[:, :16]
        with torch.no_grad():
            decoded = pipe.vae.decode(img_pred_matched).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.to(torch.float32)
        img = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)
    
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
        
        if self.ckpt_mode:
            return self._run_inference_ckpt(
                source_image, white_control_image, prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        else:
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    image=source_image,
                    mask_image=white_control_image,
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
    parser.add_argument("--base_model_path", required=False, help="当 model_path 为 checkpoint 目录时提供基础模型目录（例如 FLUX.1-Fill-dev）")
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
        inferencer = KontextInpaintInference(args.model_path, base_model_path=args.base_model_path)
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