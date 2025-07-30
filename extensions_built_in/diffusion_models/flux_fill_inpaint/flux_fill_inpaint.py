import os
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from toolkit.config_modules import ModelConfig
from toolkit.print import print_acc
from toolkit.models.base_model import BaseModel
from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxFillPipeline
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.models.flux import add_model_gpu_splitter_to_flux, bypass_flux_guidance, restore_flux_guidance
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.util.quantize import quantize, get_qtype
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# 使用与 Kontext 相同的调度器配置以获得多轮一致性
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}


class FluxFillInpaintModel(BaseModel):
    """
    基于 FLUX.1-Fill 的 Kontext-inpaint 模型
    
    核心设计：
    1. 基础模型：FLUX.1-Fill-dev 完整 checkpoint（VAE + UNet）
    2. 扩展：32→hidden 投影层（前16通道复制Fill权重，后16通道置零）
    3. 架构：借用 Kontext 的 Flow-Matching + RoPE 实现多轮一致性
    4. 数据：WhiteMaskDataset（原图 + 纯白控制图）
    """
    arch = "flux_fill_inpaint"

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        # Kontext-inpaint 特定参数
        self.kontext_inpaint_mode = model_config.model_kwargs.get('kontext_inpaint_mode', False)
        self.init_projection_from_original = model_config.model_kwargs.get('init_projection_from_original', True)
        self.two_stage_training = model_config.model_kwargs.get('two_stage_training', False)
        self.stage1_steps = model_config.model_kwargs.get('stage1_steps', 1000)
        
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs
        )
        
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['FluxTransformer2DModel']
        
        # 训练状态跟踪
        self.current_step = 0
        self.is_stage1 = True  # 当前是否在第一阶段（只训练projection）
        
        print_acc(f"🎭 FluxFillInpaintModel 初始化完成")
        print_acc(f"   - 基于: FLUX.1-Fill-dev")
        print_acc(f"   - Kontext-inpaint 模式: {self.kontext_inpaint_mode}")
        print_acc(f"   - 投影层初始化: {self.init_projection_from_original}")
        print_acc(f"   - 两阶段训练: {self.two_stage_training}")
        if self.two_stage_training:
            print_acc(f"   - 第一阶段步数: {self.stage1_steps}")

    @staticmethod
    def get_train_scheduler():
        """使用与 Kontext 相同的调度器配置"""
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        """加载 FLUX.1-Fill 模型并添加 32→hidden 投影层"""
        dtype = self.torch_dtype
        self.print_and_status_update("Loading FLUX.1-Fill model for Kontext-inpaint")
        
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path or model_path

        # 加载 Transformer（基于 FLUX.1-Fill）
        transformer_path = model_path
        transformer_subfolder = 'transformer'
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
            # 检查是否是完整checkpoint
            te_folder_path = os.path.join(model_path, 'text_encoder')
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading FLUX.1-Fill transformer")
        transformer = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=dtype
        )
        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        # 加载文本编码器（使用 FLUX.1-Fill 的配置）
        self.print_and_status_update("Loading T5")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            base_model_path, subfolder="tokenizer_2", torch_dtype=dtype
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        text_encoder_2.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing T5")
            quantize(text_encoder_2, weights=get_qtype(
                self.model_config.qtype))
            text_encoder_2.to(self.device_torch)
            flush()

        self.print_and_status_update("Loading CLIP")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder.to(self.device_torch, dtype=dtype)

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)

        # 使用 Kontext 风格的调度器
        self.noise_scheduler = FluxFillInpaintModel.get_train_scheduler()

        self.print_and_status_update("Making pipeline")
        # 使用 FluxFillPipeline 而不是 FluxKontextPipeline
        pipe: FluxFillPipeline = FluxFillPipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
        tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # 确保所有组件在正确设备上
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        text_encoder[1].to(self.device_torch)
        text_encoder[1].requires_grad_(False)
        text_encoder[1].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # 保存到模型类
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = pipe.transformer
        self.pipeline = pipe
        
        # 应用 Kontext-inpaint 修改
        if self.kontext_inpaint_mode and self.init_projection_from_original:
            self._init_kontext_inpaint_projection()
            
        if self.two_stage_training:
            self._setup_two_stage_training()
            
        self.print_and_status_update("FLUX.1-Fill Kontext-inpaint Model Loaded")

    def _init_kontext_inpaint_projection(self):
        """
        初始化 32→hidden 投影层
        策略：前16通道复制 FLUX.1-Fill 的原始权重，后16通道置零
        """
        transformer: FluxTransformer2DModel = self.model
        
        # 获取原始的 x_embedder (16→hidden)
        original_embedder = transformer.x_embedder
        original_in_channels = original_embedder.weight.shape[1]  # 应该是16*4=64 (16通道*4patch)
        hidden_size = original_embedder.weight.shape[0]
        
        print_acc(f"🔧 初始化基于 FLUX.1-Fill 的 Kontext-inpaint 投影层:")
        print_acc(f"   - FLUX.1-Fill 原始输入通道: {original_in_channels}")
        print_acc(f"   - Kontext-inpaint 目标输入通道: {original_in_channels * 2} (32通道)")
        print_acc(f"   - 隐藏层维度: {hidden_size}")
        
        # 创建新的 32→hidden 投影层
        new_in_channels = original_in_channels * 2  # 32通道 * 4patch = 128
        new_embedder = nn.Linear(new_in_channels, hidden_size, bias=True)
        
        # 初始化权重：前16通道复制 FLUX.1-Fill 权重，后16通道置零
        with torch.no_grad():
            # 前半部分：复制 FLUX.1-Fill 原始权重
            new_embedder.weight[:, :original_in_channels].copy_(original_embedder.weight)
            # 后半部分：置零（对应纯白控制图的通道）
            new_embedder.weight[:, original_in_channels:].zero_()
            
            # 偏置复制 - 正确处理 nn.Parameter
            if original_embedder.bias is not None:
                new_embedder.bias.copy_(original_embedder.bias)
            else:
                new_embedder.bias.zero_()
        
        # 替换模型的投影层
        transformer.x_embedder = new_embedder.to(self.device_torch, dtype=self.torch_dtype)
        
        print_acc(f"✅ 基于 FLUX.1-Fill 的投影层初始化完成")
        print_acc(f"   - 新投影层权重形状: {new_embedder.weight.shape}")
        print_acc(f"   - 前16通道: 复制 FLUX.1-Fill 权重")
        print_acc(f"   - 后16通道: 置零初始化")

    def _setup_two_stage_training(self):
        """设置两阶段训练：第一阶段只训练projection层"""
        self._freeze_all_except_projection()
        print_acc(f"🔄 两阶段训练设置:")
        print_acc(f"   - 第一阶段: 只训练 x_embedder 投影层")
        print_acc(f"   - 第二阶段: 全模型微调")
    
    def _freeze_all_except_projection(self):
        """冻结除投影层外的所有参数"""
        transformer: FluxTransformer2DModel = self.model
        
        # 冻结所有参数
        for param in transformer.parameters():
            param.requires_grad = False
        
        # 只解冻 x_embedder
        for param in transformer.x_embedder.parameters():
            param.requires_grad = True
            
        print_acc(f"❄️  第一阶段：冻结全模型，只训练 x_embedder")
    
    def _unfreeze_all_parameters(self):
        """解冻所有参数（第二阶段）"""
        transformer: FluxTransformer2DModel = self.model
        
        # 解冻所有参数
        for param in transformer.parameters():
            param.requires_grad = True
            
        print_acc(f"🔥 第二阶段：解冻全模型参数")
    
    def update_training_stage(self, current_step: int):
        """
        更新训练阶段
        Args:
            current_step: 当前训练步数
        """
        self.current_step = current_step
        
        if self.two_stage_training:
            # 检查是否需要切换到第二阶段
            if self.is_stage1 and current_step >= self.stage1_steps:
                print_acc(f"🔄 切换到第二阶段训练 (步数: {current_step})")
                self._unfreeze_all_parameters()
                self.is_stage1 = False
                return True  # 返回True表示阶段切换
        
        return False

    def condition_noisy_latents(self, latents: torch.Tensor, batch: 'DataLoaderBatchDTO'):
        """
        条件处理：32通道输入（原图16 + 纯白控制图16）
        基于 FLUX.1-Fill 的处理逻辑，扩展为32通道
        """
        with torch.no_grad():
            control_tensor = batch.control_tensor
            if control_tensor is not None:
                self.vae.to(self.device_torch)
                # 预处理控制张量（纯白图像）
                control_tensor = control_tensor * 2 - 1  # [0,1] -> [-1,1]
                control_tensor = control_tensor.to(self.vae_device_torch, dtype=self.torch_dtype)
                
                # 调整尺寸以匹配 latents
                if batch.tensor is not None:
                    target_h, target_w = batch.tensor.shape[2], batch.tensor.shape[3]
                else:
                    target_h = batch.file_items[0].crop_height
                    target_w = batch.file_items[0].crop_width

                if control_tensor.shape[2] != target_h or control_tensor.shape[3] != target_w:
                    control_tensor = F.interpolate(
                        control_tensor, size=(target_h, target_w), mode='bilinear'
                    )
                    
                # VAE 编码控制图像
                control_latent = self.encode_images(control_tensor).to(latents.device, latents.dtype)
                
                # 拼接：原图latent(16) + 控制图latent(16) = 32通道
                latents = torch.cat((latents, control_latent), dim=1)
                
                print_acc(f"🎭 FLUX.1-Fill Kontext-inpaint latent 拼接:")
                print_acc(f"   - 原图 latent: 16 通道")
                print_acc(f"   - 纯白控制图 latent: 16 通道") 
                print_acc(f"   - 最终输入: {latents.shape[1]} 通道")

        return latents.detach()

    def get_base_model_version(self):
        return "flux.1_fill_inpaint"

    # 以下方法继承 BaseModel 的默认实现，但使用 Fill 的逻辑
    def get_generation_pipeline(self):
        scheduler = FluxFillInpaintModel.get_train_scheduler()

        pipeline: FluxFillPipeline = FluxFillPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        return self.text_encoder[1].encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        """实现提示词编码，基于 FLUX.1-Fill 的双文本编码器"""
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        
        from toolkit import train_tools
        prompt_embeds, pooled_prompt_embeds = train_tools.encode_prompts_flux(
            self.tokenizer,
            self.text_encoder,
            prompt,
            max_length=512,
        )
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.pooled_embeds = pooled_prompt_embeds
        return pe