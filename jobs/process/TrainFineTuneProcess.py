from collections import OrderedDict
from jobs import TrainJob
from jobs.process import BaseTrainProcess
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import FluxTransformer2DModel, AutoencoderKL
from toolkit.data_loader import ImageDataset, WhiteMaskDataset
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from toolkit.models.flux import add_model_gpu_splitter_to_flux
from PIL import Image
from torchvision import transforms
import os
import psutil

# 尝试导入 AdamW8bit，如果失败则使用 AdamW
try:
    from bitsandbytes.optim import AdamW8bit
    OPTIMIZER_CLASS = AdamW8bit
except ImportError:
    OPTIMIZER_CLASS = AdamW

class TrainFineTuneProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job: TrainJob, config: OrderedDict):
        super().__init__(process_id, job, config)
        
        # Kontext-inpaint 两阶段训练参数
        self.two_stage_training = config.get('two_stage_training', True)
        self.stage1_steps = config.get('stage1_steps', 2000)
        self.stage1_lr = config.get('stage1_lr', 1e-4)
        self.stage2_lr = config.get('stage2_lr', 5e-5)
        self.current_step = 0
        self.is_stage1 = True
    
    def get_gpu_memory_info(self):
        """获取GPU显存信息"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved  # GB
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'free': memory_free,
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return None
    
    def print_memory_status(self, step=None):
        """打印内存状态"""
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            step_info = f" [Step {step}]" if step is not None else ""
            print(f"💾 GPU显存{step_info}: "
                  f"已分配 {gpu_info['allocated']:.2f}GB, "
                  f"已保留 {gpu_info['reserved']:.2f}GB, "
                  f"可用 {gpu_info['free']:.2f}GB, "
                  f"总计 {gpu_info['total']:.2f}GB")
        
        # 系统内存
        memory = psutil.virtual_memory()
        print(f"💻 系统内存: "
              f"已用 {memory.used / 1024**3:.2f}GB, "
              f"可用 {memory.available / 1024**3:.2f}GB, "
              f"总计 {memory.total / 1024**3:.2f}GB")

    def run(self):
        # 设置PyTorch显存优化环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("🔧 启用PyTorch显存片段化优化")
        
        token = "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"
        device = self.config.get('device', 'cuda')
        print(f"使用设备: {device}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        # 1. 加载 VAE - 如果有双GPU，放到GPU 1
        if torch.cuda.device_count() > 1:
            vae_device = 'cuda:1'
            print("📍 VAE 放置到 GPU 1，为主模型释放 GPU 0 显存")
        else:
            vae_device = device

        vae = AutoencoderKL.from_pretrained(
            self.config['vae_path'], token=token
        ).eval().to(vae_device)

        # 2. 数据加载器 - 使用 WhiteMaskDataset，但添加latent缓存
        from toolkit.data_loader import WhiteMaskDataset
        
        # 创建支持latent缓存的WhiteMaskDataset
        class CachedWhiteMaskDataset(WhiteMaskDataset):
            def __init__(self, config, source_dir=None, target_dir=None, mask_dir=None, vae=None):
                super().__init__(config, source_dir, target_dir, mask_dir)
                self.vae = vae
                self.latent_cache = {}
                self.disk_cache = {}
                
            def get_latent_cache_path(self, img_path):
                """获取latent缓存路径"""
                img_dir = os.path.dirname(img_path)
                latent_dir = os.path.join(img_dir, '_latent_cache')
                filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                # 简单的缓存文件名
                cache_path = os.path.join(latent_dir, f'{filename_no_ext}.safetensors')
                return cache_path
                
            def load_or_encode_latent(self, img_tensor, cache_key):
                """加载或编码latent"""
                if cache_key in self.latent_cache:
                    return self.latent_cache[cache_key]
                
                # 编码latent
                with torch.no_grad():
                    if self.vae is not None:
                        # 确保输入形状正确 [B, C, H, W]
                        if img_tensor.dim() == 3:
                            img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
                        elif img_tensor.dim() != 4:
                            raise ValueError(f"Expected 3D or 4D tensor, got {img_tensor.dim()}D")
                        
                        # 将图像tensor移动到VAE所在的设备
                        img_tensor = img_tensor.to(self.vae.device)
                        
                        latent = self.vae.encode(img_tensor).latent_dist.sample()
                        if latent.dim() == 4 and latent.shape[0] == 1:
                            latent = latent.squeeze(0)  # 移除batch维度
                    else:
                        # 如果没有VAE，返回随机latent（仅用于测试）
                        latent = torch.randn(16, 32, 32, device=img_tensor.device, dtype=img_tensor.dtype)
                
                # 缓存到内存
                self.latent_cache[cache_key] = latent
                return latent
            
            def pre_cache_all_latents(self):
                """预缓存所有latent到磁盘"""
                print("🔄 开始预缓存所有latent...")
                import os
                from safetensors.torch import save_file, load_file
                
                cached_count = 0
                total_count = len(self.file_list)
                
                for i, img_path in enumerate(self.file_list):
                    if i % 10 == 0:
                        print(f"缓存进度: {i}/{total_count}")
                    
                    # 检查是否已经缓存
                    cache_path = self.get_latent_cache_path(img_path)
                    if os.path.exists(cache_path):
                        cached_count += 1
                        continue
                    
                    try:
                        # 加载图像
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
                        img_tensor = transforms.ToTensor()(img)
                        
                        # 将图像tensor移动到VAE所在的设备
                        if self.vae is not None:
                            img_tensor = img_tensor.to(self.vae.device)
                        
                        # 编码latent
                        with torch.no_grad():
                            if img_tensor.dim() == 3:
                                img_tensor = img_tensor.unsqueeze(0)
                            latent = self.vae.encode(img_tensor).latent_dist.sample()
                            if latent.dim() == 4 and latent.shape[0] == 1:
                                latent = latent.squeeze(0)
                        
                        # 保存到磁盘
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        save_file({'latent': latent.cpu()}, cache_path)
                        cached_count += 1
                        
                    except Exception as e:
                        print(f"缓存失败 {img_path}: {e}")
                
                print(f"✅ 预缓存完成: {cached_count}/{total_count} 个latent已缓存")
            
            def load_cached_latent(self, img_path):
                """从磁盘加载缓存的latent"""
                cache_path = self.get_latent_cache_path(img_path)
                if os.path.exists(cache_path):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(cache_path, device='cpu')
                        return state_dict['latent']
                    except Exception as e:
                        print(f"加载缓存失败 {cache_path}: {e}")
                        return None
                return None
        
        dataset = CachedWhiteMaskDataset(
            config={'include_prompt': True, 'resolution': self.config.get('resolution', 256)},
            source_dir=self.config['source_image_dir'],
            target_dir=self.config['target_image_dir'],
            vae=vae  # 传入VAE用于编码
        )
        
        # 预缓存所有latent
        print("🔄 检查并预缓存latent...")
        dataset.pre_cache_all_latents()
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.config.get('batch_size', 1),
            shuffle=True,
            num_workers=0  # 减少多进程显存占用
        )

        # 3. 强制 in_channels=64, patch_size=1 (支持32通道输入)
        in_channels = 64
        patch_size = 1
        patch_dim = in_channels * patch_size * patch_size
        print(f"强制使用 in_channels={in_channels}, patch_size={patch_size}, patch_dim={patch_dim}")
        print(f"🎯 Kontext-inpaint: 32通道输入 (原图16 + 纯白控制图16)")
        
        # 4. 加载模型
        model_path = self.config['model_path']
        transformer_path = model_path + "/transformer" if not model_path.endswith("/transformer") else model_path
        
        model = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder="",
            in_channels=in_channels,
            token=token,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
            axes_dims_rope=[64, 64]  # 每个轴64维，总共128维匹配attention head
        )
        
        # 检查模型配置
        print(f"使用 joint_attention_dim 作为 hidden_size: {model.config.joint_attention_dim}")
        actual_hidden_size = model.config.joint_attention_dim
        print(f"检测到模型实际 hidden_size: {actual_hidden_size}")

        # 额外调试：图像与文本通道的隐藏维度是否一致
        try:
            x_embedder_out = int(model.x_embedder.weight.shape[0])
            x_embedder_in = int(model.x_embedder.weight.shape[1])
            print(f"x_embedder.in_features: {x_embedder_in}, x_embedder.out_features: {x_embedder_out}")
            if x_embedder_out != actual_hidden_size:
                print("⚠️ 维度不一致：x_embedder.out_features != joint_attention_dim，模型内部会做适配，但可能数值更敏感")
        except Exception as _:
            pass
        
        # 为确保数值稳定，强制重置 x_embedder（与稳定版本一致）
        try:
            torch.nn.init.xavier_uniform_(model.x_embedder.weight)
            if hasattr(model.x_embedder, 'bias') and model.x_embedder.bias is not None:
                torch.nn.init.zeros_(model.x_embedder.bias)
            print("🔧 已强制重置 x_embedder 权重 (Xavier) — 回到稳定版本策略")
        except Exception as _:
            pass

        # 两阶段训练设置
        if self.two_stage_training:
            print("🔄 启用两阶段训练:")
            print(f"   - 第一阶段: 只训练 x_embedder (步数: {self.stage1_steps})")
            print("   - 第二阶段: 全模型微调")
            print("❄️  第一阶段：冻结全模型，只训练 x_embedder")
            self._freeze_all_except_projection(model)
        
        # 开启梯度检查点以降低显存峰值
        try:
            model.enable_gradient_checkpointing()
            print("⚙️ 已开启梯度检查点 (gradient checkpointing)")
        except Exception as _:
            print("⚠️ 开启梯度检查点失败，继续以默认方式训练")
        
        # 🔥 双GPU策略：模型在GPU0，VAE和文本编码器在GPU1
        if torch.cuda.device_count() > 1:
            print(f"🔥 检测到 {torch.cuda.device_count()} 张GPU，启动双GPU模式！")
            print("💎 双GPU协同作战：160GB显存全面释放！")
            
            # 设置主设备为cuda:0
            device = 'cuda:0'
            model = model.to(device)
            
            print("🚀 双GPU模式：主模型GPU0，VAE/文本编码器GPU1")
        else:
            print("⚠️  只检测到单卡")
            device = 'cuda:0'
            model = model.to(device)

        print(f"Flux 模型加载完成，patch_size={patch_size}，patch_dim={patch_dim}, hidden_size={actual_hidden_size}, in_channels={in_channels}")
        print(f"x_embedder.weight.shape: {model.x_embedder.weight.shape}")
        print(f"model.config keys: {list(model.config.keys())}")
        print(f"模型的 axes_dims_rope: {model.config.axes_dims_rope}")
        print(f"pos_embed.axes_dim: {model.config.axes_dims_rope}")
        
        # 6. 加载文本编码器与分词器 - 如果有双GPU，也放到GPU 1
        # 将文本编码器放置到 CPU，显著降低显存占用
        text_encoder_device = 'cpu'
        print("📍 文本编码器放置到 CPU，降低显存峰值")
            
        # 加载第一个文本编码器 (CLIP)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.config['text_encoder_path'], token=token
        ).eval().to(text_encoder_device)
        
        # 加载第二个文本编码器 (T5)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.config['text_encoder_2_path'], token=token
        ).eval().to(text_encoder_device)
        
        # 加载对应的tokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "/cloud/cloud-ssd1/FLUX.1-Fill-dev/tokenizer", token=token
        )
        
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            "/cloud/cloud-ssd1/FLUX.1-Fill-dev/tokenizer_2", token=token
        )
        
        # 文本投影层放在主设备（GPU），以避免优化器/8bit在CPU上的设备不一致问题
        # 维度按模型 joint_attention_dim（如 4096）
        joint_dim = int(model.config.joint_attention_dim)
        proj_hidden = torch.nn.Linear(
            text_encoder.config.hidden_size,
            joint_dim
        ).to(device)
        pooled_projection_dim = model.config.get('pooled_projection_dim', 768)
        proj_pooled = torch.nn.Linear(
            text_encoder.config.projection_dim,
            pooled_projection_dim
        ).to(device)
        
        # 正确初始化投影层权重，防止NaN
        print("🔧 初始化投影层权重...")
        torch.nn.init.xavier_uniform_(proj_hidden.weight)
        torch.nn.init.zeros_(proj_hidden.bias)
        torch.nn.init.xavier_uniform_(proj_pooled.weight)
        torch.nn.init.zeros_(proj_pooled.bias)
        # 将投影层缩放，避免过大数值进入主干
        with torch.no_grad():
            proj_hidden.weight.mul_(0.02)
            proj_pooled.weight.mul_(0.02)
        print(f"✅ 投影层权重初始化完成 (附加缩放 0.02)，proj_hidden.out_features={proj_hidden.out_features}, joint_dim={joint_dim}")

        # 如 x_embedder 权重异常为全零，则进行一次稳健初始化
        try:
            if float(model.x_embedder.weight.norm().item()) == 0.0:
                torch.nn.init.xavier_uniform_(model.x_embedder.weight)
                if hasattr(model.x_embedder, 'bias') and model.x_embedder.bias is not None:
                    torch.nn.init.zeros_(model.x_embedder.bias)
                with torch.no_grad():
                    model.x_embedder.weight.mul_(0.02)
                print("⚠️ 检测到 x_embedder 权重范数为 0，已执行 Xavier 初始化并缩放 0.02")
        except Exception:
            pass

        # 7. 优化器（为 x_embedder 设置更稳健的学习率与权重衰减）
        stage1_lr = self.config.get('stage1_lr', self.config.get('lr', 1e-4))
        # x_embedder 用更低 LR（上限 5e-5）与更强 WD，缓解权重范数膨胀
        x_embedder_lr = min(stage1_lr, 5e-5)
        x_embedder_wd = 0.05
        other_wd = 0.01
        x_embedder_params = list(model.x_embedder.parameters()) if hasattr(model, 'x_embedder') else []
        other_params = [p for p in model.parameters() if p.requires_grad and (id(p) not in {id(pp) for pp in x_embedder_params})]
        proj_params = list(proj_hidden.parameters()) + list(proj_pooled.parameters())

        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit([
                {"params": x_embedder_params, "lr": x_embedder_lr, "weight_decay": x_embedder_wd},
                {"params": other_params, "lr": stage1_lr, "weight_decay": other_wd},
                {"params": proj_params, "lr": stage1_lr, "weight_decay": other_wd},
            ])
            print("使用 AdamW8bit 优化器以节省显存（x_embedder 启用更低 LR/更高 WD）")
        except ImportError:
            optimizer = OPTIMIZER_CLASS([
                {"params": x_embedder_params, "lr": x_embedder_lr, "weight_decay": x_embedder_wd},
                {"params": other_params, "lr": stage1_lr, "weight_decay": other_wd},
                {"params": proj_params, "lr": stage1_lr, "weight_decay": other_wd},
            ])
            print(f"使用 {OPTIMIZER_CLASS.__name__} 优化器（x_embedder 启用更低 LR/更高 WD）")

        # 8. 混合精度训练
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print("启用混合精度训练以减少显存使用")
        
        # 9. 训练主循环（以 steps 为主驱动，epoch 仅用于计数）
        num_epochs = self.config.get('num_epochs', 1)
        total_steps = self.config.get('steps', 10000)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        
        print(f"开始训练: {num_epochs} epochs (仅用于显示), {total_steps} steps (主驱动), 梯度累积: {gradient_accumulation_steps}")
        try:
            print(f"数据集样本数（配对）: {len(dataset)}")
        except Exception:
            pass
        
        global_step = 0
        # 文本编码缓存：caption -> (hidden_states_cpu, pooled_state_cpu)
        text_encode_cache = {}
        epoch = 0
        
        while global_step < total_steps:
            for step, batch in enumerate(train_loader):
                # 更新训练阶段
                self.update_training_stage(global_step, model, optimizer)
                
                # 批次数据移至 GPU
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                optimizer.zero_grad()
                
                # 使用缓存的latent - 避免重复VAE编码
                with torch.no_grad():
                    # 生成缓存键
                    source_key = f"source_{step}"
                    control_key = f"control_{step}"
                    target_key = f"target_{step}"
                    
                    # 使用缓存的latent或重新编码
                    if hasattr(dataset, 'load_or_encode_latent'):
                        # 使用自定义缓存
                        latent_source = dataset.load_or_encode_latent(batch['source_image'], source_key)
                        latent_control = dataset.load_or_encode_latent(batch['control_tensor'], control_key)
                        latent_target = dataset.load_or_encode_latent(batch['tensor'], target_key)
                    else:
                        # 回退到直接编码
                        if torch.cuda.device_count() > 1:
                            source_img = batch['source_image'].to('cuda:1')
                            control_img = batch['control_tensor'].to('cuda:1')
                            target_img = batch['tensor'].to('cuda:1')
                            
                            latent_source = vae.encode(source_img).latent_dist.sample()
                            latent_control = vae.encode(control_img).latent_dist.sample()
                            latent_target = vae.encode(target_img).latent_dist.sample()
                            
                            latent_source = latent_source.to('cuda:0')
                            latent_control = latent_control.to('cuda:0')
                            latent_target = latent_target.to('cuda:0')
                            
                            del source_img, control_img, target_img
                        else:
                            latent_source = vae.encode(batch['source_image']).latent_dist.sample()
                            latent_control = vae.encode(batch['control_tensor']).latent_dist.sample()
                            latent_target = vae.encode(batch['tensor']).latent_dist.sample()
                    
                    torch.cuda.empty_cache()  # 清理显存
                    
                # 32通道输入：原图16 + 纯白控制图16
                # 检查latent维度并确保形状正确
                if latent_source.dim() == 3:
                    # 如果是3D，添加batch维度
                    latent_source = latent_source.unsqueeze(0)
                    latent_control = latent_control.unsqueeze(0)
                    latent_target = latent_target.unsqueeze(0)
                
                # 只取前16通道
                latent_source = latent_source[:, :16, :, :]  # 只取前16通道
                latent_control = latent_control[:, :16, :, :]  # 只取前16通道
                
                # 拼接：原图latent(16) + 控制图latent(16) = 32通道
                model_input = torch.cat([latent_source, latent_control], dim=1)  # [B, 32, H, W]
                
                # 自动补零到in_channels=64
                if model_input.shape[1] < in_channels:
                    pad = torch.zeros((model_input.shape[0], in_channels - model_input.shape[1], model_input.shape[2], model_input.shape[3]), device=model_input.device, dtype=model_input.dtype)
                    model_input = torch.cat([model_input, pad], dim=1)
                elif model_input.shape[1] > in_channels:
                    model_input = model_input[:, :in_channels, :, :]
                
                if global_step % 100 == 0:  # 每百步检查一次
                    print(f"Step {global_step}: 最终送入模型的shape: {model_input.shape}")
                    try:
                        print(f"model_input stats -> min: {model_input.min().item():.4f}, max: {model_input.max().item():.4f}, mean: {model_input.mean().item():.4f}")
                    except Exception:
                        pass
                
                # patchify 检查
                B, C, H, W = model_input.shape
                patches = model_input.unfold(2, 1, 1).unfold(3, 1, 1)
                patches = patches.contiguous().view(B, C, -1, 1, 1)
                patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, C * 1 * 1)
                
                if H % patch_size != 0 or W % patch_size != 0:
                    raise ValueError(f"H/W必须能被patch_size整除，当前H={H}, W={W}, patch_size={patch_size}")
                if C != in_channels:
                    raise ValueError(f"输入通道数应为{in_channels}，实际为{C}")

                # 文本编码（无梯度）+ 投影（需梯度）
                # 文本侧统一在 CPU 编码
                text_device = 'cpu'

                with torch.no_grad():
                    # 统一生成缓存键（支持字符串或列表）
                    cap = batch['caption']
                    if isinstance(cap, (list, tuple)):
                        cap_key = "\n".join([str(x) for x in cap])
                    else:
                        cap_key = str(cap)
                    if cap_key in text_encode_cache:
                        hidden_states, pooled_state = text_encode_cache[cap_key]
                    else:
                        # CLIP 编码（CPU）
                        clip_ids = clip_tokenizer(
                            cap, padding='max_length',
                            max_length=clip_tokenizer.model_max_length,
                            truncation=True, return_tensors='pt'
                        ).input_ids.to(text_device)
                        clip_out = text_encoder(
                            clip_ids, output_hidden_states=True, return_dict=True
                        )
                        hidden_states = clip_out.last_hidden_state.cpu()  # 缓存为CPU张量
                        pooled_state = clip_out.text_embeds.cpu()

                        # 可选：T5 编码（CPU）目前不参与投影，计算成本大，默认跳过
                        # 如需启用，取消下方注释并可加入缓存
                        # t5_ids = t5_tokenizer(
                        #     cap, padding='max_length',
                        #     max_length=t5_tokenizer.model_max_length,
                        #     truncation=True, return_tensors='pt'
                        # ).input_ids.to(text_device)
                        # t5_out = text_encoder_2(
                        #     t5_ids, output_hidden_states=True, return_dict=True
                        # )
                        # t5_hidden_states = t5_out.last_hidden_state

                        text_encode_cache[cap_key] = (hidden_states, pooled_state)

                # 投影（启用梯度；在GPU）
                enc_states = proj_hidden(hidden_states.to(device))
                pool_proj = proj_pooled(pooled_state.to(device))

                # enc_states / pool_proj 已在 GPU

                # 与模型输入dtype对齐并裁剪，提升数值稳定性
                enc_states = enc_states.to(dtype=model_input.dtype)
                pool_proj = pool_proj.to(dtype=model_input.dtype)
                enc_states = torch.clamp(enc_states, -5.0, 5.0)
                pool_proj = torch.clamp(pool_proj, -5.0, 5.0)
                
                if global_step % 100 == 0:
                    print(f"pool_proj shape: {pool_proj.shape}")
                    try:
                        print(f"pool_proj stats -> min: {pool_proj.min().item():.4f}, max: {pool_proj.max().item():.4f}, mean: {pool_proj.mean().item():.4f}")
                    except Exception:
                        pass
                    print(f"文本编码 hidden_states shape: {hidden_states.shape}")
                    print(f"投影后 enc_states shape: {enc_states.shape}")
                    try:
                        print(f"enc_states stats -> min: {enc_states.min().item():.4f}, max: {enc_states.max().item():.4f}, mean: {enc_states.mean().item():.4f}")
                    except Exception:
                        pass
                    try:
                        if 'x_embedder_out' in locals() and enc_states.shape[-1] != x_embedder_out:
                            print("⚠️ 维度不一致：enc_states最后维度与x_embedder.out_features不同，可能造成融合不稳定")
                    except Exception:
                        pass
                    print(f"实际文本序列长度: {hidden_states.shape[1]}")
                
                # 模型输入序列化
                model_input_seq = patches  # [B*H*W, C*patch_size*patch_size]
                if global_step % 100 == 0:
                    try:
                        print(f"model_input_seq shape: {model_input_seq.shape}")
                    except Exception:
                        pass
                
                if global_step % 100 == 0:
                    print(f"送入 model 前的 shape: {model_input_seq.shape}")
                
                # 位置编码ID生成
                txt_ids = torch.zeros((B, hidden_states.shape[1], 2), device=device, dtype=torch.long)  # [B, 77, 2]
                y_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()  # [4096]
                x_coords = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).flatten()  # [4096]
                img_ids_single = torch.stack([y_coords, x_coords], dim=1)  # [4096, 2]
                img_ids = img_ids_single.unsqueeze(0).expand(B, -1, -1)  # [B, 4096, 2]
                
                if global_step % 100 == 0:
                    print(f"传入的 txt_ids shape: {txt_ids.shape}")  # [B, 77, 2]
                    print(f"传入的 img_ids shape: {img_ids.shape}")  # [B, 4096, 2]
                    print(f"模拟 cat 后的 shape: {torch.cat([txt_ids, img_ids], dim=1).shape}")  # [B, 4173, 2]
                    print(f"ids 最后一维 (n_axes): {txt_ids.shape[-1]}")
                    total_seq_len = txt_ids.shape[1] + img_ids.shape[1]
                    print(f"总序列长度: {total_seq_len} (txt: {txt_ids.shape[1]} + img: {img_ids.shape[1]})")

                # Flux 模型前向
                timesteps = torch.zeros(model_input.shape[0], dtype=torch.float32, device=device)  # [B] float32
                guidance = timesteps.clone()  # [B] float32
                
                if global_step % 100 == 0:
                    print("🔍 检查输入数据...")
                
                # 清理潜在的非有限值（保存原逻辑，但阈值收紧一点）
                model_input = torch.nan_to_num(model_input, nan=0.0, posinf=1e2, neginf=-1e2)
                # 进一步稳健化：对图像侧输入做夹断，避免异常幅值导致数值爆炸
                model_input = torch.clamp(model_input, -5.0, 5.0)
                enc_states = torch.nan_to_num(enc_states, nan=0.0, posinf=1e2, neginf=-1e2)
                pool_proj = torch.nan_to_num(pool_proj, nan=0.0, posinf=1e2, neginf=-1e2)

                with autocast(dtype=torch.bfloat16):
                    pred = model(
                        hidden_states=model_input_seq,
                        encoder_hidden_states=enc_states,
                        pooled_projections=pool_proj,
                        guidance=guidance,
                        timestep=timesteps,
                        txt_ids=txt_ids, # 传入 [B, 77, 2]
                        img_ids=img_ids, # 传入 [B, 4096, 2]
                        return_dict=False
                    )[0]

                if global_step % 100 == 0:
                    print(f"模型输出 pred shape: {pred.shape}")
                    print(f"目标 latent_target shape: {latent_target.shape}")

                # 数值稳定性检查
                if not torch.isfinite(pred).all():
                    print("❌ 检测到 pred 含有非有限值 (NaN/Inf)，跳过此步")
                    continue

                # 模型输出直接就是图像部分
                img_pred = pred  # [B, img_seq_len, out_channels] = [1, 4096, 64]
                
                if global_step % 100 == 0:
                    print(f"图像输出 img_pred shape: {img_pred.shape}")

                # 重新整形为图像格式 [B, C, H, W]
                B = img_pred.shape[0]  # 1
                img_seq_len = img_pred.shape[1]  # 4096
                out_channels = img_pred.shape[2]  # 64
                H = W = int(img_seq_len ** 0.5)  # sqrt(4096) = 64
                
                img_pred_reshaped = img_pred.permute(0, 2, 1).reshape(B, out_channels, H, W)
                
                if global_step % 100 == 0:
                    print(f"重新整形后 img_pred_reshaped shape: {img_pred_reshaped.shape}")

                # 只取前16个通道来匹配目标（因为目标是16通道）
                img_pred_matched = img_pred_reshaped[:, :16, :, :]  # [1, 16, 64, 64]
                
                if global_step % 100 == 0:
                    print(f"通道匹配后 img_pred_matched shape: {img_pred_matched.shape}")
                
                # 检查NaN值
                if torch.isnan(img_pred_matched).any():
                    print("❌ 警告：模型输出包含NaN值！")
                    continue
                
                if torch.isnan(latent_target).any():
                    print("❌ 警告：目标值包含NaN值！")
                    continue
                
                with autocast(dtype=torch.bfloat16):
                    # 差异加权 MSE（温和版 + 第二阶段热身）：
                    # 第二阶段切换后的前300步，进一步减弱权重强度以提高数值稳定性
                    with torch.no_grad():
                        delta = (latent_target - latent_source).abs()  # [B, 16, H, W]
                        delta = torch.clamp(delta - delta.mean(), -2.0, 2.0)
                        warmup = (not self.is_stage1) and (global_step < self.stage1_steps + 300)
                        base_scale = 1.0 if warmup else 2.0  # 热身期更温和
                        slope = 2.0 if warmup else 3.0
                        weight = 1.0 + base_scale * torch.sigmoid(slope * delta)
                    weighted_mse = ((img_pred_matched - latent_target) ** 2) * weight
                    loss = weighted_mse.mean() / gradient_accumulation_steps
                    # 兜底：如出现非有限值，回退到纯 MSE
                    if not torch.isfinite(loss):
                        loss = F.mse_loss(img_pred_matched, latent_target, reduction='mean')
                        loss = loss / gradient_accumulation_steps
                    
                    # 检查损失值是否为NaN
                    if torch.isnan(loss):
                        print("❌ 损失值为NaN，跳过此步")
                        continue
                    
                    if global_step % 100 == 0:
                        print(f"✅ 损失计算正常: {loss.item():.6f}")

                if global_step % 100 == 0:
                    print(f"🎉 前向传播和损失计算成功！Loss: {loss.item():.6f}")
                    print("开始反向传播...")
                
                # 反向传播
                scaler.scale(loss).backward()

                # 梯度裁剪，防止梯度爆炸
                try:
                    scaler.unscale_(optimizer)
                    max_norm = 0.5 if ((not self.is_stage1) and (global_step < self.stage1_steps + 300)) else 1.0
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(proj_hidden.parameters()) + list(proj_pooled.parameters()),
                        max_norm=max_norm
                    )
                except Exception:
                    pass

                # 只在累积步数达到时才更新优化器
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # 额外稳健化：限制 x_embedder 权重范数，防止爆炸
                    try:
                        if hasattr(model, 'x_embedder'):
                            with torch.no_grad():
                                w = model.x_embedder.weight
                                norm = w.norm()
                                max_norm = 60.0
                                if torch.isfinite(norm) and norm.item() > max_norm:
                                    w.mul_(max_norm / (norm + 1e-6))
                    except Exception:
                        pass
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                if global_step % 10 == 0:
                    print(f"Epoch {epoch} Step {global_step} Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                if global_step % 100 == 0:
                    try:
                        print(f"x_embedder.weight.norm(): {model.x_embedder.weight.norm().item():.6f}")
                        print(f"proj_hidden.weight.norm(): {proj_hidden.weight.norm().item():.6f}")
                        print(f"当前学习率: {[g['lr'] for g in optimizer.param_groups]}")
                    except Exception:
                        pass
                
                # 采样功能（包括 step=0），输出到带时间戳的 training_folder/samples
                if global_step % 1000 == 0:
                    self.sample_images(model, vae, text_encoder, text_encoder_2, 
                                     clip_tokenizer, t5_tokenizer, proj_hidden, proj_pooled, 
                                     global_step, device)
                    
                # 模型保存：输出到带时间戳的 training_folder/checkpoints
                if global_step % 2000 == 0 and global_step > 0:
                    self.save_model(model, vae, text_encoder, text_encoder_2, 
                                  clip_tokenizer, t5_tokenizer, proj_hidden, proj_pooled, 
                                  global_step, device)
                    
                global_step += 1
                if global_step >= total_steps:  # 训练到指定步数
                    print(f"✅ 训练完成！共训练 {total_steps} 步")
                    break
            epoch += 1
            # while 循环会根据 global_step 决定是否继续
                    
        print("训练完成")
    
    def _freeze_all_except_projection(self, model):
        """冻结除投影层外的所有参数"""
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 只解冻 x_embedder
        for param in model.x_embedder.parameters():
            param.requires_grad = True
            
        print(f"❄️  第一阶段：冻结全模型，只训练 x_embedder")
    
    def _unfreeze_all_parameters(self, model):
        """解冻所有参数（第二阶段）"""
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
            
        print(f"🔥 第二阶段：解冻全模型参数")

    def _unfreeze_last_n_blocks(self, model, n: int = 4):
        """仅解冻最后 n 个 block 与 x_embedder，降低阶段2显存压力"""
        # 先全部冻结
        for p in model.parameters():
            p.requires_grad = False
        # 始终解冻 x_embedder
        if hasattr(model, 'x_embedder'):
            for p in model.x_embedder.parameters():
                p.requires_grad = True
        # 尝试找到 blocks 列表属性
        blocks_attr = None
        for attr in ['transformer_blocks', 'layers', 'blocks']:
            if hasattr(model, attr):
                blocks_attr = attr
                break
        if blocks_attr is not None:
            blocks = getattr(model, blocks_attr)
            try:
                total = len(blocks)
                start = max(0, total - n)
                for i in range(start, total):
                    for p in blocks[i].parameters():
                        p.requires_grad = True
                print(f"🔥 第二阶段：仅解冻最后 {n} 个block（attr={blocks_attr}，共{total}个）")
                return True
            except Exception:
                pass
        # 回退：解冻全部
        for p in model.parameters():
            p.requires_grad = True
        print("⚠️ 未找到可识别的blocks属性，已回退为解冻全模型")
        return False
    
    def update_training_stage(self, current_step: int, model, optimizer):
        """
        更新训练阶段
        Args:
            current_step: 当前训练步数
            model: 模型
            optimizer: 优化器
        """
        self.current_step = current_step
        
        if self.two_stage_training:
            # 检查是否需要切换到第二阶段
            if self.is_stage1 and current_step >= self.stage1_steps:
                print(f"🔄 切换到第二阶段训练 (步数: {current_step})")
                # 仅解冻最后4个 block 以降低阶段2显存
                ok = self._unfreeze_last_n_blocks(model, n=4)
                if not ok:
                    self._unfreeze_all_parameters(model)
                self.is_stage1 = False
                
                # 更新学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.stage2_lr
                print(f"📈 学习率更新为: {self.stage2_lr}")
                
                return True  # 返回True表示阶段切换
        
        return False
    
    def sample_images(self, model, vae, text_encoder, text_encoder_2, 
                     clip_tokenizer, t5_tokenizer, proj_hidden, proj_pooled, 
                     step, device):
        """采样生成图像"""
        print(f"🎨 开始采样 (步数: {step})")
        
        # 采样配置
        prompts = [
            "add a sofa",
            "add a table", 
            "add a bed",
            "add furniture"
        ]
        
        # 使用固定的测试图像
        test_image_path = "/cloud/cloud-ssd1/test.png"
        if not os.path.exists(test_image_path):
            print(f"⚠️ 测试图像不存在: {test_image_path}")
            return
            
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # 加载测试图像
            test_image = Image.open(test_image_path).convert('RGB')
            test_image = test_image.resize((512, 512))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            test_tensor = transform(test_image).unsqueeze(0).to(device)
            
            # 创建纯白控制图像
            white_image = torch.ones_like(test_tensor)
            
            # VAE编码
            with torch.no_grad():
                if torch.cuda.device_count() > 1:
                    test_latent = vae.encode(test_tensor.to('cuda:1')).latent_dist.sample().to('cuda:0')
                    white_latent = vae.encode(white_image.to('cuda:1')).latent_dist.sample().to('cuda:0')
                else:
                    test_latent = vae.encode(test_tensor).latent_dist.sample()
                    white_latent = vae.encode(white_image).latent_dist.sample()
            
            # 生成4张图像
            for i, prompt in enumerate(prompts):
                # 文本编码：统一在 CPU，上投影前再搬到 GPU
                with torch.no_grad():
                    text_device = 'cpu'
                    clip_ids = clip_tokenizer(
                        prompt, padding='max_length',
                        max_length=clip_tokenizer.model_max_length,
                        truncation=True, return_tensors='pt'
                    ).input_ids.to(text_device)
                    clip_out = text_encoder(
                        clip_ids, output_hidden_states=True, return_dict=True
                    )
                    hidden_states = clip_out.last_hidden_state  # CPU
                    pooled_state = clip_out.text_embeds          # CPU
                    
                    t5_ids = t5_tokenizer(
                        prompt, padding='max_length',
                        max_length=t5_tokenizer.model_max_length,
                        truncation=True, return_tensors='pt'
                    ).input_ids.to(text_device)
                    t5_out = text_encoder_2(
                        t5_ids, output_hidden_states=True, return_dict=True
                    )
                    t5_hidden_states = t5_out.last_hidden_state

                # 投影（GPU）
                enc_states = proj_hidden(hidden_states.to(device))
                pool_proj = proj_pooled(pooled_state.to(device))
                
                # 准备模型输入
                test_latent_16 = test_latent[:, :16, :, :]
                white_latent_16 = white_latent[:, :16, :, :]
                
                # 拼接与训练对齐：原图16 + 控制图16 = 32通道
                model_input = torch.cat([test_latent_16, white_latent_16], dim=1)
                
                # 补零到64通道
                if model_input.shape[1] < 64:
                    pad = torch.zeros((model_input.shape[0], 64 - model_input.shape[1], 
                                     model_input.shape[2], model_input.shape[3]), 
                                    device=model_input.device, dtype=model_input.dtype)
                    model_input = torch.cat([model_input, pad], dim=1)
                
                # 模型前向
                B, C, H, W = model_input.shape
                model_input_permuted = model_input.permute(0, 2, 3, 1).contiguous()
                model_input_seq = model_input_permuted.reshape(model_input_permuted.shape[0], -1, model_input_permuted.shape[-1])
                
                # 构造位置编码
                txt_ids = torch.zeros(B, enc_states.shape[1], 2, device=device, dtype=torch.long)
                for b in range(B):
                    for j in range(enc_states.shape[1]):
                        txt_ids[b, j] = torch.tensor([j // W, j % W], device=device)
                
                y_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()
                x_coords = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).flatten()
                img_ids_single = torch.stack([y_coords, x_coords], dim=1)
                img_ids = img_ids_single.unsqueeze(0).expand(B, -1, -1)
                
                timesteps = torch.zeros(B, dtype=torch.float32, device=device)
                guidance = timesteps.clone()
                
                with torch.no_grad():
                    pred = model(
                        hidden_states=model_input_seq,
                        encoder_hidden_states=enc_states,
                        pooled_projections=pool_proj,
                        guidance=guidance,
                        timestep=timesteps,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False
                    )[0]
                
                # 处理输出
                img_pred = pred
                B, img_seq_len, out_channels = img_pred.shape
                H = W = int(img_seq_len ** 0.5)
                img_pred_reshaped = img_pred.permute(0, 2, 1).reshape(B, out_channels, H, W)
                img_pred_matched = img_pred_reshaped[:, :16, :, :]
                
                # VAE解码
                with torch.no_grad():
                    if torch.cuda.device_count() > 1:
                        decoded = vae.decode(img_pred_matched.to('cuda:1')).sample.to('cuda:0')
                    else:
                        decoded = vae.decode(img_pred_matched).sample
                
                # 保存图像
                import torchvision.utils as vutils
                # 使用带时间戳的训练根目录
                output_root = getattr(self, 'training_folder', None)
                if output_root is None:
                    output_root = getattr(self.job, 'training_folder', "/cloud/cloud-ssd1/training_output")
                output_dir = os.path.join(output_root, "samples")
                os.makedirs(output_dir, exist_ok=True)
                
                # 创建对比图：原图 + 结果
                comparison = torch.cat([test_tensor, decoded], dim=3)  # 水平拼接
                
                # 添加文本到图像
                from PIL import Image, ImageDraw, ImageFont
                comparison_pil = vutils.make_grid(comparison, nrow=1, padding=0, normalize=True)
                comparison_pil = transforms.ToPILImage()(comparison_pil)
                
                # 添加文本
                draw = ImageDraw.Draw(comparison_pil)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), f"Step {step} - {prompt}", fill="white", font=font)
                
                # 保存
                output_path = f"{output_dir}/step_{step:06d}_sample_{i:02d}.jpg"
                comparison_pil.save(output_path)
                print(f"💾 保存采样图像: {output_path}")
                
        except Exception as e:
            print(f"❌ 采样失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_model(self, model, vae, text_encoder, text_encoder_2, 
                  clip_tokenizer, t5_tokenizer, proj_hidden, proj_pooled, 
                  step, device):
        """保存模型"""
        print(f"💾 保存模型 (步数: {step})")
        
        try:
            # 使用带时间戳的训练根目录
            output_root = getattr(self, 'training_folder', None)
            if output_root is None:
                output_root = getattr(self.job, 'training_folder', "/cloud/cloud-ssd1/training_output")
            output_dir = os.path.join(output_root, f"checkpoints/step_{step:06d}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存transformer
            model.save_pretrained(f"{output_dir}/transformer")
            
            # 保存VAE
            vae.save_pretrained(f"{output_dir}/vae")
            
            # 保存文本编码器
            text_encoder.save_pretrained(f"{output_dir}/text_encoder")
            text_encoder_2.save_pretrained(f"{output_dir}/text_encoder_2")
            
            # 保存tokenizer
            clip_tokenizer.save_pretrained(f"{output_dir}/tokenizer")
            t5_tokenizer.save_pretrained(f"{output_dir}/tokenizer_2")
            
            # 保存投影层
            torch.save(proj_hidden.state_dict(), f"{output_dir}/proj_hidden.pt")
            torch.save(proj_pooled.state_dict(), f"{output_dir}/proj_pooled.pt")
            
            # 保存训练信息
            import yaml
            meta = {
                'step': step,
                'two_stage_training': self.two_stage_training,
                'stage1_steps': self.stage1_steps,
                'stage1_lr': self.stage1_lr,
                'stage2_lr': self.stage2_lr,
                'is_stage1': self.is_stage1
            }
            with open(f"{output_dir}/meta.yaml", 'w') as f:
                yaml.dump(meta, f)
            
            print(f"✅ 模型保存完成: {output_dir}")
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            import traceback
            traceback.print_exc()