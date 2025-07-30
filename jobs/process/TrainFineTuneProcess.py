from collections import OrderedDict
from jobs import TrainJob
from jobs.process import BaseTrainProcess
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import FluxTransformer2DModel, AutoencoderKL
from toolkit.data_loader import ImageDataset
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from toolkit.models.flux import add_model_gpu_splitter_to_flux
import os

class TrainFineTuneProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job: TrainJob, config: OrderedDict):
        super().__init__(process_id, job, config)

    def run(self):
        # 设置PyTorch显存优化环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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

        # 2. 数据加载器
        dataset = ImageDataset(
            {'include_prompt': True, 'resolution': self.config.get('resolution', 512)},
            source_dir=self.config['source_image_dir'],
            target_dir=self.config['target_image_dir'],
            mask_dir=self.config['mask_dir']
        )
        train_loader = DataLoader(
            dataset, 
            batch_size=self.config.get('batch_size', 1),
            shuffle=True,
            num_workers=0  # 减少多进程显存占用
        )

        # 3. 强制 in_channels=64, patch_size=1
        in_channels = 64
        patch_size = 1
        patch_dim = in_channels * patch_size * patch_size
        print(f"强制使用 in_channels={in_channels}, patch_size={patch_size}, patch_dim={patch_dim}")
        
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
        
        # 5. 自动推断 hidden_size
        if hasattr(model.config, 'joint_attention_dim') or 'joint_attention_dim' in model.config:
            hidden_size = getattr(model.config, 'joint_attention_dim', None) or model.config.get('joint_attention_dim', None)
            print(f"使用 joint_attention_dim 作为 hidden_size: {hidden_size}")
        elif (hasattr(model.config, 'num_attention_heads') or 'num_attention_heads' in model.config) and (hasattr(model.config, 'attention_head_dim') or 'attention_head_dim' in model.config):
            num_heads = getattr(model.config, 'num_attention_heads', None) or model.config.get('num_attention_heads', None)
            head_dim = getattr(model.config, 'attention_head_dim', None) or model.config.get('attention_head_dim', None)
            hidden_size = num_heads * head_dim
            print(f"使用 num_attention_heads * attention_head_dim 作为 hidden_size: {hidden_size}")
        else:
            raise ValueError(f"无法在 config 中找到 hidden_size 相关字段，所有 key: {list(model.config.keys())}")
        
        # 调试：检查模型的实际 hidden_size
        try:
            # 检查第一个 transformer block 的 norm 层
            first_block = model.transformer_blocks[0] if hasattr(model, 'transformer_blocks') else None
            if first_block and hasattr(first_block, 'norm1'):
                actual_hidden_size = first_block.norm1.norm.normalized_shape[0]
                print(f"检测到模型实际 hidden_size: {actual_hidden_size}")
                if actual_hidden_size != hidden_size:
                    print(f"警告：配置的 hidden_size ({hidden_size}) 与实际不符，使用实际值 {actual_hidden_size}")
                    hidden_size = actual_hidden_size
        except Exception as e:
            print(f"无法检测模型实际 hidden_size: {e}")
            
        model.x_embedder = torch.nn.Linear(patch_dim, hidden_size, bias=True).to(device)
        torch.nn.init.trunc_normal_(model.x_embedder.weight, std=0.02)
        model.to(device)
        
        # 启用梯度检查点以大幅减少显存使用
        model.enable_gradient_checkpointing()
        print("启用梯度检查点以减少显存使用")
        
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

        print(f"Flux 模型加载完成，patch_size={patch_size}，patch_dim={patch_dim}, hidden_size={hidden_size}, in_channels={in_channels}")
        print(f"x_embedder.weight.shape: {model.x_embedder.weight.shape}")
        print(f"model.config keys: {list(model.config.keys())}")
        
        # 调试信息：打印 axes_dims_rope 的实际值
        print(f"模型的 axes_dims_rope: {model.config.get('axes_dims_rope', 'NOT_FOUND')}")
        print(f"pos_embed.axes_dim: {getattr(model.pos_embed, 'axes_dim', 'NOT_FOUND')}")

        # 6. 加载文本编码器与分词器 - 如果有双GPU，也放到GPU 1
        if torch.cuda.device_count() > 1:
            text_encoder_device = 'cuda:1'
            print("📍 文本编码器放置到 GPU 1，进一步释放 GPU 0 显存")
        else:
            text_encoder_device = device
            
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.config['text_encoder_path'], token=token
        ).eval().to(text_encoder_device)
        
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", token=token
        )
        
        # 文本投影层设备设置 - 跟随文本编码器
        proj_hidden = torch.nn.Linear(
            text_encoder.config.hidden_size,
            4096  # 使用 joint_attention_dim，不是实际的 hidden_size
        ).to(text_encoder_device)
        pooled_projection_dim = model.config.get('pooled_projection_dim', 768)
        proj_pooled = torch.nn.Linear(
            text_encoder.config.projection_dim,
            pooled_projection_dim
        ).to(text_encoder_device)
        
        # 正确初始化投影层权重，防止NaN
        print("🔧 初始化投影层权重...")
        torch.nn.init.xavier_uniform_(proj_hidden.weight)
        torch.nn.init.zeros_(proj_hidden.bias)
        torch.nn.init.xavier_uniform_(proj_pooled.weight)
        torch.nn.init.zeros_(proj_pooled.bias)
        print("✅ 投影层权重初始化完成")

        # 7. 优化器
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                list(model.parameters()) + list(proj_hidden.parameters()) + list(proj_pooled.parameters()),
                lr=self.config.get('lr', 1e-4)
            )
            print("使用 AdamW8bit 优化器以节省显存")
        except ImportError:
            optimizer = AdamW(
                list(model.parameters()) + list(proj_hidden.parameters()) + list(proj_pooled.parameters()),
                lr=self.config.get('lr', 1e-4),
                eps=1e-8,
                weight_decay=0.01
            )
            print("使用标准 AdamW 优化器")

        # 8. 混合精度训练
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print("启用混合精度训练以减少显存使用")
        
        # 9. 训练主循环
        num_epochs = self.config.get('num_epochs', 1)
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_loader):
                # 批次数据移至 GPU
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                optimizer.zero_grad()
                
                # VAE 编码
                with torch.no_grad():
                    # 如果VAE在GPU 1，需要移动数据
                    if torch.cuda.device_count() > 1:
                        masked_img = batch['masked_image'].to('cuda:1')
                        target_img = batch['image'].to('cuda:1')
                        
                        latent_masked = vae.encode(masked_img).latent_dist.sample()
                        latent_target = vae.encode(target_img).latent_dist.sample()
                        
                        # 编码后移回GPU 0
                        latent_masked = latent_masked.to('cuda:0')
                        latent_target = latent_target.to('cuda:0')
                        
                        del masked_img, target_img  # 清理GPU 1上的数据
                    else:
                        latent_masked = vae.encode(batch['masked_image']).latent_dist.sample()
                        latent_target = vae.encode(batch['image']).latent_dist.sample()
                    torch.cuda.empty_cache()  # 清理 VAE 编码后的显存
                    
                noise = torch.randn_like(latent_masked)
                mask = batch['mask']
                print(f"原始mask shape: {mask.shape}")
                if mask.shape[1] > 1:
                    print(f"警告：mask通道数为{mask.shape[1]}，将只取第一个通道")
                    mask = mask[:, :1, :, :]
                mask = F.interpolate(mask, size=latent_masked.shape[-2:], mode='nearest')
                print(f"处理后mask shape: {mask.shape}")
                
                # 只取前16通道 latent_masked 和 noise
                latent_masked = latent_masked[:, :16, :, :]
                noise = noise[:, :16, :, :]
                # mask 仍为1通道
                model_input = torch.cat([latent_masked, noise, mask], dim=1)  # [B, 33, H, W]
                
                # 自动补零到in_channels=64
                if model_input.shape[1] < in_channels:
                    pad = torch.zeros((model_input.shape[0], in_channels - model_input.shape[1], model_input.shape[2], model_input.shape[3]), device=model_input.device, dtype=model_input.dtype)
                    model_input = torch.cat([model_input, pad], dim=1)
                elif model_input.shape[1] > in_channels:
                    model_input = model_input[:, :in_channels, :, :]
                print(f"最终送入模型的shape: {model_input.shape}")
                
                # patchify 检查
                B, C, H, W = model_input.shape
                patches = model_input.unfold(2, 1, 1).unfold(3, 1, 1)
                patches = patches.contiguous().view(B, C, -1, 1, 1)
                patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, C * 1 * 1)
                print(f"patchify 后的 shape: {patches.shape}")
                if H % patch_size != 0 or W % patch_size != 0:
                    raise ValueError(f"H/W必须能被patch_size整除，当前H={H}, W={W}, patch_size={patch_size}")
                if C != in_channels:
                    raise ValueError(f"输入通道数应为{in_channels}，实际为{C}")

                # 文本编码 + 投射
                with torch.no_grad():
                    # 确定文本编码设备
                    if torch.cuda.device_count() > 1:
                        text_device = 'cuda:1'
                        clip_ids = clip_tokenizer(
                            batch['caption'], padding='max_length',
                            max_length=clip_tokenizer.model_max_length,
                            truncation=True, return_tensors='pt'
                        ).input_ids.to(text_device)
                        clip_out = text_encoder(
                            clip_ids, output_hidden_states=True, return_dict=True
                        )
                        hidden_states = clip_out.last_hidden_state
                        pooled_state  = clip_out.text_embeds
                        enc_states = proj_hidden(hidden_states)
                        pool_proj  = proj_pooled(pooled_state)
                        
                        # 移回GPU 0用于模型计算
                        enc_states = enc_states.to('cuda:0')
                        pool_proj = pool_proj.to('cuda:0')
                    else:
                        clip_ids = clip_tokenizer(
                            batch['caption'], padding='max_length',
                            max_length=clip_tokenizer.model_max_length,
                            truncation=True, return_tensors='pt'
                        ).input_ids.to(device)
                        clip_out = text_encoder(
                            clip_ids, output_hidden_states=True, return_dict=True
                        )
                        hidden_states = clip_out.last_hidden_state
                        pooled_state  = clip_out.text_embeds
                        enc_states = proj_hidden(hidden_states)
                        pool_proj  = proj_pooled(pooled_state)
                    torch.cuda.empty_cache()  # 清理文本编码后的显存
                    
                print("pool_proj shape:", pool_proj.shape)  # 应该是 [B, 768]
                print(f"文本编码 hidden_states shape: {hidden_states.shape}")
                print(f"投影后 enc_states shape: {enc_states.shape}")
                txt_actual_seq_len = enc_states.shape[1]  # 实际文本序列长度
                print(f"实际文本序列长度: {txt_actual_seq_len}")

                # 送入模型前 permute 并 reshape 为序列格式
                model_input_permuted = model_input.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                model_input_seq = model_input_permuted.reshape(model_input_permuted.shape[0], -1, model_input_permuted.shape[-1])  # [B, H*W, C]
                print(f"送入 model 前的 shape: {model_input_seq.shape}")

                # 构造 img_ids，txt_ids
                B, C, H, W = model_input.shape  # B=batch_size, H=64, W=64
                
                # txt_ids: 对应文本序列长度，使用简单的位置编码
                txt_ids = torch.zeros(B, txt_actual_seq_len, 2, device=device, dtype=torch.long)
                for b in range(B):
                    for i in range(txt_actual_seq_len):
                        txt_ids[b, i] = torch.tensor([i // W, i % W], device=device)  # 模拟 2D 位置
                
                # img_ids: 对应图像序列长度，使用真实的 2D 网格坐标
                img_seq_len = H * W  # 4096 for 64x64
                y_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()  # [4096]
                x_coords = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).flatten()  # [4096]
                img_ids_single = torch.stack([y_coords, x_coords], dim=1)  # [4096, 2]
                img_ids = img_ids_single.unsqueeze(0).expand(B, -1, -1)  # [B, 4096, 2]
                
                print(f"传入的 txt_ids shape: {txt_ids.shape}")  # [B, 77, 2]
                print(f"传入的 img_ids shape: {img_ids.shape}")  # [B, 4096, 2]
                print(f"模拟 cat 后的 shape: {torch.cat([txt_ids, img_ids], dim=1).shape}")  # [B, 4173, 2]
                print(f"ids 最后一维 (n_axes): {txt_ids.shape[-1]}")
                total_seq_len = txt_ids.shape[1] + img_ids.shape[1]
                print(f"总序列长度: {total_seq_len} (txt: {txt_ids.shape[1]} + img: {img_ids.shape[1]})")

                # Flux 模型前向
                timesteps = torch.zeros(model_input.shape[0], dtype=torch.float32, device=device)  # [B] float32
                guidance = timesteps.clone()  # [B] float32
                
                print("🔍 检查输入数据...")
                
                with autocast():
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

                print(f"模型输出 pred shape: {pred.shape}")
                print(f"目标 latent_target shape: {latent_target.shape}")

                # 模型输出直接就是图像部分
                img_pred = pred  # [B, img_seq_len, out_channels] = [1, 4096, 64]
                print(f"图像输出 img_pred shape: {img_pred.shape}")

                # 重新整形为图像格式 [B, C, H, W]
                B = img_pred.shape[0]  # 1
                img_seq_len = img_pred.shape[1]  # 4096
                out_channels = img_pred.shape[2]  # 64
                H = W = int(img_seq_len ** 0.5)  # sqrt(4096) = 64
                
                img_pred_reshaped = img_pred.permute(0, 2, 1).reshape(B, out_channels, H, W)
                print(f"重新整形后 img_pred_reshaped shape: {img_pred_reshaped.shape}")

                # 只取前16个通道来匹配目标（因为目标是16通道）
                img_pred_matched = img_pred_reshaped[:, :16, :, :]  # [1, 16, 64, 64]
                print(f"通道匹配后 img_pred_matched shape: {img_pred_matched.shape}")

                # 梯度累积
                gradient_accumulation_steps = 16 if torch.cuda.device_count() > 1 else 8
                print(f"🔥 使用{gradient_accumulation_steps}步梯度累积")
                
                with autocast():
                    loss = F.mse_loss(img_pred_matched, latent_target, reduction='mean')
                    loss = loss / gradient_accumulation_steps
                    print(f"✅ 损失计算正常: {loss.item():.6f}")

                print(f"🎉 前向传播和损失计算成功！Loss: {loss.item():.6f}")
                print("开始反向传播...")
                
                # 反向传播
                scaler.scale(loss).backward()

                # 只在累积步数达到时才更新优化器
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                if step % 10 == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                    
                if step >= 19:  # 测试前20步
                    print("✅ 测试完成！前20步训练成功")
                    break
                    
        print("训练完成")