# 故障排除指南

## 🔧 常见问题解决

### 1. 配置文件错误

**问题**: `yaml.parser.ParserError`
```bash
yaml.parser.ParserError: while parsing a block mapping
```

**解决方案**:
- 检查 YAML 格式缩进是否正确
- 确保没有 Tab 字符，只使用空格
- 验证冒号后有空格

**验证命令**:
```bash
python -c "import yaml; yaml.safe_load(open('config/train_flux_inpainting.yaml'))"
```

### 2. 数据路径问题

**问题**: `FileNotFoundError: data path not found`

**解决方案**:
1. 检查路径是否存在:
```bash
ls data/target_images/
ls data/masked_images/
ls data/masks/
```

2. 运行数据迁移:
```bash
python migrate_your_data.py
```

3. 确保文件命名一致 (同名的图像和掩码)

### 3. 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
```yaml
# 方案A: 降低批次大小
train:
  batch_size: 1
  gradient_accumulation_steps: 2

# 方案B: 降低分辨率  
datasets:
  - resolution: [512]  # 只用 512x512

# 方案C: 启用更多优化
model:
  quantize: true
  low_cpu_mem_usage: true
train:
  gradient_checkpointing: true
```

### 4. 模型下载失败

**问题**: `Repository not found` 或下载速度慢

**解决方案**:
```bash
# 方案A: 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案B: 手动下载到本地
# 然后修改配置:
model:
  name_or_path: "./local_models/FLUX.1-dev"
```

### 5. Loss 为 NaN

**问题**: 训练开始后 loss 变为 NaN

**检查项**:
- ✅ 数据格式正确 (图像 0-255, 掩码 0-1)
- ✅ 学习率不要过大 (推荐 1e-4)
- ✅ 混合精度设置正确

**解决方案**:
```yaml
train:
  lr: 5e-5  # 降低学习率
  dtype: bf16  # 使用 bf16 而不是 fp16
```

### 6. 训练速度慢

**检查项**:
- ✅ `cache_latents_to_disk: true` 
- ✅ `quantize: true`
- ✅ `optimizer: "adamw8bit"`

**优化建议**:
```yaml
# 完整优化配置
datasets:
  - cache_latents_to_disk: true  # 🔥 最重要
    
train:
  gradient_checkpointing: true
  optimizer: "adamw8bit"  
  dtype: bf16

model:
  quantize: true
```

### 7. 采样图像质量差

**问题**: 训练过程中生成的预览图质量不好

**调优方案**:
```yaml
sample:
  guidance_scale: 4  # 调整引导强度 (1-8)
  sample_steps: 20   # 增加采样步数
  prompts:
    - "high quality detailed image"  # 改善提示词
```

## 🚨 紧急恢复

### 训练中断恢复
```bash
# AI-toolkit 自动从最新检查点恢复
python run.py config/train_flux_inpainting.yaml
```

### 检查点损坏
```bash
# 删除损坏的检查点，从上一个恢复
rm output/flux_inpainting_migrated_v1/step_XXXX.safetensors
```

### 回滚到自定义实现
```bash
# 如果需要临时回滚
git checkout backup/TrainFineTuneProcess.py
```

## 📞 获取帮助

1. **检查日志**: 查看 `output/` 文件夹中的训练日志
2. **验证数据**: 确保图像和掩码正确匹配
3. **测试配置**: 运行 `python test_standard_trainer.py`
4. **社区支持**: AI-toolkit GitHub Issues

---

**记住**: 大部分问题都是配置或数据格式问题，仔细检查配置文件和数据路径通常能解决！
