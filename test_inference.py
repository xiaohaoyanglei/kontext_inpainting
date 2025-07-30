#!/usr/bin/env python3
"""
简化版FLUX Fill推理测试脚本
用于快速验证训练好的模型
"""

import torch
from PIL import Image
import os
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline


def test_model(checkpoint_path):
    """测试训练好的模型"""
    print("🚀 开始测试FLUX Fill模型...")
    
    # 1. 加载模型
    print(f"📁 加载checkpoint: {checkpoint_path}")
    try:
        from safetensors.torch import load_file
        
        # 基础模型路径
        base_model = "/cloud/cloud-ssd1/FLUX.1-Fill-dev"
        
        # 加载基础pipeline
        pipeline = FluxFillPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16
        )
        
        # 如果是AI-toolkit的checkpoint，加载训练的transformer权重
        if os.path.exists(checkpoint_path) and checkpoint_path != base_model:
            print(f"⚡ 加载AI-toolkit训练权重: {checkpoint_path}")
            
            # AI-toolkit的checkpoint格式：/path/to/checkpoint/transformer/
            transformer_path = os.path.join(checkpoint_path, "transformer")
            
            if os.path.exists(transformer_path):
                # 查找transformer权重文件
                transformer_files = [f for f in os.listdir(transformer_path) if f.endswith('.safetensors')]
                if transformer_files:
                    transformer_weight_path = os.path.join(transformer_path, transformer_files[0])
                    print(f"   📄 加载transformer权重: {transformer_weight_path}")
                    
                    # 加载权重
                    state_dict = load_file(transformer_weight_path)
                    
                    # 过滤和加载到transformer
                    transformer_state_dict = {}
                    for key, value in state_dict.items():
                        # 移除可能的前缀
                        clean_key = key.replace("transformer.", "").replace("model.", "")
                        transformer_state_dict[clean_key] = value
                    
                    # 加载到pipeline的transformer
                    pipeline.transformer.load_state_dict(transformer_state_dict, strict=False)
                    print("   ✅ Transformer权重加载成功")
                else:
                    print("   ⚠️ 未找到transformer权重文件，使用基础模型")
            else:
                print("   ⚠️ Transformer目录不存在，使用基础模型")
        
        pipeline.to("cuda")
        print("✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 创建测试图像
    print("🎨 创建测试图像...")
    try:
        # 创建简单的测试图像
        source_image = Image.new('RGB', (512, 512), color='lightblue')
        
        # 创建mask (中心区域需要修复)
        mask_image = Image.new('L', (512, 512), color=0)  # 黑色背景
        
        # 在中心画一个白色方块作为需要修复的区域
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_image)
        draw.rectangle([200, 200, 312, 312], fill=255)  # 白色区域
        
        print("✅ 测试图像创建成功")
        
        # 保存测试图像用于检查
        source_image.save("test_source.png")
        mask_image.save("test_mask.png")
        print("💾 测试图像已保存: test_source.png, test_mask.png")
        
    except Exception as e:
        print(f"❌ 测试图像创建失败: {e}")
        return False
    
    # 3. 运行推理
    print("🔮 运行推理测试...")
    try:
        prompt = "fill the area with beautiful flowers"
        
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image=source_image,
                mask_image=mask_image,
                num_inference_steps=10,  # 少步数快速测试
                guidance_scale=4.0,
            )
        
        # 保存结果
        result_image = result.images[0]
        result_image.save("test_result.png")
        
        print("✅ 推理测试成功!")
        print("💾 结果已保存: test_result.png")
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_model():
    """测试基础FLUX Fill模型是否工作"""
    print("🧪 测试基础FLUX Fill模型...")
    
    base_model_path = "/cloud/cloud-ssd1/FLUX.1-Fill-dev"
    
    if not os.path.exists(base_model_path):
        print(f"❌ 基础模型不存在: {base_model_path}")
        return False
    
    return test_model(base_model_path)


if __name__ == "__main__":
    import sys
    
    print("🔍 FLUX Fill推理测试工具")
    
    if len(sys.argv) > 1:
        # 测试指定的checkpoint
        checkpoint_path = sys.argv[1]
        print(f"测试checkpoint: {checkpoint_path}")
        success = test_model(checkpoint_path)
    else:
        # 测试基础模型
        print("测试基础FLUX Fill模型...")
        success = test_base_model()
    
    if success:
        print("\n🎉 测试成功! 模型工作正常")
        print("📋 下一步:")
        print("  1. 检查生成的测试图像")
        print("  2. 使用真实图像进行测试")
        print("  3. 调整推理参数优化效果")
    else:
        print("\n💥 测试失败! 请检查模型和环境")
        print("🔧 建议:")
        print("  1. 确认模型路径正确")
        print("  2. 检查CUDA内存是否足够")
        print("  3. 验证依赖库版本") 