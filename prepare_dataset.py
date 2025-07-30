import os
from datasets import load_dataset
from PIL import Image
import sys

# --- 配置你的路径 ---
# Hugging Face Hub 上的数据集名称
dataset_name = "osunlp/MagicBrush" 

# 你希望在服务器上存放图片的最终文件夹
# 从命令行参数获取基础路径，如果没有则使用默认值
base_output_dir = sys.argv[1] if len(sys.argv) > 1 else "/cloud/cloud-ssd1/my_dataset"
output_source_dir = os.path.join(base_output_dir, "source_images")
output_target_dir = os.path.join(base_output_dir, "target_images")
output_mask_dir = os.path.join(base_output_dir, "masks")

# --- 脚本主逻辑 ---

def prepare_and_save_dataset():
    # 确保输出文件夹存在
    os.makedirs(output_source_dir, exist_ok=True)
    os.makedirs(output_target_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    print(f"正在从 Hugging Face 下载 '{dataset_name}' 数据集...")
    # 加载数据集
    try:
        dataset = load_dataset(dataset_name, split='train') # 我们只用训练集
    except Exception as e:
        print(f"下载数据集失败，请检查网络连接或数据集名称是否正确。错误: {e}")
        return
    
    print("下载完成，开始提取并保存图片和掩码...")
    
    # 遍历数据集并保存文件
    for i, item in enumerate(dataset):
        # 获取图片、掩码和指令文本
        source_image = item.get('source_img')
        target_image = item.get('target_img')
        mask_image = item.get('mask_img')
        instruction = item.get('instruction', '')
        
        # 定义文件名 (使用5位补零的索引，确保文件顺序和对应关系)
        filename = f"{i:05d}.png"
        source_path = os.path.join(output_source_dir, filename)
        target_path = os.path.join(output_target_dir, filename)
        mask_path = os.path.join(output_mask_dir, filename)
        
        # 保存原始图片 (source_img)
        if isinstance(source_image, Image.Image):
            source_image.save(source_path)
        
        # 保存目标图片 (target_img)
        if isinstance(target_image, Image.Image):
            target_image.save(target_path)
        
        # 保存掩码图片
        if isinstance(mask_image, Image.Image):
            # 确保掩码是黑白的（L模式），这是后续处理需要的标准格式
            mask_image.convert("L").save(mask_path)
        
        # 保存指令文本为同名的.txt文件
        if instruction:
            text_filename = f"{i:05d}.txt"
            text_path = os.path.join(output_source_dir, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(instruction)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} / {len(dataset)} 条数据...")
            
    print("\n----------------------------------------")
    print("      数据集准备完成！")
    print("----------------------------------------")
    print(f"原始图片 (source_img) 已保存至: {output_source_dir}")
    print(f"目标图片 (target_img) 已保存至: {output_target_dir}")
    print(f"掩码图片已保存至: {output_mask_dir}")
    print(f"指令文本文件已保存至: {output_source_dir}")
    print("\n现在，你可以在 'config/train_flux_fill.yaml' 文件中使用这些路径了。")

if __name__ == "__main__":
    prepare_and_save_dataset() 