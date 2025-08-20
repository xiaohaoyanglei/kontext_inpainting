#!/usr/bin/env python3
"""
Kontext-inpaint 两阶段训练脚本
基于现有的 ai-toolkit 框架，实现：
1. 第一阶段：只训练 32→hidden 投影层
2. 第二阶段：全模型 fine-tune
"""

import os
import sys
import torch
from collections import OrderedDict
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc


class KontextInpaintTrainer:
    """Kontext-inpaint 两阶段训练器"""
    
    def __init__(self, config_path):
        """
        初始化训练器
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.accelerator = get_accelerator()
        
    def run_training(self):
        """运行两阶段训练"""
        print_acc("🎭 开始 Kontext-inpaint 两阶段训练")
        print_acc(f"📁 配置文件: {self.config_path}")
        
        try:
            # 使用 ai-toolkit 的标准 job 系统
            job = get_job(self.config_path)
            
            # 获取训练过程
            if len(job.process) > 0:
                train_process = job.process[0]
                
                # 检查是否是 Kontext-inpaint 模型
                if hasattr(train_process, 'sd') and hasattr(train_process.sd, 'update_training_stage'):
                    print_acc("✅ 检测到 Kontext-inpaint 模型，启用两阶段训练监控")
                    
                    # 添加训练步数回调
                    original_train_step = getattr(train_process, 'train_step', None)
                    if original_train_step:
                        def enhanced_train_step(*args, **kwargs):
                            # 调用原始训练步骤
                            result = original_train_step(*args, **kwargs)
                            
                            # 更新训练阶段
                            current_step = getattr(train_process, 'step', 0)
                            if hasattr(train_process.sd, 'update_training_stage'):
                                stage_changed = train_process.sd.update_training_stage(current_step)
                                
                                if stage_changed:
                                    # 阶段切换时需要重建优化器
                                    print_acc("🔄 阶段切换，重建优化器...")
                                    self._rebuild_optimizer(train_process)
                            
                            return result
                        
                        # 替换训练步骤方法
                        train_process.train_step = enhanced_train_step
            
            # 运行训练
            job.run()
            job.cleanup()
            
            print_acc("✅ Kontext-inpaint 训练完成！")
            
        except Exception as e:
            print_acc(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _rebuild_optimizer(self, train_process):
        """重建优化器（阶段切换时）"""
        try:
            if hasattr(train_process, 'sd') and hasattr(train_process.sd, 'get_trainable_parameters'):
                # 获取当前阶段的可训练参数
                trainable_params = train_process.sd.get_trainable_parameters()
                
                # 获取当前阶段的学习率
                if hasattr(train_process.sd, 'get_current_learning_rate'):
                    lr = train_process.sd.get_current_learning_rate()
                else:
                    lr = 5e-5  # 默认第二阶段学习率
                
                print_acc(f"🔧 重建优化器:")
                print_acc(f"   - 可训练参数数量: {len(trainable_params)}")
                print_acc(f"   - 学习率: {lr}")
                
                # 重建优化器
                if hasattr(train_process, 'optimizer'):
                    optimizer_class = type(train_process.optimizer)
                    train_process.optimizer = optimizer_class(trainable_params, lr=lr)
                    print_acc("✅ 优化器重建完成")
                
        except Exception as e:
            print_acc(f"⚠️ 优化器重建失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kontext-inpaint 两阶段训练")
    parser.add_argument("config", help="训练配置文件路径")
    parser.add_argument("--log", help="日志文件路径")
    
    args = parser.parse_args()
    
    if args.log:
        from toolkit.print import setup_log_to_file
        setup_log_to_file(args.log)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print_acc(f"❌ 配置文件不存在: {args.config}")
        return
    
    # 创建训练器并运行
    trainer = KontextInpaintTrainer(args.config)
    trainer.run_training()


if __name__ == "__main__":
    main()