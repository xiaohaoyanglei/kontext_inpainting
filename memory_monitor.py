#!/usr/bin/env python3
"""
外挂显存监控脚本
在训练过程中实时监控GPU显存和系统内存使用情况
"""

import time
import psutil
import torch
import threading
import os
import signal
import sys

class MemoryMonitor:
    def __init__(self, interval=5):
        """
        初始化显存监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        
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
    
    def get_system_memory_info(self):
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'used': memory.used / 1024**3,  # GB
            'available': memory.available / 1024**3,  # GB
            'total': memory.total / 1024**3,  # GB
            'percent': memory.percent
        }
    
    def print_memory_status(self):
        """打印内存状态"""
        timestamp = time.strftime("%H:%M:%S")
        
        # GPU显存
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            print(f"[{timestamp}] 💾 GPU显存: "
                  f"已分配 {gpu_info['allocated']:.2f}GB, "
                  f"已保留 {gpu_info['reserved']:.2f}GB, "
                  f"可用 {gpu_info['free']:.2f}GB, "
                  f"总计 {gpu_info['total']:.2f}GB")
        
        # 系统内存
        sys_info = self.get_system_memory_info()
        print(f"[{timestamp}] 💻 系统内存: "
              f"已用 {sys_info['used']:.2f}GB, "
              f"可用 {sys_info['available']:.2f}GB, "
              f"总计 {sys_info['total']:.2f}GB "
              f"({sys_info['percent']:.1f}%)")
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"[{timestamp}] 🔥 CPU使用率: {cpu_percent:.1f}%")
        
        print("-" * 80)
    
    def monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self.print_memory_status()
                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if self.running:
            print("监控已在运行中")
            return
        
        self.running = True
        print(f"🚀 开始显存监控 (间隔: {self.interval}秒)")
        print("按 Ctrl+C 停止监控")
        print("=" * 80)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        try:
            # 主线程等待
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("\n🛑 显存监控已停止")

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n收到停止信号，正在退出...")
    sys.exit(0)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="显存监控工具")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="监控间隔（秒），默认5秒")
    parser.add_argument("--once", "-o", action="store_true",
                       help="只打印一次内存状态然后退出")
    
    args = parser.parse_args()
    
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor = MemoryMonitor(interval=args.interval)
    
    if args.once:
        # 只打印一次
        monitor.print_memory_status()
    else:
        # 持续监控
        monitor.start()

if __name__ == "__main__":
    main()
