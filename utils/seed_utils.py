"""
随机种子工具函数
用于统一设置所有随机数生成器的种子，确保结果可复现
"""

import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机数生成器的种子，确保结果可复现
    
    Args:
        seed: 随机种子值，默认为42
    """
    # Python内置random模块
    random.seed(seed)
    
    # NumPy随机数生成器
    np.random.seed(seed)
    
    # PyTorch随机数生成器
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子
    
    # 设置PyTorch的确定性行为（可能会影响性能）
    # 注意：这会使得某些操作变慢，但能确保更强的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS workspace config for deterministic GEMM on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        # warn_only avoids hard crash if a rare non-deterministic op appears.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    
    # 设置环境变量（用于某些第三方库）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[Seed] 已设置随机种子为: {seed}")
    print(f"[Seed] Python random, NumPy, PyTorch, CUDA 随机数生成器已统一设置")
    if torch.cuda.is_available():
        print(f"[Seed] CUDA确定性模式已启用（可能会影响性能）")


def worker_init_fn(worker_id):
    """
    DataLoader的worker初始化函数，确保每个worker进程的随机种子不同但可复现
    
    Args:
        worker_id: worker进程的ID
    """
    # 为每个worker设置不同的种子（基于基础种子+worker_id）
    # 这样可以确保不同worker的随机序列不同，但整体可复现
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

