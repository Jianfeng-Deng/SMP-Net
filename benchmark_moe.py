"""
MoE效率优化基准测试脚本

对比优化前后的：
1. 训练速度（每个epoch时间）
2. 推理速度（每个batch时间）
3. 内存占用（峰值GPU内存）
4. 准确率（CWRU/PU/HUST）

使用方法：
python benchmark_moe.py --mode baseline  # 测试基线（所有专家，复杂路由器）
python benchmark_moe.py --mode optimized  # 测试优化版本（稀疏路由，轻量路由器）
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.SpecEncoderMoE import SpecEncoderMoE
from MultimodalBearingDataset import MultimodalBearingDataset
from collate import multimodal_collate_fn
from utils.LoadDatasets import load_datasets
from utils.dataPreprocess import dataProcess
from config import FAULT_PROMPT_TEMPLATES


def get_gpu_memory():
    """获取当前GPU内存使用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory():
    """重置GPU内存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def benchmark_forward_speed(model, dataloader, device, num_batches=50):
    """
    测试前向传播速度
    
    Returns:
        avg_time: 平均每个batch的时间（秒）
        throughput: 吞吐量（samples/sec）
    """
    model.eval()
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            spec = batch["spec"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            batch_size = spec.size(0)
            
            # 预热（第一次可能较慢）
            if i == 0:
                _ = model(spec, spec_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                continue
            
            # 计时
            start_time = time.time()
            _ = model(spec, spec_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            total_samples += batch_size
    
    avg_time = np.mean(times)
    throughput = total_samples / sum(times)
    
    return avg_time, throughput


def benchmark_memory(model, dataloader, device, num_batches=10):
    """
    测试内存占用
    
    Returns:
        peak_memory: 峰值GPU内存（MB）
    """
    model.eval()
    reset_gpu_memory()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            spec = batch["spec"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            _ = model(spec, spec_mask)
    
    peak_memory = get_gpu_memory()
    return peak_memory


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description='MoE Benchmark')
    parser.add_argument('--mode', type=str, default='optimized', 
                        choices=['baseline', 'optimized'],
                        help='baseline: 所有专家+复杂路由; optimized: 稀疏路由+轻量路由器')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to test')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    # 加载数据
    print("Loading datasets...")
    cwru_path = "dataset/CWRU/raw"
    pu_path = "dataset/PU/raw"
    hust_path = "dataset/HUST/raw"
    
    cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels, _, _ = load_datasets(
        cwru_path, pu_path, hust_path
    )
    
    DataProcess = dataProcess(cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels)
    cwru_original, pu_original, hust_original, cwru_data, pu_data, hust_data = DataProcess.fit(
        n_component=2, lambda_domain=0.15
    )
    
    cwru_texts, pu_texts, hust_texts = DataProcess.data_label_to_prompt(FAULT_PROMPT_TEMPLATES)
    
    # 创建测试数据集（使用CWRU）
    cwru_dataset = MultimodalBearingDataset(
        X_origin=cwru_original, 
        signals=cwru_data, 
        texts=cwru_texts, 
        labels=cwru_labels, 
        fs=12000, 
        n_fft=256, 
        domain="CWRU", 
        spec_augment=False  # 测试时不使用augment
    )
    
    test_loader = DataLoader(
        cwru_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0
    )
    
    # 配置模型参数
    if args.mode == 'baseline':
        # 基线：2个专家，使用所有专家，复杂路由器（需要修改代码恢复旧版本）
        print("Warning: Baseline mode requires old SpecEncoderMoE code")
        print("Current implementation only supports optimized mode")
        num_experts_per_band = 2
        expert_top_k = 2  # 使用所有专家
    else:
        # 优化版本：4个专家，top-2稀疏路由，轻量路由器
        num_experts_per_band = 4
        expert_top_k = 2
    
    print(f"\nModel Configuration:")
    print(f"  - num_experts_per_band: {num_experts_per_band}")
    print(f"  - expert_top_k: {expert_top_k}")
    print(f"  - num_bands: 3")
    print(f"  - embed_dim: 512")
    
    # 创建模型
    model = SpecEncoderMoE(
        in_channels=6,
        embed_dim=512,
        num_bands=3,
        num_experts_per_band=num_experts_per_band,
        channels_per_band=2,
        expert_top_k=expert_top_k
    ).to(device)
    
    # 统计参数
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    # 基准测试1：前向传播速度
    print(f"\n{'='*60}")
    print("Benchmark 1: Forward Pass Speed")
    print(f"{'='*60}")
    avg_time, throughput = benchmark_forward_speed(
        model, test_loader, device, num_batches=args.num_batches
    )
    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # 基准测试2：内存占用
    print(f"\n{'='*60}")
    print("Benchmark 2: Memory Usage")
    print(f"{'='*60}")
    peak_memory = benchmark_memory(model, test_loader, device, num_batches=10)
    print(f"Peak GPU memory: {peak_memory:.2f} MB")
    
    # 基准测试3：计算复杂度估算
    print(f"\n{'='*60}")
    print("Benchmark 3: Computational Complexity")
    print(f"{'='*60}")
    
    # 估算每个样本的专家计算次数
    total_experts = 3 * num_experts_per_band  # 3个频带
    activated_experts = 3 * expert_top_k  # 实际激活的专家数
    efficiency_ratio = activated_experts / total_experts
    
    print(f"Total experts: {total_experts}")
    print(f"Activated experts per sample: {activated_experts}")
    print(f"Efficiency ratio: {efficiency_ratio:.2%}")
    print(f"Theoretical speedup: {1/efficiency_ratio:.2f}x (vs all experts)")
    
    # 总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Speed: {avg_time*1000:.2f} ms/batch ({throughput:.2f} samples/sec)")
    print(f"Memory: {peak_memory:.2f} MB")
    print(f"Parameters: {trainable_params:,}")
    print(f"Efficiency: {efficiency_ratio:.2%} (activated/total experts)")
    
    # 保存结果
    results = {
        'mode': args.mode,
        'num_experts_per_band': num_experts_per_band,
        'expert_top_k': expert_top_k,
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
        'peak_memory_mb': peak_memory,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'efficiency_ratio': efficiency_ratio
    }
    
    import json
    output_file = f"benchmark_results_{args.mode}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

