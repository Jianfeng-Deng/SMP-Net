"""
对比基准测试结果

使用方法：
python compare_benchmark.py
"""

import json
import os


def load_results(mode):
    """加载基准测试结果"""
    filename = f"benchmark_results_{mode}.json"
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def compare_results(baseline, optimized):
    """对比两个结果"""
    print("=" * 80)
    print("MoE Optimization Benchmark Comparison")
    print("=" * 80)
    
    print("\n{:<30} {:<20} {:<20} {:<15}".format(
        "Metric", "Baseline", "Optimized", "Improvement"
    ))
    print("-" * 80)
    
    # 专家配置
    print("{:<30} {:<20} {:<20} {:<15}".format(
        "Experts per band",
        baseline['num_experts_per_band'],
        optimized['num_experts_per_band'],
        f"+{optimized['num_experts_per_band'] - baseline['num_experts_per_band']}"
    ))
    
    print("{:<30} {:<20} {:<20} {:<15}".format(
        "Activated experts (top-k)",
        baseline['expert_top_k'],
        optimized['expert_top_k'],
        "-"
    ))
    
    # 速度
    speed_improvement = (baseline['avg_time_ms'] - optimized['avg_time_ms']) / baseline['avg_time_ms'] * 100
    print("{:<30} {:<20.2f} {:<20.2f} {:<15}".format(
        "Avg time (ms/batch)",
        baseline['avg_time_ms'],
        optimized['avg_time_ms'],
        f"{speed_improvement:+.1f}%"
    ))
    
    throughput_improvement = (optimized['throughput'] - baseline['throughput']) / baseline['throughput'] * 100
    print("{:<30} {:<20.2f} {:<20.2f} {:<15}".format(
        "Throughput (samples/sec)",
        baseline['throughput'],
        optimized['throughput'],
        f"{throughput_improvement:+.1f}%"
    ))
    
    # 内存
    memory_improvement = (baseline['peak_memory_mb'] - optimized['peak_memory_mb']) / baseline['peak_memory_mb'] * 100
    print("{:<30} {:<20.2f} {:<20.2f} {:<15}".format(
        "Peak memory (MB)",
        baseline['peak_memory_mb'],
        optimized['peak_memory_mb'],
        f"{memory_improvement:+.1f}%"
    ))
    
    # 参数数量
    params_change = (optimized['trainable_params'] - baseline['trainable_params']) / baseline['trainable_params'] * 100
    print("{:<30} {:<20,} {:<20,} {:<15}".format(
        "Trainable parameters",
        baseline['trainable_params'],
        optimized['trainable_params'],
        f"{params_change:+.1f}%"
    ))
    
    # 效率
    print("{:<30} {:<20.1%} {:<20.1%} {:<15}".format(
        "Efficiency ratio",
        baseline['efficiency_ratio'],
        optimized['efficiency_ratio'],
        f"{(optimized['efficiency_ratio'] - baseline['efficiency_ratio']) * 100:+.1f}pp"
    ))
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  - Speed improvement: {speed_improvement:+.1f}%")
    print(f"  - Memory reduction: {memory_improvement:+.1f}%")
    print(f"  - Model capacity: {params_change:+.1f}% (more experts)")
    print(f"  - Efficiency: {(optimized['efficiency_ratio'] - baseline['efficiency_ratio']) * 100:+.1f} percentage points")
    print("=" * 80)


def main():
    baseline = load_results('baseline')
    optimized = load_results('optimized')
    
    if baseline is None and optimized is None:
        print("Error: No benchmark results found.")
        print("Please run benchmark_moe.py first:")
        print("  python benchmark_moe.py --mode baseline")
        print("  python benchmark_moe.py --mode optimized")
        return
    
    if baseline is None:
        print("Warning: Baseline results not found.")
        print("Showing optimized results only:\n")
        print(json.dumps(optimized, indent=2))
        return
    
    if optimized is None:
        print("Warning: Optimized results not found.")
        print("Showing baseline results only:\n")
        print(json.dumps(baseline, indent=2))
        return
    
    compare_results(baseline, optimized)


if __name__ == '__main__':
    main()

