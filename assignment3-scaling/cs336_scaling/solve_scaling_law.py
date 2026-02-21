import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional

# ==========================================
# 1. 核心计算函数
# ==========================================

def fit_power_law(computes: np.ndarray, params: np.ndarray) -> Tuple[float, float]:
    """
    在对数空间拟合幂律关系: N = k * C^a
    log(N) = a * log(C) + log(k)
    返回: (a, k)
    """
    log_c = np.log10(computes)
    log_n = np.log10(params)

    # 使用一次多项式拟合 (线性回归)
    # slope = a, intercept = log10(k)
    slope, intercept = np.polyfit(log_c, log_n, 1)
    
    a = slope
    k = 10 ** intercept
    return a, k

def estimate_non_embedding_params(num_layers: int, d_model: int) -> int:
    """
    根据作业要求估算非嵌入参数量: 12 * L * d_model^2
    """
    return 12 * num_layers * (d_model ** 2)

def find_optimal_hyperparameters(
    target_params: float, 
    d_model_candidates: List[int] = [768, 1024, 1280, 1600, 2048, 2560]
) -> Tuple[int, int, int]:
    """
    根据目标参数量，搜索最接近的 (num_layers, d_model) 组合。
    返回: (best_layers, best_d_model, best_est_params)
    """
    best_config = None
    min_diff = float('inf')

    print(f"\n[配置搜索] 目标参数量: {target_params/1e6:.2f}M")
    
    for h in d_model_candidates:
        # 根据 N ≈ 12 * L * h^2 反推 L
        # L = N / (12 * h^2)
        est_l = target_params / (12 * (h**2))
        
        # 尝试向上和向下取偶数层 (Transformer 层数通常为偶数)
        candidates_l = [int(est_l), int(est_l) + 1, int(est_l) + 2]
        candidates_l = [l for l in candidates_l if l % 2 == 0 and l >= 2]
        
        if not candidates_l:
            # 如果没有偶数，取最近的整数
            candidates_l = [max(2, round(est_l))]

        for l in candidates_l:
            est_p = estimate_non_embedding_params(l, h)
            diff = abs(est_p - target_params)
            
            # 打印一些接近的候选项供参考
            if diff / target_params < 0.1: # 误差小于10%才显示
                print(f"  - 候选: d_model={h}, layers={l} -> 估算参数: {est_p/1e6:.2f}M (误差: {diff/1e6:.2f}M)")

            if diff < min_diff:
                min_diff = diff
                best_config = (l, h, est_p)

    return best_config

# ==========================================
# 2. 主流程
# ==========================================

def main():
    # 数据文件路径
    json_path = 'data/isoflops_curves.json'
    
    print(f"正在读取数据: {json_path} ...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_path}")
        print("请确保你已下载作业数据或将路径修改为正确位置。")
        return

    # --- 步骤 1: 提取每个计算预算下的最优模型 ---
    budget_groups = {}
    for run in data:
        budget = run['compute_budget']
        loss = run['final_loss']
        params = run['parameters']
        
        if budget not in budget_groups:
            budget_groups[budget] = {'min_loss': float('inf'), 'opt_params': None}
        
        # 找到该 budget 下 loss 最小的 run
        if loss < budget_groups[budget]['min_loss']:
            budget_groups[budget]['min_loss'] = loss
            budget_groups[budget]['opt_params'] = params

    # 准备拟合数据
    budgets = sorted(budget_groups.keys())
    opt_params = [budget_groups[b]['opt_params'] for b in budgets]
    
    X = np.array(budgets)
    Y = np.array(opt_params)

    print("\n[数据提取] IsoFLOPs 最优配置:")
    for b, p in zip(X, Y):
        print(f"  Budget: {b:.1e} FLOPs -> Params: {p/1e6:.2f}M")

    # --- 步骤 2: 拟合 Scaling Law ---
    a, k = fit_power_law(X, Y)
    print("\n[拟合结果]")
    print(f"  Scaling Law 公式: N_opt = {k:.4e} * C ^ {a:.4f}")
    print(f"  幂指数 (a): {a:.4f} (理论值通常在 0.45-0.50 之间)")

    # --- 步骤 3: 预测 1e19 FLOPs ---
    target_flops = 1e19
    predicted_params = k * (target_flops ** a)
    print(f"\n[预测结果]")
    print(f"  目标预算: {target_flops:.1e} FLOPs")
    print(f"  预测最优参数量 (N_opt): {predicted_params/1e9:.4f} B (十亿)")

    # --- 步骤 4: 确定超参数 ---
    # 作业提示: 寻找 d_model 和 num_layers
    best_layers, best_d_model, best_est_p = find_optimal_hyperparameters(predicted_params)
    
    print(f"\n[最终推荐配置]")
    print(f"  d_model:    {best_d_model}")
    print(f"  num_layers: {best_layers}")
    print(f"  num_heads:  {best_d_model // 64} (假设 head_dim=64)")
    print(f"  估算参数量: {best_est_p/1e9:.4f} B")
    print(f"  参数误差:   {abs(best_est_p - predicted_params)/1e6:.2f} M")

    # --- 步骤 5: 绘图 (可选) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='red', label='Experimental Data (IsoFLOPs minima)')
    
    # 绘制拟合线 (延伸到 1e19)
    # x_plot = np.logspace(np.log10(min(X)), np.log10(target_flops), 100)
    # 终点取 '实验数据的最大值' 和 '目标预算' 中的较大者
    x_plot = np.logspace(np.log10(min(X)), np.log10(max(X.max(), target_flops)), 100)
    y_plot = k * (x_plot ** a)
    plt.plot(x_plot, y_plot, color='blue', linestyle='--', label=f'Fit: $N={k:.2e} C^{{{a:.2f}}}$')
    
    # 标记预测点
    plt.scatter([target_flops], [predicted_params], color='green', marker='*', s=200, label='Prediction (1e19)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Optimal Parameters (N)')
    plt.title('Scaling Law: Compute vs Optimal Model Size')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    output_img = 'scaling_law_plot.png'
    plt.savefig(output_img)
    print(f"\n图表已保存至: {output_img}")

if __name__ == "__main__":
    main()