#!/usr/bin/env python3
"""
绘制 TAR(Overall) 随 Step Limit 变化的曲线图
分成三组：RQ1, Ablation Study, Strategy Comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 定义文件映射
file_patterns = {
    # Group 1: RQ1 - VASSODroid vs Baselines
    'rq1': {
        r'VASSODroid': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        'AutoDroid': 'AutoDroid_26-01-29',
    },
    # Group 2: Ablation Study
    'ablation': {
        r'VASSODroid': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        r'VASSODroid w/o VASSO': 'VASSODroid_No_VASSO_version_26-01-26',
        r'VASSODroid w/o Interaction Validation': 'VASSODroid_No_Interaction_Validation_version_01-31',
    },
    # Group 3: Strategy Comparison
    'strategy': {
        r'Follow and Adapt In Time (V3.2)': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        r'Follow and Adapt In Time (V3)': 'VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3_07-09',
        r'First Follow then Adapt (V3.2)': 'VASSODroid_Full_First_Follow_then_Adapt_version_10-22',
        r'First Follow then Adapt (V3)': 'VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3_06-30',
    },
}

base_dir = Path('dumped_stats_final')
step_limits = [10, 20, 30]

def get_tar_value(filename_pattern, step_limit):
    """从CSV文件中获取TAR(Overall)值"""
    # 查找匹配的文件
    for csv_file in base_dir.glob(f'evaluation_metrics_TestbedEvaluator_{filename_pattern}_limit_{step_limit}_*.csv'):
        df = pd.read_csv(csv_file)
        # 获取 TAR 值
        tar_row = df[(df['metric_type'] == 'task_overall') & (df['metric_name'] == 'TAR')]
        if len(tar_row) > 0:
            return tar_row['value'].values[0] * 100  # 转换为百分比
    return None

# 收集数据
data = {}
for group_name, agents in file_patterns.items():
    data[group_name] = {}
    for agent_label, filename_pattern in agents.items():
        data[group_name][agent_label] = []
        for step_limit in step_limits:
            tar_value = get_tar_value(filename_pattern, step_limit)
            data[group_name][agent_label].append(tar_value)
            print(f"{group_name} - {agent_label} - step {step_limit}: TAR = {tar_value:.2f}%" if tar_value else f"{group_name} - {agent_label} - step {step_limit}: TAR = None")

# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 颜色和标记设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# Group 1: RQ1
ax1 = axes[0]
for i, (agent_label, tar_values) in enumerate(data['rq1'].items()):
    ax1.plot(step_limits, tar_values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax1.set_xlabel('Step Limit', fontsize=12)
ax1.set_ylabel('TAR (%)', fontsize=12)
ax1.set_title('RQ1: VASSODroid vs Baseline', fontsize=14)
ax1.set_xticks(step_limits)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 25)

# Group 2: Ablation Study
ax2 = axes[1]
for i, (agent_label, tar_values) in enumerate(data['ablation'].items()):
    ax2.plot(step_limits, tar_values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax2.set_xlabel('Step Limit', fontsize=12)
ax2.set_ylabel('TAR (%)', fontsize=12)
ax2.set_title('RQ3: Ablation Study', fontsize=14)
ax2.set_xticks(step_limits)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim(0, 25)

# Group 3: Strategy Comparison
ax3 = axes[2]
for i, (agent_label, tar_values) in enumerate(data['strategy'].items()):
    ax3.plot(step_limits, tar_values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax3.set_xlabel('Step Limit', fontsize=12)
ax3.set_ylabel('TAR (%)', fontsize=12)
ax3.set_title('RQ4: Strategy Comparison', fontsize=14)
ax3.set_xticks(step_limits)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_ylim(0, 25)

plt.tight_layout()

# 保存图片
output_path = base_dir / 'tar_overall_by_step_limit.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 同时保存为PDF
output_path_pdf = base_dir / 'tar_overall_by_step_limit.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"PDF已保存到: {output_path_pdf}")

plt.show()

# 打印数据表格
print("\n" + "=" * 80)
print("TAR (Overall) 数据汇总")
print("=" * 80)
for group_name, agents in data.items():
    print(f"\n--- {group_name.upper()} ---")
    print(f"{'Agent':<45} | Step 10 | Step 20 | Step 30")
    print("-" * 80)
    for agent_label, tar_values in agents.items():
        values_str = " | ".join([f"{v:7.2f}" if v else "   N/A " for v in tar_values])
        print(f"{agent_label:<45} | {values_str}")
