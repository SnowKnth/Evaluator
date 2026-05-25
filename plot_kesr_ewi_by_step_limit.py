#!/usr/bin/env python3
"""
绘制 KESR(Overall) 和 AWI(Overall) 随 Step Limit 变化的曲线图
分成三组：RQ1, RQ3 Ablation Study, RQ4 Strategy Comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 定义文件映射
file_patterns = {
    # Group 1: RQ1 - VASSODroid vs Baseline
    'rq1': {
        'VASSODroid': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        'AutoDroid': 'AutoDroid_26-01-29',
    },
    # Group 2: RQ3 Ablation Study
    'ablation': {
        'VASSODroid': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        'VASSODroid w/o VASSO': 'VASSODroid_No_VASSO_version_26-01-26',
        'VASSODroid w/o Interaction Validation': 'VASSODroid_No_Interaction_Validation_version_01-31',
    },
    # Group 3: RQ4 Strategy Comparison
    'strategy': {
        'Follow and Adapt In Time (V3.2)': 'VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02',
        'Follow and Adapt In Time (V3)': 'VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3_07-09',
        'First Follow then Adapt (V3.2)': 'VASSODroid_Full_First_Follow_then_Adapt_version_10-22',
        'First Follow then Adapt (V3)': 'VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3_06-30',
    },
}

base_dir = Path('dumped_stats_final')
step_limits = [10, 20, 30]

def get_raw_stat(df, difficulty, stat_name):
    """从CSV中获取原始统计值"""
    metric_type = f'raw_stats_{difficulty}'
    row = df[(df['metric_type'] == metric_type) & (df['metric_name'] == stat_name)]
    if len(row) > 0:
        val = row['value'].values[0]
        return int(val) if pd.notna(val) else 0
    return 0

def get_kesr_f1(filename_pattern, step_limit):
    """从CSV文件中获取KESR F1值"""
    for csv_file in base_dir.glob(f'evaluation_metrics_TestbedEvaluator_{filename_pattern}_limit_{step_limit}_*.csv'):
        df = pd.read_csv(csv_file)
        
        # 计算 KESR 指标
        tp = get_raw_stat(df, 'overall', 'pages_annotated_hit')
        pages_gen = get_raw_stat(df, 'overall', 'pages_gen_total')
        pages_annotated = get_raw_stat(df, 'overall', 'pages_annotated_total')
        
        fp = pages_gen - tp
        fn = pages_annotated - tp
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    return None

def get_awi_accuracy(filename_pattern, step_limit):
    """从CSV文件中获取AWI Conditional Accuracy值"""
    for csv_file in base_dir.glob(f'evaluation_metrics_TestbedEvaluator_{filename_pattern}_limit_{step_limit}_*.csv'):
        df = pd.read_csv(csv_file)
        
        # 计算 AWI 指标
        recognized = get_raw_stat(df, 'overall', 'recognized_pages_with_widget_assertions')
        identified = get_raw_stat(df, 'overall', 'recognized_pages_with_widget_assertions_and_evidence_widget_hits')
        
        cond_acc = identified / recognized * 100 if recognized > 0 else 0
        return cond_acc
    return None

# 收集数据
kesr_data = {}
awi_data = {}

for group_name, agents in file_patterns.items():
    kesr_data[group_name] = {}
    awi_data[group_name] = {}
    
    for agent_label, filename_pattern in agents.items():
        kesr_data[group_name][agent_label] = []
        awi_data[group_name][agent_label] = []
        
        for step_limit in step_limits:
            kesr_value = get_kesr_f1(filename_pattern, step_limit)
            awi_value = get_awi_accuracy(filename_pattern, step_limit)
            
            kesr_data[group_name][agent_label].append(kesr_value)
            awi_data[group_name][agent_label].append(awi_value)
            
            print(f"{group_name} - {agent_label} - step {step_limit}: KESR F1 = {kesr_value:.2f}%, AWI = {awi_value:.2f}%" if kesr_value and awi_value else f"{group_name} - {agent_label} - step {step_limit}: Data missing")

# 颜色和标记设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# ============== KESR F1 图 ==============
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

# Group 1: RQ1
ax1 = axes1[0]
for i, (agent_label, values) in enumerate(kesr_data['rq1'].items()):
    ax1.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax1.set_xlabel('Step Limit', fontsize=12)
ax1.set_ylabel('ESR F1 (%)', fontsize=12)
ax1.set_title('RQ2: VASSODroid vs Baseline', fontsize=14)
ax1.set_xticks(step_limits)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 60)

# Group 2: RQ3 Ablation Study
ax2 = axes1[1]
for i, (agent_label, values) in enumerate(kesr_data['ablation'].items()):
    ax2.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax2.set_xlabel('Step Limit', fontsize=12)
ax2.set_ylabel('ESR F1 (%)', fontsize=12)
ax2.set_title('RQ3: Ablation Study', fontsize=14)
ax2.set_xticks(step_limits)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim(0, 60)

# Group 3: RQ4 Strategy Comparison
ax3 = axes1[2]
for i, (agent_label, values) in enumerate(kesr_data['strategy'].items()):
    ax3.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax3.set_xlabel('Step Limit', fontsize=12)
ax3.set_ylabel('ESR F1 (%)', fontsize=12)
ax3.set_title('RQ4: Strategy Comparison', fontsize=14)
ax3.set_xticks(step_limits)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_ylim(0, 60)

plt.tight_layout()

# 保存 KESR 图片
output_path_kesr = base_dir / 'kesr_f1_overall_by_step_limit.png'
plt.savefig(output_path_kesr, dpi=300, bbox_inches='tight')
print(f"\nESR图片已保存到: {output_path_kesr}")

output_path_kesr_pdf = base_dir / 'kesr_f1_overall_by_step_limit.pdf'
plt.savefig(output_path_kesr_pdf, bbox_inches='tight')
print(f"ESR PDF已保存到: {output_path_kesr_pdf}")

# ============== AWI Conditional Accuracy 图 ==============
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Group 1: RQ1
ax1 = axes2[0]
for i, (agent_label, values) in enumerate(awi_data['rq1'].items()):
    ax1.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax1.set_xlabel('Step Limit', fontsize=12)
ax1.set_ylabel('AWI Conditional Accuracy (%)', fontsize=12)
ax1.set_title('RQ2: VASSODroid vs Baseline', fontsize=14)
ax1.set_xticks(step_limits)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 100)

# Group 2: RQ3 Ablation Study
ax2 = axes2[1]
for i, (agent_label, values) in enumerate(awi_data['ablation'].items()):
    ax2.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax2.set_xlabel('Step Limit', fontsize=12)
ax2.set_ylabel('AWI Conditional Accuracy (%)', fontsize=12)
ax2.set_title('RQ3: Ablation Study', fontsize=14)
ax2.set_xticks(step_limits)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim(0, 100)

# Group 3: RQ4 Strategy Comparison
ax3 = axes2[2]
for i, (agent_label, values) in enumerate(awi_data['strategy'].items()):
    ax3.plot(step_limits, values, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=agent_label)
ax3.set_xlabel('Step Limit', fontsize=12)
ax3.set_ylabel('AWI Conditional Accuracy (%)', fontsize=12)
ax3.set_title('RQ4: Strategy Comparison', fontsize=14)
ax3.set_xticks(step_limits)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_ylim(0, 100)

plt.tight_layout()

# 保存 AWI 图片
output_path_awi = base_dir / 'awi_overall_by_step_limit.png'
plt.savefig(output_path_awi, dpi=300, bbox_inches='tight')
print(f"\nAWI图片已保存到: {output_path_awi}")

output_path_awi_pdf = base_dir / 'awi_overall_by_step_limit.pdf'
plt.savefig(output_path_awi_pdf, bbox_inches='tight')
print(f"AWI PDF已保存到: {output_path_awi_pdf}")

plt.show()

# 打印数据表格
print("\n" + "=" * 100)
print("ESR F1 (Overall) 数据汇总")
print("=" * 100)
for group_name, agents in kesr_data.items():
    print(f"\n--- {group_name.upper()} ---")
    print(f"{'Agent':<45} | Step 10 | Step 20 | Step 30")
    print("-" * 100)
    for agent_label, values in agents.items():
        values_str = " | ".join([f"{v:7.2f}" if v else "   N/A " for v in values])
        print(f"{agent_label:<45} | {values_str}")

print("\n" + "=" * 100)
print("AWI Conditional Accuracy (Overall) 数据汇总")
print("=" * 100)
for group_name, agents in awi_data.items():
    print(f"\n--- {group_name.upper()} ---")
    print(f"{'Agent':<45} | Step 10 | Step 20 | Step 30")
    print("-" * 100)
    for agent_label, values in agents.items():
        values_str = " | ".join([f"{v:7.2f}" if v else "   N/A " for v in values])
        print(f"{agent_label:<45} | {values_str}")
