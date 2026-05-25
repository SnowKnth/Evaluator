#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 LaTeX 表格，统计 TAR 和 KSAR 指标（Overall, Easy, Medium, Hard）
"""

import os
import csv
import glob
from collections import defaultdict
from typing import Dict, List, Any


def load_metrics_file(filepath: str) -> Dict[str, Any]:
    """读取单个 evaluation_metrics CSV 文件"""
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['metric_type']}_{row['metric_name']}"
                try:
                    metrics[key] = float(row['value'])
                except (ValueError, TypeError):
                    metrics[key] = row['value']
                # 也保存 numerator 和 denominator
                metrics[f"{key}_num"] = row.get('numerator', '')
                metrics[f"{key}_den"] = row.get('denominator', '')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return metrics


def parse_filename(filename: str) -> Dict[str, str]:
    """从文件名解析信息"""
    info = {
        'agent': None,
        'step_limit': None,
        'deepseek_version': 'V3.2',  # 默认为 V3.2 (R1)
        'filename': filename
    }
    
    basename = os.path.basename(filename)
    
    # 检查是否包含 DeepseekV3 标记
    if 'DeepseekV3' in basename:
        info['deepseek_version'] = 'V3'
    
    # 解析 step_limit
    if 'limit_10' in basename:
        info['step_limit'] = 10
    elif 'limit_20' in basename:
        info['step_limit'] = 20
    elif 'limit_30' in basename:
        info['step_limit'] = 30
    
    # 解析 agent 名称
    if 'Full_Follow_and_Adapt_InTime_DeepseekV3' in basename:
        info['agent'] = 'VASSODroid_Full_InTime_V3'
    elif 'Full_Follow_and_Adapt_InTime_Version' in basename:
        info['agent'] = 'VASSODroid_Full_InTime_V3.2'
    elif 'Full_First_Follow_then_Adapt_DeepseekV3' in basename:
        info['agent'] = 'VASSODroid_Full_First_V3'
    elif 'Full_First_Follow_then_Adapt_version' in basename:
        info['agent'] = 'VASSODroid_Full_First_V3.2'
    elif 'No_VASSO_version' in basename:
        info['agent'] = 'VASSODroid_No_VASSO'
    elif 'No_Interaction_Validation' in basename:
        info['agent'] = 'VASSODroid_No_Validation'
    elif 'AutoDroid' in basename:
        info['agent'] = 'AutoDroid'
    
    return info


def main():
    # 查找所有 evaluation_metrics 文件
    base_dir = "/data/wxd/LlamaTouch/Evaluator/dumped_stats_final"
    files = glob.glob(os.path.join(base_dir, "evaluation_metrics*.csv"))
    
    print(f"找到 {len(files)} 个 evaluation_metrics 文件\n")
    
    # 存储数据: data[agent][step_limit] = metrics
    data = defaultdict(lambda: defaultdict(dict))
    
    for filepath in files:
        info = parse_filename(filepath)
        if info['agent'] and info['step_limit']:
            metrics = load_metrics_file(filepath)
            data[info['agent']][info['step_limit']] = metrics
            print(f"加载: {info['agent']} - limit {info['step_limit']}")
    
    # 指标映射
    tar_metrics = ['task_overall_TAR', 'task_easy_TAR', 'task_mid_TAR', 'task_hard_TAR']
    ksar_metrics = ['task_overall_KSAR', 'task_easy_KSAR', 'task_mid_KSAR', 'task_hard_KSAR']
    avg_actions_metrics = ['efficiency_overall_avg_actions_pos', 'efficiency_easy_avg_actions_pos', 'efficiency_mid_avg_actions_pos', 'efficiency_hard_avg_actions_pos']
    
    # 打印统计结果
    print("\n" + "="*120)
    print("TAR 和 KSAR 统计结果 (Step Limit = 30)")
    print("="*120)
    
    # 表头
    print(f"\n{'Agent':<45} | {'TAR Overall':>12} {'TAR Easy':>12} {'TAR Mid':>12} {'TAR Hard':>12} | {'KSAR Overall':>12} {'KSAR Easy':>12} {'KSAR Mid':>12} {'KSAR Hard':>12} | {'AvgAct Overall':>14} {'AvgAct Easy':>12} {'AvgAct Mid':>12} {'AvgAct Hard':>12}")
    print("-"*200)
    
    # 我们的 agents (取 limit=30 的结果)
    our_agents = [
        'VASSODroid_Full_InTime_V3.2',
        'VASSODroid_Full_InTime_V3',
        'VASSODroid_Full_First_V3.2',
        'VASSODroid_Full_First_V3',
        'VASSODroid_No_VASSO',
        'VASSODroid_No_Validation',
        'AutoDroid',
    ]
    
    for agent in our_agents:
        if agent in data and 30 in data[agent]:
            metrics = data[agent][30]
            tar_values = [metrics.get(m, 0) * 100 for m in tar_metrics]
            ksar_values = [metrics.get(m, 0) * 100 for m in ksar_metrics]
            avg_actions_values = [metrics.get(m, 0) for m in avg_actions_metrics]
            
            print(f"{agent:<45} | {tar_values[0]:>12.2f} {tar_values[1]:>12.2f} {tar_values[2]:>12.2f} {tar_values[3]:>12.2f} | {ksar_values[0]:>12.2f} {ksar_values[1]:>12.2f} {ksar_values[2]:>12.2f} {ksar_values[3]:>12.2f} | {avg_actions_values[0]:>14.2f} {avg_actions_values[1]:>12.2f} {avg_actions_values[2]:>12.2f} {avg_actions_values[3]:>12.2f}")
    
    # 外部 baseline（从你提供的数据）
    print("-"*200)
    external_baselines = {
        'Auto-UI': {'TAR': [4.44, 4.95, 4.19, 4.82], 'KSAR': ['-', '-', '-', '-'], 'AvgAct': ['-', '-', '-', '-']},
        'AppAgent': {'TAR': [10.91, 16.83, 10.97, 3.61], 'KSAR': ['-', '-', '-', '-'], 'AvgAct': ['-', '-', '-', '-']},
        'CoCo-Agent': {'TAR': [4.47, 7.92, 3.91, 2.41], 'KSAR': ['-', '-', '-', '-'], 'AvgAct': ['-', '-', '-', '-']},
    }
    
    for agent, values in external_baselines.items():
        tar_str = ' '.join([f"{v:>12.2f}" if isinstance(v, (int, float)) else f"{v:>12}" for v in values['TAR']])
        ksar_str = ' '.join([f"{v:>12}" for v in values['KSAR']])
        avgact_str = ' '.join([f"{v:>12}" for v in values['AvgAct']])
        print(f"{agent:<45} | {tar_str} | {ksar_str} | {avgact_str}")
    
    # 生成 LaTeX 表格
    print("\n\n" + "="*120)
    print("LaTeX 表格代码")
    print("="*120)
    
    latex_code = r"""
\begin{table}[t]
  \centering
  \caption{Task and Key Subtask Accomplishment Assessment}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|rrrr|rrrr}
    \toprule
    \textbf{Approach} & \multicolumn{4}{c|}{\textbf{TAR (\%)}} & \multicolumn{4}{c}{\textbf{KSAR (\%)}} \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}
     & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\
    \midrule
"""
    
    # 添加我们的 agents
    agent_display_names = {
        'VASSODroid_Full_InTime_V3.2': r'\toolNameSmall{} (V3.2)',
        'VASSODroid_Full_InTime_V3': r'\toolNameSmall{} (V3)',
        'VASSODroid_Full_First_V3.2': r'\toolNameSmall{} First-Follow (V3.2)',
        'VASSODroid_Full_First_V3': r'\toolNameSmall{} First-Follow (V3)',
        'VASSODroid_No_VASSO': r'\toolNameSmall{} w/o VASSO',
        'VASSODroid_No_Validation': r'\toolNameSmall{} w/o Validation',
        'AutoDroid': 'AutoDroid',
    }
    
    for agent in our_agents:
        if agent in data and 30 in data[agent]:
            metrics = data[agent][30]
            tar_values = [metrics.get(m, 0) * 100 for m in tar_metrics]
            ksar_values = [metrics.get(m, 0) * 100 for m in ksar_metrics]
            avg_actions_values = [metrics.get(m, 0) for m in avg_actions_metrics]
            
            display_name = agent_display_names.get(agent, agent)
            
            # 找最大值加粗（这里简单处理，实际可能需要更复杂的逻辑）
            tar_str = ' & '.join([f"{v:.2f}" for v in tar_values])
            ksar_str = ' & '.join([f"{v:.2f}" for v in ksar_values])
            
            latex_code += f"    {display_name} & {tar_str} & {ksar_str} \\\\\n"
    
    # 添加外部 baselines
    latex_code += r"    \midrule" + "\n"
    for agent, values in external_baselines.items():
        tar_str = ' & '.join([f"{v:.2f}" if isinstance(v, (int, float)) else '-' for v in values['TAR']])
        ksar_str = ' & '.join(['-' for _ in values['KSAR']])
        latex_code += f"    {agent} & {tar_str} & {ksar_str} \\\\\n"
    
    latex_code += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:effectiveness-results}%
\end{table}%
"""
    
    print(latex_code)
    
    # 保存到文件
    output_file = os.path.join(base_dir, "latex_table_tar_ksar.tex")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"\nLaTeX 表格已保存到: {output_file}")
    
    # 生成 Avg. Actions 单独的表格
    print("\n\n" + "="*120)
    print("Avg. Actions LaTeX 表格代码")
    print("="*120)
    
    latex_code_avg = r"""
\begin{table}[t]
  \centering
  \caption{Average Actions Taken to Accomplish Key Subtasks}
  \resizebox{0.7\columnwidth}{!}{
    \begin{tabular}{l|rrrr}
    \toprule
    \textbf{Approach} & \multicolumn{4}{c}{\textbf{Average Actions}} \\
    \cmidrule(lr){2-5}
     & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\
    \midrule
"""
    
    for agent in our_agents:
        if agent in data and 30 in data[agent]:
            metrics = data[agent][30]
            avg_actions_values = [metrics.get(m, 0) for m in avg_actions_metrics]
            
            display_name = agent_display_names.get(agent, agent)
            avg_actions_str = ' & '.join([f"{v:.2f}" for v in avg_actions_values])
            
            latex_code_avg += f"    {display_name} & {avg_actions_str} \\\\\n"
    
    latex_code_avg += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:avg-actions}%
\end{table}%
"""
    
    print(latex_code_avg)
    
    # 保存 Avg. Actions 表格到文件
    output_file_avg = os.path.join(base_dir, "latex_table_avg_actions.tex")
    with open(output_file_avg, 'w', encoding='utf-8') as f:
        f.write(latex_code_avg)
    print(f"\nAvg. Actions LaTeX 表格已保存到: {output_file_avg}")
    
    # 额外输出：不同 step_limit 的对比
    print("\n\n" + "="*120)
    print("不同 Step Limit 的 TAR 对比")
    print("="*120)
    
    for agent in our_agents:
        if agent in data:
            print(f"\n{agent}:")
            for limit in [10, 20, 30]:
                if limit in data[agent]:
                    metrics = data[agent][limit]
                    tar_overall = metrics.get('task_overall_TAR', 0) * 100
                    ksar_overall = metrics.get('task_overall_KSAR', 0) * 100
                    print(f"  Limit {limit}: TAR={tar_overall:.2f}%, KSAR={ksar_overall:.2f}%")


if __name__ == "__main__":
    main()
