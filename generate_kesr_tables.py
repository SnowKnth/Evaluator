#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 KESR (Key Essential State Recognition) 和 EWI (Evidence Widget Identification) 表格
以及 Ablation Study 表格

KESR 计算方法 (Page-level):
- TP = pages_annotated_hit (有标注且命中)
- FP = pages_gen_total - pages_annotated_hit (生成了但没命中)
- FN = pages_annotated_total - pages_annotated_hit (有标注但没命中)
- TN = pages_no_annotation_nor_gen_total (没标注也没生成)

EWI 计算方法:
- Key Essential States Correctly Recognized = pages_annotated_hit (TP)
- Evidence Widget Correctly Identified = recognized_pages_with_widget_assertions_and_evidence_widget_hits
- Conditional Accuracy = Evidence Widget Correctly Identified / Key Essential States Correctly Recognized
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
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return metrics


def parse_filename(filename: str) -> Dict[str, str]:
    """从文件名解析信息"""
    info = {
        'agent': None,
        'step_limit': None,
        'deepseek_version': 'V3.2',
        'filename': filename
    }
    
    basename = os.path.basename(filename)
    
    if 'DeepseekV3' in basename:
        info['deepseek_version'] = 'V3'
    
    if 'limit_10' in basename:
        info['step_limit'] = 10
    elif 'limit_20' in basename:
        info['step_limit'] = 20
    elif 'limit_30' in basename:
        info['step_limit'] = 30
    
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


def calculate_kesr_metrics(metrics: Dict) -> Dict:
    """计算 KESR 指标"""
    # 从 raw_stats 中提取数据
    pages_annotated_hit = metrics.get('raw_stats_overall_pages_annotated_hit', 0)
    pages_gen_total = metrics.get('raw_stats_overall_pages_gen_total', 0)
    pages_annotated_total = metrics.get('raw_stats_overall_pages_annotated_total', 0)
    pages_no_annotation_nor_gen_total = metrics.get('raw_stats_overall_pages_no_annotation_nor_gen_total', 0)
    pages_exec_total = metrics.get('raw_stats_overall_pages_exec_total', 0)
    
    # 计算混淆矩阵
    TP = pages_annotated_hit
    FP = pages_gen_total - pages_annotated_hit
    FN = pages_annotated_total - pages_annotated_hit
    TN = pages_no_annotation_nor_gen_total
    
    # 计算指标
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total * 100 if total > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def calculate_ewi_metrics(metrics: Dict) -> Dict:
    """计算 EWI 指标"""
    pages_annotated_hit = metrics.get('raw_stats_overall_pages_annotated_hit', 0)
    evidence_widget_hits = metrics.get('raw_stats_overall_recognized_pages_with_widget_assertions_and_evidence_widget_hits', 0)
    
    conditional_accuracy = evidence_widget_hits / pages_annotated_hit * 100 if pages_annotated_hit > 0 else 0
    
    return {
        'Key_Essential_States_Recognized': pages_annotated_hit,
        'Evidence_Widget_Identified': evidence_widget_hits,
        'Conditional_Accuracy': conditional_accuracy
    }


def main():
    base_dir = "/data/wxd/LlamaTouch/Evaluator/dumped_stats_final"
    files = glob.glob(os.path.join(base_dir, "evaluation_metrics*.csv"))
    
    print(f"找到 {len(files)} 个 evaluation_metrics 文件\n")
    
    # 存储数据
    data = defaultdict(lambda: defaultdict(dict))
    
    for filepath in files:
        info = parse_filename(filepath)
        if info['agent'] and info['step_limit']:
            metrics = load_metrics_file(filepath)
            data[info['agent']][info['step_limit']] = metrics
    
    # 定义要分析的 agents
    agents_to_analyze = [
        ('VASSODroid_Full_InTime_V3.2', r'\toolNameSmall{}'),
        ('VASSODroid_No_VASSO', r'\toolNameSmall{} w/o VASSO'),
        ('VASSODroid_No_Validation', r'\toolNameSmall{} w/o Interaction Validation'),
        ('AutoDroid', 'AutoDroid'),
    ]
    
    print("="*120)
    print("KESR (Key Essential State Recognition) 统计 - Step Limit = 30")
    print("="*120)
    print(f"\n{'Agent':<50} | {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8} | {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'Precision':>10}")
    print("-"*130)
    
    kesr_results = {}
    ewi_results = {}
    tar_ksar_results = {}
    
    for agent_key, agent_name in agents_to_analyze:
        if agent_key in data and 30 in data[agent_key]:
            metrics = data[agent_key][30]
            kesr = calculate_kesr_metrics(metrics)
            ewi = calculate_ewi_metrics(metrics)
            
            kesr_results[agent_key] = kesr
            ewi_results[agent_key] = ewi
            
            # 获取 TAR 和 KSAR
            tar = metrics.get('task_overall_TAR', 0) * 100
            ksar = metrics.get('task_overall_KSAR', 0) * 100
            completed_tasks = metrics.get('raw_stats_overall_completed_episodes_count', 0)
            completed_subtasks = metrics.get('raw_stats_overall_completed_key_subtasks', 0)
            tar_ksar_results[agent_key] = {
                'TAR': tar,
                'KSAR': ksar,
                'completed_tasks': completed_tasks,
                'completed_subtasks': completed_subtasks
            }
            
            print(f"{agent_name:<50} | {kesr['TP']:>8.0f} {kesr['FP']:>8.0f} {kesr['FN']:>8.0f} {kesr['TN']:>8.0f} | {kesr['Accuracy']:>10.2f} {kesr['F1']:>10.2f} {kesr['Recall']:>10.2f} {kesr['Precision']:>10.2f}")
    
    print("\n" + "="*120)
    print("EWI (Evidence Widget Identification) 统计 - Step Limit = 30")
    print("="*120)
    print(f"\n{'Agent':<50} | {'Key Essential States Recognized':>35} {'Evidence Widget Identified':>30} {'Conditional Accuracy':>20}")
    print("-"*140)
    
    for agent_key, agent_name in agents_to_analyze:
        if agent_key in ewi_results:
            ewi = ewi_results[agent_key]
            print(f"{agent_name:<50} | {ewi['Key_Essential_States_Recognized']:>35.0f} {ewi['Evidence_Widget_Identified']:>30.0f} {ewi['Conditional_Accuracy']:>20.2f}%")
    
    print("\n" + "="*120)
    print("TAR 和 KSAR 统计 - Step Limit = 30")
    print("="*120)
    print(f"\n{'Agent':<50} | {'Accomplished Tasks':>20} {'TAR (%)':>15} {'Accomplished Subtasks':>25} {'KSAR (%)':>15}")
    print("-"*130)
    
    for agent_key, agent_name in agents_to_analyze:
        if agent_key in tar_ksar_results:
            r = tar_ksar_results[agent_key]
            print(f"{agent_name:<50} | {r['completed_tasks']:>20.0f} {r['TAR']:>15.2f} {r['completed_subtasks']:>25.0f} {r['KSAR']:>15.2f}")
    
    # 生成 LaTeX 表格
    print("\n\n" + "="*120)
    print("LaTeX 表格代码")
    print("="*120)
    
    # KESR 表格
    kesr_latex = r"""
\begin{table*}[t]
  \centering
  \caption{Key Essential State Recognition (KESR)}
    \begin{tabular}{l|r|r|r|r!{\vrule width 0.8pt}r|r|r|r}
    \toprule
    \textbf{Configuration} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & \textbf{Accuracy} & \textbf{F1} & \textbf{Recall} & \textbf{Precision}  \\
    \midrule
"""
    for agent_key, agent_name in agents_to_analyze:
        if agent_key in kesr_results:
            k = kesr_results[agent_key]
            kesr_latex += f"    {agent_name} & {k['TP']:.0f} & {k['FP']:.0f} & {k['FN']:.0f} & {k['TN']:.0f} & {k['Accuracy']:.2f} & {k['F1']:.2f} & {k['Recall']:.2f} & {k['Precision']:.2f} \\\\\n"
    kesr_latex += r"""    \bottomrule
    \end{tabular}%
  \label{tab:KESR}%
\end{table*}%
"""
    print("\n--- KESR 表格 ---")
    print(kesr_latex)
    
    # EWI 表格
    ewi_latex = r"""
\begin{table*}[t]
  \centering
  \caption{Evidence Widget Identification (EWI)}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|r|r|r}
    \toprule
    \textbf{Configuration} & \textbf{\makecell{Key Essential States\\Correctly Recognized}} & \textbf{\makecell{Evidence Widget\\Correctly Identified}} & \textbf{Conditional Accuracy (\%)} \\
    \midrule
"""
    for agent_key, agent_name in agents_to_analyze:
        if agent_key in ewi_results:
            e = ewi_results[agent_key]
            ewi_latex += f"    {agent_name} & {e['Key_Essential_States_Recognized']:.0f} & {e['Evidence_Widget_Identified']:.0f} & {e['Conditional_Accuracy']:.2f} \\\\\n"
    ewi_latex += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:EWI}%
\end{table*}%
"""
    print("\n--- EWI 表格 ---")
    print(ewi_latex)
    
    # Ablation Study TAR 表格
    ablation_tar_latex = r"""
\begin{table}[h]
  \centering
  \caption{Ablation Study of \toolNameSmall{} Components for Task and Key Subtask Accomplishment}
  \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|r|r|r|r}
    \toprule
    \textbf{Configuration} & \multicolumn{2}{c|}{\textbf{Task Accomplishment}} & \multicolumn{2}{c}{\textbf{Key Subtask Accomplishment}} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}
     & \textbf{Accomplished Tasks} & \textbf{TAR (\%)} & \textbf{Accomplished Subtasks} & \textbf{KSAR (\%)} \\
    \midrule
"""
    ablation_agents = [
        ('VASSODroid_Full_InTime_V3.2', r'\toolNameSmall{} (Full)'),
        ('VASSODroid_No_VASSO', r'\toolNameSmall{} w/o VASSO'),
        ('VASSODroid_No_Validation', r'\toolNameSmall{} w/o Interaction Validation'),
    ]
    for agent_key, agent_name in ablation_agents:
        if agent_key in tar_ksar_results:
            r = tar_ksar_results[agent_key]
            # 标记最高的 TAR
            tar_str = f"\\textbf{{{r['TAR']:.2f}\\%}}" if agent_key == 'VASSODroid_Full_InTime_V3.2' else f"{r['TAR']:.2f}\\%"
            ksar_str = f"\\textbf{{{r['KSAR']:.2f}\\%}}" if agent_key == 'VASSODroid_Full_InTime_V3.2' else f"{r['KSAR']:.2f}\\%"
            ablation_tar_latex += f"    {agent_name} & {r['completed_tasks']:.0f} & {tar_str} & {r['completed_subtasks']:.0f} & {ksar_str} \\\\\n"
    ablation_tar_latex += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:ablation-study-TAR}%
\end{table}%
"""
    print("\n--- Ablation Study TAR 表格 ---")
    print(ablation_tar_latex)
    
    # Ablation Study KESR 表格
    ablation_kesr_latex = r"""
\begin{table}[h]
  \centering
  \caption{Key Essential State Recognition (KESR) Comparison in Ablation Study}
  \label{tab:ablation-assertion-KESR}
  \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|r|r|r|r!{\vrule width 0.8pt}r|r|r|r}
      \toprule
      \textbf{Configuration} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & \textbf{Accuracy} & \textbf{F1} & \textbf{Recall} & \textbf{Precision} \\
      \midrule
"""
    for agent_key, agent_name in ablation_agents:
        if agent_key in kesr_results:
            k = kesr_results[agent_key]
            ablation_kesr_latex += f"      {agent_name} & {k['TP']:.0f} & {k['FP']:.0f} & {k['FN']:.0f} & {k['TN']:.0f} & {k['Accuracy']:.2f} & {k['F1']:.2f} & {k['Recall']:.2f} & {k['Precision']:.2f} \\\\\n"
    ablation_kesr_latex += r"""      \bottomrule
    \end{tabular}
  }
\end{table}
"""
    print("\n--- Ablation Study KESR 表格 ---")
    print(ablation_kesr_latex)
    
    # Ablation Study EWI 表格
    ablation_ewi_latex = r"""
\begin{table}[h]
  \centering
  \caption{Evidence Widget Identification (EWI) Comparison in Ablation Study}
  \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|r|r|r}
    \toprule
    \textbf{Configuration} & \textbf{\makecell{Key Essential States\\Correctly Recognized}} & \textbf{\makecell{Evidence Widget\\Correctly Identified}} & \textbf{Conditional Accuracy (\%)} \\
    \midrule
"""
    for agent_key, agent_name in ablation_agents:
        if agent_key in ewi_results:
            e = ewi_results[agent_key]
            ablation_ewi_latex += f"    {agent_name} & {e['Key_Essential_States_Recognized']:.0f} & {e['Evidence_Widget_Identified']:.0f} & {e['Conditional_Accuracy']:.2f} \\\\\n"
    ablation_ewi_latex += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:ablation-assertion-EWI}%
\end{table}%
"""
    print("\n--- Ablation Study EWI 表格 ---")
    print(ablation_ewi_latex)
    
    # 计算改进百分比
    print("\n\n" + "="*120)
    print("改进百分比计算")
    print("="*120)
    
    if 'VASSODroid_Full_InTime_V3.2' in tar_ksar_results and 'AutoDroid' in tar_ksar_results:
        full_tar = tar_ksar_results['VASSODroid_Full_InTime_V3.2']['TAR']
        auto_tar = tar_ksar_results['AutoDroid']['TAR']
        full_ksar = tar_ksar_results['VASSODroid_Full_InTime_V3.2']['KSAR']
        auto_ksar = tar_ksar_results['AutoDroid']['KSAR']
        
        tar_improvement = (full_tar - auto_tar) / auto_tar * 100 if auto_tar > 0 else 0
        ksar_improvement = (full_ksar - auto_ksar) / auto_ksar * 100 if auto_ksar > 0 else 0
        
        print(f"\n\\toolNameSmall{{}} vs AutoDroid:")
        print(f"  TAR: {full_tar:.2f}% vs {auto_tar:.2f}% => 改进 {tar_improvement:.2f}%")
        print(f"  KSAR: {full_ksar:.2f}% vs {auto_ksar:.2f}% => 改进 {ksar_improvement:.2f}%")
    
    # Ablation 改进
    if 'VASSODroid_Full_InTime_V3.2' in tar_ksar_results:
        full_tar = tar_ksar_results['VASSODroid_Full_InTime_V3.2']['TAR']
        
        if 'VASSODroid_No_VASSO' in tar_ksar_results:
            no_vasso_tar = tar_ksar_results['VASSODroid_No_VASSO']['TAR']
            vasso_improvement = (full_tar - no_vasso_tar) / no_vasso_tar * 100 if no_vasso_tar > 0 else 0
            print(f"\nVASSO 贡献 (Full vs w/o VASSO):")
            print(f"  TAR: {full_tar:.2f}% vs {no_vasso_tar:.2f}% => VASSO 贡献 {vasso_improvement:.2f}% 相对改进")
        
        if 'VASSODroid_No_Validation' in tar_ksar_results:
            no_val_tar = tar_ksar_results['VASSODroid_No_Validation']['TAR']
            val_improvement = (full_tar - no_val_tar) / no_val_tar * 100 if no_val_tar > 0 else 0
            print(f"\nInteraction Validation 贡献 (Full vs w/o Validation):")
            print(f"  TAR: {full_tar:.2f}% vs {no_val_tar:.2f}% => Validation 贡献 {val_improvement:.2f}% 相对改进")
    
    # 保存所有表格到文件
    output_file = os.path.join(base_dir, "latex_tables_all.tex")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% ===== RQ1: Effectiveness =====\n")
        f.write("% TAR and KSAR table is in latex_table_tar_ksar.tex\n\n")
        f.write("% ===== RQ2: Assertion Quality =====\n")
        f.write(kesr_latex)
        f.write("\n")
        f.write(ewi_latex)
        f.write("\n\n% ===== RQ3: Ablation Study =====\n")
        f.write(ablation_tar_latex)
        f.write("\n")
        f.write(ablation_kesr_latex)
        f.write("\n")
        f.write(ablation_ewi_latex)
    
    print(f"\n\n所有 LaTeX 表格已保存到: {output_file}")


if __name__ == "__main__":
    main()
