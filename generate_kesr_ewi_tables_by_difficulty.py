#!/usr/bin/env python3
"""
生成包含各难度级别的 KESR 和 EWI LaTeX 表格
"""

import pandas as pd
from pathlib import Path

# 定义文件映射 (Step Limit = 30)
file_mapping = {
    r'\toolNameSmall{} (V3.2)': 'evaluation_metrics_TestbedEvaluator_VASSODroid_Full_Follow_and_Adapt_InTime_Version_26-02-02_limit_30_2026-02-09-19:02:28.csv',
    r'\toolNameSmall{} (V3)': 'evaluation_metrics_TestbedEvaluator_VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3_07-09_limit_30_2026-02-09-19:06:42.csv',
    r'\toolNameSmall{} First-Follow (V3.2)': 'evaluation_metrics_TestbedEvaluator_VASSODroid_Full_First_Follow_then_Adapt_version_10-22_limit_30_2026-02-09-19:04:02.csv',
    r'\toolNameSmall{} First-Follow (V3)': 'evaluation_metrics_TestbedEvaluator_VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3_06-30_limit_30_2026-02-09-19:07:40.csv',
    r'\toolNameSmall{} w/o VASSO': 'evaluation_metrics_TestbedEvaluator_VASSODroid_No_VASSO_version_26-01-26_limit_30_2026-02-09-19:03:10.csv',
    r'\toolNameSmall{} w/o Validation': 'evaluation_metrics_TestbedEvaluator_VASSODroid_No_Interaction_Validation_version_01-31_limit_30_2026-02-09-19:04:46.csv',
    'AutoDroid': 'evaluation_metrics_TestbedEvaluator_AutoDroid_26-01-29_limit_30_2026-02-09-19:05:34.csv',
}

base_dir = Path('dumped_stats_final')

def get_metric_value(df, metric_type, metric_name):
    """从CSV中获取指定指标的值"""
    row = df[(df['metric_type'] == metric_type) & (df['metric_name'] == metric_name)]
    if len(row) > 0:
        return row['value'].values[0]
    return None

def get_raw_stat(df, difficulty, stat_name):
    """从CSV中获取原始统计值"""
    metric_type = f'raw_stats_{difficulty}'
    row = df[(df['metric_type'] == metric_type) & (df['metric_name'] == stat_name)]
    if len(row) > 0:
        val = row['value'].values[0]
        return int(val) if pd.notna(val) else 0
    return 0

def calculate_kesr_metrics(df, difficulty):
    """计算KESR指标"""
    tp = get_raw_stat(df, difficulty, 'pages_annotated_hit')
    pages_gen = get_raw_stat(df, difficulty, 'pages_gen_total')
    pages_annotated = get_raw_stat(df, difficulty, 'pages_annotated_total')
    tn = get_raw_stat(df, difficulty, 'pages_no_annotation_nor_gen_total')
    
    fp = pages_gen - tp
    fn = pages_annotated - tp
    
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }

def calculate_ewi_metrics(df, difficulty):
    """计算EWI指标"""
    recognized = get_raw_stat(df, difficulty, 'recognized_pages_with_widget_assertions')
    identified = get_raw_stat(df, difficulty, 'recognized_pages_with_widget_assertions_and_evidence_widget_hits')
    cond_acc = identified / recognized * 100 if recognized > 0 else 0
    
    return {
        'recognized': recognized,
        'identified': identified,
        'cond_acc': cond_acc
    }

# 收集所有数据
results = {}
for agent_name, filename in file_mapping.items():
    csv_path = base_dir / filename
    if not csv_path.exists():
        print(f'File not found: {csv_path}')
        continue
    
    df = pd.read_csv(csv_path)
    
    kesr_data = {}
    ewi_data = {}
    for difficulty in ['overall', 'easy', 'mid', 'hard']:
        kesr_data[difficulty] = calculate_kesr_metrics(df, difficulty)
        ewi_data[difficulty] = calculate_ewi_metrics(df, difficulty)
    
    results[agent_name] = {'kesr': kesr_data, 'ewi': ewi_data}

# 打印数据验证
print("=" * 120)
print("KESR 数据验证 (Step Limit = 30)")
print("=" * 120)
print(f"{'Agent':<45} | {'Overall':>10} | {'Easy':>10} | {'Medium':>10} | {'Hard':>10}")
print("-" * 120)
for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['kesr']
    print(f"{agent_name:<45} | Acc: {data['overall']['accuracy']:>6.2f} | {data['easy']['accuracy']:>6.2f} | {data['mid']['accuracy']:>6.2f} | {data['hard']['accuracy']:>6.2f}")
    print(f"{'':45} |  F1: {data['overall']['f1']:>6.2f} | {data['easy']['f1']:>6.2f} | {data['mid']['f1']:>6.2f} | {data['hard']['f1']:>6.2f}")
    print(f"{'':45} | Rec: {data['overall']['recall']:>6.2f} | {data['easy']['recall']:>6.2f} | {data['mid']['recall']:>6.2f} | {data['hard']['recall']:>6.2f}")
    print(f"{'':45} | Pre: {data['overall']['precision']:>6.2f} | {data['easy']['precision']:>6.2f} | {data['mid']['precision']:>6.2f} | {data['hard']['precision']:>6.2f}")
    print()

print("\n" + "=" * 120)
print("EWI 数据验证 (Step Limit = 30)")
print("=" * 120)
print(f"{'Agent':<45} | {'Overall':>15} | {'Easy':>15} | {'Medium':>15} | {'Hard':>15}")
print("-" * 120)
for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['ewi']
    print(f"{agent_name:<45} | {data['overall']['identified']}/{data['overall']['recognized']} ({data['overall']['cond_acc']:.1f}%) | {data['easy']['identified']}/{data['easy']['recognized']} ({data['easy']['cond_acc']:.1f}%) | {data['mid']['identified']}/{data['mid']['recognized']} ({data['mid']['cond_acc']:.1f}%) | {data['hard']['identified']}/{data['hard']['recognized']} ({data['hard']['cond_acc']:.1f}%)")

# 生成 LaTeX 表格
print("\n" + "=" * 120)
print("LaTeX 表格")
print("=" * 120)

# KESR Accuracy 和 F1 表格
kesr_table = r"""\begin{table}[t]
  \centering
  \caption{Key Essential State Recognition (KESR) Assessment}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|rrrr|rrrr}
    \toprule
    \textbf{Approach} & \multicolumn{4}{c|}{\textbf{KESR Accuracy (\%)}} & \multicolumn{4}{c}{\textbf{KESR F1 (\%)}} \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}
     & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\
    \midrule
"""

for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['kesr']
    kesr_table += f"    {agent_name} & {data['overall']['accuracy']:.2f} & {data['easy']['accuracy']:.2f} & {data['mid']['accuracy']:.2f} & {data['hard']['accuracy']:.2f} & {data['overall']['f1']:.2f} & {data['easy']['f1']:.2f} & {data['mid']['f1']:.2f} & {data['hard']['f1']:.2f} \\\\\n"

kesr_table += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:kesr-assessment}%
\end{table}%"""

print("\n--- KESR Accuracy & F1 表格 ---\n")
print(kesr_table)

# KESR Recall 和 Precision 表格
kesr_recall_prec_table = r"""\begin{table}[t]
  \centering
  \caption{Key Essential State Recognition (KESR) - Recall and Precision}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|rrrr|rrrr}
    \toprule
    \textbf{Approach} & \multicolumn{4}{c|}{\textbf{KESR Recall (\%)}} & \multicolumn{4}{c}{\textbf{KESR Precision (\%)}} \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}
     & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\
    \midrule
"""

for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['kesr']
    kesr_recall_prec_table += f"    {agent_name} & {data['overall']['recall']:.2f} & {data['easy']['recall']:.2f} & {data['mid']['recall']:.2f} & {data['hard']['recall']:.2f} & {data['overall']['precision']:.2f} & {data['easy']['precision']:.2f} & {data['mid']['precision']:.2f} & {data['hard']['precision']:.2f} \\\\\n"

kesr_recall_prec_table += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:kesr-recall-precision}%
\end{table}%"""

print("\n--- KESR Recall & Precision 表格 ---\n")
print(kesr_recall_prec_table)

# EWI 表格
ewi_table = r"""\begin{table}[t]
  \centering
  \caption{Evidence Widget Identification (EWI) Assessment}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|rrrr}
    \toprule
    \textbf{Approach} & \multicolumn{4}{c}{\textbf{EWI Conditional Accuracy (\%)}} \\
    \cmidrule(lr){2-5}
     & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\
    \midrule
"""

for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['ewi']
    ewi_table += f"    {agent_name} & {data['overall']['cond_acc']:.2f} & {data['easy']['cond_acc']:.2f} & {data['mid']['cond_acc']:.2f} & {data['hard']['cond_acc']:.2f} \\\\\n"

ewi_table += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:ewi-assessment}%
\end{table}%"""

print("\n--- EWI 表格 ---\n")
print(ewi_table)

# EWI 详细表格 (包含 recognized 和 identified)
ewi_detailed_table = r"""\begin{table}[t]
  \centering
  \caption{Evidence Widget Identification (EWI) - Detailed Statistics}
  \resizebox{\textwidth}{!}{
    \begin{tabular}{l|rrr|rrr|rrr|rrr}
    \toprule
    \textbf{Approach} & \multicolumn{3}{c|}{\textbf{Overall}} & \multicolumn{3}{c|}{\textbf{Easy}} & \multicolumn{3}{c|}{\textbf{Medium}} & \multicolumn{3}{c}{\textbf{Hard}} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}
     & \textbf{Rec.} & \textbf{Id.} & \textbf{Acc.} & \textbf{Rec.} & \textbf{Id.} & \textbf{Acc.} & \textbf{Rec.} & \textbf{Id.} & \textbf{Acc.} & \textbf{Rec.} & \textbf{Id.} & \textbf{Acc.} \\
    \midrule
"""

for agent_name in file_mapping.keys():
    if agent_name not in results:
        continue
    data = results[agent_name]['ewi']
    ewi_detailed_table += f"    {agent_name} & {data['overall']['recognized']} & {data['overall']['identified']} & {data['overall']['cond_acc']:.1f} & {data['easy']['recognized']} & {data['easy']['identified']} & {data['easy']['cond_acc']:.1f} & {data['mid']['recognized']} & {data['mid']['identified']} & {data['mid']['cond_acc']:.1f} & {data['hard']['recognized']} & {data['hard']['identified']} & {data['hard']['cond_acc']:.1f} \\\\\n"

ewi_detailed_table += r"""    \bottomrule
    \end{tabular}%
  }
  \label{tab:ewi-detailed}%
\end{table}%"""

print("\n--- EWI 详细表格 ---\n")
print(ewi_detailed_table)

# 保存所有表格到文件
output_path = base_dir / 'latex_table_kesr_ewi.tex'
with open(output_path, 'w') as f:
    f.write("% KESR Accuracy & F1 Table\n")
    f.write(kesr_table)
    f.write("\n\n")
    f.write("% KESR Recall & Precision Table\n")
    f.write(kesr_recall_prec_table)
    f.write("\n\n")
    f.write("% EWI Table\n")
    f.write(ewi_table)
    f.write("\n\n")
    f.write("% EWI Detailed Table\n")
    f.write(ewi_detailed_table)

print(f"\n\n所有 LaTeX 表格已保存到: {output_path}")
