#!/usr/bin/env python3
"""
生成按难度级别分成四个表的 KESR 统计表格
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

def get_raw_stat(df, difficulty, stat_name):
    metric_type = f'raw_stats_{difficulty}'
    row = df[(df['metric_type'] == metric_type) & (df['metric_name'] == stat_name)]
    if len(row) > 0:
        val = row['value'].values[0]
        return int(val) if pd.notna(val) else 0
    return 0

def calculate_kesr_metrics(df, difficulty):
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

# 收集所有数据
results = {}
for agent_name, filename in file_mapping.items():
    csv_path = base_dir / filename
    if not csv_path.exists():
        print(f'File not found: {csv_path}')
        continue
    df = pd.read_csv(csv_path)
    kesr_data = {}
    for difficulty in ['overall', 'easy', 'mid', 'hard']:
        kesr_data[difficulty] = calculate_kesr_metrics(df, difficulty)
    results[agent_name] = kesr_data

# 生成四个难度级别的表格
difficulty_labels = {'overall': 'Overall', 'easy': 'Easy', 'mid': 'Medium', 'hard': 'Hard'}

all_tables = '% KESR Tables by Difficulty Level\n% F1, Recall, Precision, Accuracy, TP, FP, FN, TN\n\n'

for diff_key, diff_label in difficulty_labels.items():
    table = f'''\\begin{{table}}[h]
  \\centering
  \\caption{{Key Essential State Recognition (KESR) - {diff_label} Difficulty}}
  \\label{{tab:kesr-{diff_key}}}
  \\resizebox{{\\columnwidth}}{{!}}{{
    \\begin{{tabular}}{{l|r|r|r|r!{{\\vrule width 0.8pt}}r|r|r|r}}
      \\toprule
      \\textbf{{Approach}} & \\textbf{{F1}} & \\textbf{{Recall}} & \\textbf{{Precision}} & \\textbf{{Accuracy}} & \\textbf{{TP}} & \\textbf{{FP}} & \\textbf{{FN}} & \\textbf{{TN}} \\\\
      \\midrule
'''
    
    for agent_name in file_mapping.keys():
        if agent_name not in results:
            continue
        data = results[agent_name][diff_key]
        table += f'      {agent_name} & {data["f1"]:.2f} & {data["recall"]:.2f} & {data["precision"]:.2f} & {data["accuracy"]:.2f} & {data["tp"]} & {data["fp"]} & {data["fn"]} & {data["tn"]} \\\\\n'
    
    table += '''      \\bottomrule
    \\end{tabular}
  }
\\end{table}

'''
    all_tables += table

print(all_tables)

# 保存到文件
output_path = base_dir / 'latex_table_kesr_by_difficulty.tex'
with open(output_path, 'w') as f:
    f.write(all_tables)

print(f"\n\n表格已保存到: {output_path}")
