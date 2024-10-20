import ast
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

random_trained = False
random_tested = False
directory = f'runs/generalist'
csv_path = os.path.join(directory, f'tested_best_genomes.csv')

df = pd.read_csv(csv_path)

df['gain'] = df['gain'].apply(ast.literal_eval)
df['gain'] = df['gain'].apply(sum)

for group in df['group'].unique():
    ea1_data = df[(df['ea'] == 1) & (df['group'] == group)]['gain']
    ea2_data = df[(df['ea'] == 2) & (df['group'] == group)]['gain']

    t_stat, p_value = stats.ttest_ind(ea1_data, ea2_data)

    print(f"Group: {group}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
