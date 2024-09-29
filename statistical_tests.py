import numpy as np
from scipy.stats import ttest_rel, mannwhitneyu, shapiro
import os
import pandas as pd
import numpy as np


def perform_statistical_tests(data1, data2, test_type='ttest'):
    if test_type == 'ttest':
        test_stat, p_value = ttest_rel(data1, data2)
    elif test_type == 'mannwhitney':
        test_stat, p_value = mannwhitneyu(data1, data2)
    else:
        raise ValueError("Invalid test_type. Use 'ttest' or 'mannwhitney'.")

    return test_stat, p_value


def perform_normality_test(data):
    stat, p_value = shapiro(data)
    return stat, p_value


def decide_test(data1, data2, alpha=0.05):
    # Calculate differences
    differences = np.array(data1) - np.array(data2)

    # Perform normality test on the differences
    stat, p_value = perform_normality_test(differences)
    print(f"Normality Test (Shapiro-Wilk) p-value: {p_value}")

    # Decide on test based on normality
    if p_value > alpha:
        print("Data is normally distributed. Performing paired t-test...")
        test_stat, test_p_value = perform_statistical_tests(data1, data2, test_type='ttest')
    else:
        print("Data is not normally distributed. Performing Mann-Whitney U test...")
        test_stat, test_p_value = perform_statistical_tests(data1, data2, test_type='mannwhitney')

    return {
        'test_stat': test_stat,
        'p_value': test_p_value,
        'normality_p_value': p_value,
        'test_type': 'ttest' if p_value > alpha else 'mannwhitney'
    }

# Loop over both EA
for ea in [1, 2]:
    for enemy in [2, 3, 5]:
        ea1_results = []
        ea2_results = []

        for i in range(10):
            # Load the results for EA1
            directory_ea1 = f'runs/ea1/enemy{enemy}/test_4_{i + 1}_100pop_30gen_enemy{enemy}'
            df_ea1 = pd.read_csv(os.path.join(directory_ea1, 'all_statistics.csv'))
            gain_ea1 = df_ea1['Mean Player Energy'].mean() - df_ea1['Mean Enemy Energy'].mean()
            ea1_results.append(gain_ea1)

            # Load the results for EA2
            directory_ea2 = f'runs/ea2/enemy{enemy}/test_4_{i + 1}_100pop_30gen_enemy{enemy}'
            df_ea2 = pd.read_csv(os.path.join(directory_ea2, 'all_statistics.csv'))
            gain_ea2 = df_ea2['Mean Player Energy'].mean() - df_ea2['Mean Enemy Energy'].mean()
            ea2_results.append(gain_ea2)

        result = decide_test(ea1_results, ea2_results)
        print(f"Results for EA1 vs EA2 for Enemy {enemy}: {result}")

