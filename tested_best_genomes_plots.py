import ast
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random_trained = False
random_tested = False
directory = f'runs/generalist'
csv_path = os.path.join(directory, f'tested_best_genomes.csv')

df = pd.read_csv(csv_path)
# Convert the 'gain' column from strings to actual lists using ast.literal_eval
df['gain'] = df['gain'].apply(ast.literal_eval)

# Replace the list with the sum of the values in the list
df['gain'] = df['gain'].apply(sum)

# # Flatten the DataFrame: create a new DataFrame for plotting
# flattened_data = []
#
# for index, row in df.iterrows():
#     for gain in row['gain']:
#         flattened_data.append({'ea': row['ea'], 'group': row['group'], 'gain': gain})
#
# # Create a new DataFrame from the flattened data
# flattened_df = pd.DataFrame(flattened_data)

# Create a boxplot where the x-axis is the enemy and EA algorithm comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='gain', hue='ea', data=df, palette='coolwarm')

# Set titles and labels
plt.title(f'Gain Distribution by Enemy Group and EA')
plt.xlabel('Enemy Group')
plt.ylabel('Gain')
plt.legend(title='EA', title_fontsize="16", fontsize=16)

# Save or show the figure
plt.tight_layout()
plt.savefig(os.path.join(directory, f'plots/gain_boxplots.png'))
plt.show()
