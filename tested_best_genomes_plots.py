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
# Convert the gain from strings to lists
df['gain'] = df['gain'].apply(ast.literal_eval)
# Replace the list with the sum of the values in the list
df['gain'] = df['gain'].apply(sum)

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
