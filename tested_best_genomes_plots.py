import ast
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random_trained = False
random_tested = True
directory = f'runs/{"random/" if random_trained else ""}'
csv_path = os.path.join(directory, f'best_genomes_{"random" if random_tested else "fixed"}.csv')

df = pd.read_csv(csv_path)
# Convert the 'gain' column from strings to actual lists using ast.literal_eval
df['gain'] = df['gain'].apply(ast.literal_eval)

# Flatten the DataFrame: create a new DataFrame for plotting
flattened_data = []

for index, row in df.iterrows():
    for gain in row['gain']:
        flattened_data.append({'ea': row['ea'], 'enemy': row['enemy'], 'gain': gain})

# Create a new DataFrame from the flattened data
flattened_df = pd.DataFrame(flattened_data)

# Create a boxplot where the x-axis is the enemy and EA algorithm comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='enemy', y='gain', hue='ea', data=flattened_df, palette='Set2')

# Set titles and labels
plt.title(f'Gain Distribution by Enemy and EA: {"Random" if random_trained else "Fixed"} position trained, {"Random" if random_tested else "Fixed"} position tested')
plt.xlabel('Enemy')
plt.ylabel('Gain')

# Adjust x-axis ticks to only show the unique enemy values
unique_enemies = sorted(flattened_df['enemy'].unique())
plt.xticks(ticks=range(len(unique_enemies)), labels=unique_enemies)

# Show the legend for EA algorithms
plt.legend(title='EA')

# Save or show the figure
plt.tight_layout()
plt.savefig(os.path.join(directory, f'plots/{"r" if random_trained else "f"}trained_{"r" if random_tested else "f"}test_gain_boxplot.png'))
plt.show()
