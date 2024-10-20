import os
import pandas as pd

# Prepare a list to hold the best genomes
best_genomes = []
random_start = False
groups = ['all_enemies', '2358']
eas = [1, 2]

# Iterate over each EA and enemy group
for ea in eas:
    for group in groups:
        directory = f'runs/generalist/ea{ea}_{group}'
        for run in range(10):
            csv_path = os.path.join(directory, f'test_{run + 1}/all_parents.csv')

            genomes = pd.read_csv(csv_path)
            # Retrieve the best genome based on fitness
            run_best_fitness = genomes['Fitness'].idxmax()
            run_best_genome = genomes.loc[run_best_fitness]

            best_genomes.append({
                'ea': ea,
                'group': group,
                'run': run + 1,
                'generation': run_best_genome['Generation'],
                'genome': run_best_genome['Genome'],
                'fitness': run_best_genome['Fitness']
            })

# # Set up directory and file path
# directory = 'runs/generalist'
# run = 4
# csv_path = os.path.join(directory, f'run_{run}/all_parents.csv')
#
# # Load the genomes CSV
# genomes = pd.read_csv(csv_path)
#
# # Group by 'Generation' and find the genome with the highest fitness in each group
# grouped = genomes.groupby('Generation')
# for generation, group in grouped:
#     # Get the best genome in this generation based on fitness
#     best_genome = group.loc[group['Fitness'].idxmax()]
#
#     best_genomes.append({
#         'generation': best_genome['Generation'],
#         'genome': best_genome['Genome'],
#         'fitness': best_genome['Fitness']
#     })
#
best_genomes_df = pd.DataFrame(best_genomes)

# output_csv_path = f'runs/{"random/" if random_start else ""}best_genomes.csv'
output_csv_path = f'runs/generalist/best_genomes.csv'
best_genomes_df.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
