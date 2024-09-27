import os
import pandas as pd

# Prepare a list to hold the best genomes
best_genomes = []
random_start = True
eas = [1, 2]
enemies = [2, 3, 5]

# Iterate over each EA and enemy combination
for ea in eas:
    for enemy in enemies:
        directory = f'runs/{"random/" if random_start else ""}ea{ea}/enemy{enemy}'
        best_fitness = 0
        best_genome = None
        for i in range(10):
            csv_path = os.path.join(directory, f'test_4_{i + 1}_100pop_30gen_enemy{enemy}/all_parents.csv')

            if os.path.exists(csv_path):
                genomes = pd.read_csv(csv_path)
                # Retrieve the best genome based on fitness
                run_best_fitness = genomes['Fitness'].idxmax()
                run_best_genome = genomes.loc[run_best_fitness]

                if run_best_fitness > best_fitness:
                    best_fitness = run_best_fitness
                    best_genome = run_best_genome
            else:
                print(f"File not found: {csv_path}")

        best_genomes.append({
            'ea': ea,
            'enemy': enemy,
            'generation': best_genome['Generation'],
            'genome': best_genome['Genome'],
            'fitness': best_genome['Fitness']
        })

best_genomes_df = pd.DataFrame(best_genomes)

output_csv_path = f'runs/{"random/" if random_start else ""}best_genomes.csv'
best_genomes_df.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
