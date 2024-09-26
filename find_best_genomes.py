import os
import pandas as pd

# Prepare a list to hold the best genomes
best_genomes = []

# Iterate over each evolutionary algorithm (EA) and enemy combination
for ea in [1, 2]:
    for enemy in [2, 3, 5]:
        directory = f'runs/ea{ea}/enemy{enemy}'
        best_fitness = 0
        best_genome = None
        for i in range(10):
            csv_path = os.path.join(directory, f'test_4_{i + 1}_100pop_30gen_enemy{enemy}/all_parents.csv')
            # Check if the file exists to avoid errors
            if os.path.exists(csv_path):
                genomes = pd.read_csv(csv_path)
                # Retrieve the best genome based on fitness
                run_best_fitness = genomes['Fitness'].idxmax()
                run_best_genome = genomes.loc[run_best_fitness]

                # Append the details to the list
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

# Convert the list of best genomes to a DataFrame
best_genomes_df = pd.DataFrame(best_genomes)

# Write the DataFrame to a CSV file
output_csv_path = 'runs/best_genomes.csv'
best_genomes_df.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
