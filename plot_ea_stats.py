import os
import matplotlib.pyplot as plt
import csv
import numpy as np

# Parameters
n_runs = 10
n_generations = 30
eas = [1, 2]
enemies = [2, 3, 5]
random_start = False
directory = f'runs/{"random/" if random_start else ""}'

# Create experiment directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)


def compute_mean_stdev(data):
    return np.mean(data, axis=0), np.std(data, axis=0)


# Loop over all enemies
for enemy in enemies:
    # Initialize arrays to store cumulative statistics for both algorithms
    all_max_fitness_algo1 = np.zeros((n_runs, n_generations))
    all_mean_fitness_algo1 = np.zeros((n_runs, n_generations))

    all_max_fitness_algo2 = np.zeros((n_runs, n_generations))
    all_mean_fitness_algo2 = np.zeros((n_runs, n_generations))

    for ea in eas:
        for i in range(10):
            experiment_dir = os.path.join(directory, f'ea{ea}/enemy{enemy}/test_4_{i + 1}_100pop_30gen_enemy{enemy}')
            csv_file_path = os.path.join(experiment_dir, 'all_statistics.csv')

            with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip the header

                # Loop through each row in the CSV
                for row in csvreader:
                    generation = int(row[0])
                    highest_fitness = float(row[1])
                    mean_fitness = float(row[2])
                    std_dev = float(row[3])
                    player_wins = int(row[4])
                    enemy_wins = int(row[5])
                    mean_player_energy = float(row[6])
                    mean_enemy_energy = float(row[7])
                    mean_play_time = float(row[8])
                    min_play_time = float(row[9])
                    max_play_time = float(row[10])

                    # Store the fitness values
                    if ea == 1:
                        all_max_fitness_algo1[i, generation - 1] = highest_fitness
                        all_mean_fitness_algo1[i, generation - 1] = mean_fitness
                    else:
                        all_max_fitness_algo2[i, generation - 1] = highest_fitness
                        all_mean_fitness_algo2[i, generation - 1] = mean_fitness

    # Calculate mean and std deviation for both EAs
    mean_max_fitness_algo1, std_max_fitness_algo1 = compute_mean_stdev(all_max_fitness_algo1)
    mean_mean_fitness_algo1, std_mean_fitness_algo1 = compute_mean_stdev(all_mean_fitness_algo1)

    mean_max_fitness_algo2, std_max_fitness_algo2 = compute_mean_stdev(all_max_fitness_algo2)
    mean_mean_fitness_algo2, std_mean_fitness_algo2 = compute_mean_stdev(all_mean_fitness_algo2)

    # Normalize the fitness values with max fitness per fitness function
    max_fitness_algo1, min_fitness_algo1 = np.max(all_max_fitness_algo1), np.min(all_mean_fitness_algo1)
    max_fitness_algo2, min_fitness_algo2 = np.max(all_max_fitness_algo2), np.min(all_mean_fitness_algo2)

    normalized_max_fitness_algo1 = (all_max_fitness_algo1 - 0) / (100 - 0)
    normalized_mean_fitness_algo1 = (all_mean_fitness_algo1 - 0) / (100 - 0)

    normalized_max_fitness_algo2 = (all_max_fitness_algo2 - 0) / (120 - 0)
    normalized_mean_fitness_algo2 = (all_mean_fitness_algo2 - 0) / (120 - 0)

    # Calculate mean and std deviation for normalized fitness values
    mean_norm_max_fitness_algo1, std_norm_max_fitness_algo1 = compute_mean_stdev(normalized_max_fitness_algo1)
    mean_norm_mean_fitness_algo1, std_norm_mean_fitness_algo1 = compute_mean_stdev(normalized_mean_fitness_algo1)

    mean_norm_max_fitness_algo2, std_norm_max_fitness_algo2 = compute_mean_stdev(normalized_max_fitness_algo2)
    mean_norm_mean_fitness_algo2, std_norm_mean_fitness_algo2 = compute_mean_stdev(normalized_mean_fitness_algo2)

    plot_dir = os.path.join(directory, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot Regular Fitness
    plt.figure(figsize=(10, 6))
    plt.title(f'Regular Fitness Comparison Plot: Enemy {enemy}')

    # Plot Algorithm 1 Regular Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_max_fitness_algo1, label="EA1 Mean Max Fitness", color="blue", marker='o')
    plt.fill_between(range(n_generations), mean_max_fitness_algo1 - std_max_fitness_algo1,
                     mean_max_fitness_algo1 + std_max_fitness_algo1, color='lightblue', alpha=0.5,
                     label="EA1 Max Fitness Std Dev")

    # Plot Algorithm 1 Regular Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_mean_fitness_algo1, label="EA1 Mean Avg Fitness", color="green",
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_mean_fitness_algo1 - std_mean_fitness_algo1,
                     mean_mean_fitness_algo1 + std_mean_fitness_algo1, color='lightgreen', alpha=0.5,
                     label="EA1 Avg Fitness Std Dev")

    # Plot Algorithm 2 Regular Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_max_fitness_algo2, label="EA2 Mean Max Fitness", color="red", marker='o')
    plt.fill_between(range(n_generations), mean_max_fitness_algo2 - std_max_fitness_algo2,
                     mean_max_fitness_algo2 + std_max_fitness_algo2, color='lightcoral', alpha=0.5,
                     label="EA2 Max Fitness Std Dev")

    # Plot Algorithm 2 Regular Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_mean_fitness_algo2, label="EA2 Mean Avg Fitness", color="orange",
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_mean_fitness_algo2 - std_mean_fitness_algo2,
                     mean_mean_fitness_algo2 + std_mean_fitness_algo2, color='navajowhite', alpha=0.5,
                     label="EA2 Avg Fitness Std Dev")

    # Labels and legend
    plt.ylim(0, 120)
    plt.yticks(np.arange(0, 121, 20))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(fontsize=16)
    plt.savefig(f'runs/{"random/" if random_start else ""}plots/enemy{enemy}_regular_fitness_comparison_plot.png')
    plt.show()

    # Plot Normalized Fitness
    plt.figure(figsize=(10, 6))
    plt.title(f'Normalized Fitness Comparison Plot: Enemy {enemy}')

    # Plot Algorithm 1 Normalized Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_norm_max_fitness_algo1, label="EA1 Norm Max Fitness", color="blue",
             marker='o')
    plt.fill_between(range(n_generations), mean_norm_max_fitness_algo1 - std_norm_max_fitness_algo1,
                     mean_norm_max_fitness_algo1 + std_norm_max_fitness_algo1, color='lightblue', alpha=0.5,
                     label="EA1 Norm Max Fitness Std Dev")

    # Plot Algorithm 1 Normalized Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_norm_mean_fitness_algo1, label="EA1 Norm Avg Fitness", color="green",
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_norm_mean_fitness_algo1 - std_norm_mean_fitness_algo1,
                     mean_norm_mean_fitness_algo1 + std_norm_mean_fitness_algo1, color='lightgreen', alpha=0.5,
                     label="EA1 Norm Avg Fitness Std Dev")

    # Plot Algorithm 2 Normalized Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_norm_max_fitness_algo2, label="EA2 Norm Max Fitness", color="red", marker='o')
    plt.fill_between(range(n_generations), mean_norm_max_fitness_algo2 - std_norm_max_fitness_algo2,
                     mean_norm_max_fitness_algo2 + std_norm_max_fitness_algo2, color='lightcoral', alpha=0.5,
                     label="EA2 Norm Max Fitness Std Dev")

    # Plot Algorithm 2 Normalized Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_norm_mean_fitness_algo2, label="EA2 Norm Avg Fitness", color="orange",
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_norm_mean_fitness_algo2 - std_norm_mean_fitness_algo2,
                     mean_norm_mean_fitness_algo2 + std_norm_mean_fitness_algo2, color='navajowhite', alpha=0.5,
                     label="EA2 Norm Avg Fitness Std Dev")

    # Labels and legend
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Generation')
    plt.ylabel('Normalized Fitness')
    plt.legend(fontsize=16)
    plt.savefig(f'runs/{"random/" if random_start else ""}plots/enemy{enemy}_normalized_fitness_comparison_plot.png')
    plt.show()
