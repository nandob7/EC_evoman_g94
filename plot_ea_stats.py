import os
import matplotlib.pyplot as plt
import csv
import numpy as np

# Parameters
n_runs = 7
n_generations = 30
eas = [1, 2]
groups = ['all_enemies', '2358']
random_start = False
directory = f'runs/generalist'


def compute_mean_stdev(data):
    return np.mean(data, axis=0), np.std(data, axis=0)


# Loop over all enemies
for group in groups:
    # Initialize arrays to store cumulative statistics for both algorithms
    all_max_fitness_algo1 = np.zeros((n_runs, n_generations))
    all_mean_fitness_algo1 = np.zeros((n_runs, n_generations))

    all_max_fitness_algo2 = np.zeros((n_runs, n_generations))
    all_mean_fitness_algo2 = np.zeros((n_runs, n_generations))

    for ea in eas:
        for run in range(n_runs):
            experiment_dir = os.path.join(directory, f'ea{ea}_{group}/test_{run + 1}')
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
                    mean_player_energy = float(row[4])
                    mean_enemy_energy = float(row[5])
                    mean_play_time = float(row[6])
                    min_play_time = float(row[7])
                    max_play_time = float(row[8])

                    # Store the fitness values
                    if ea == 1:
                        all_max_fitness_algo1[run, generation - 1] = highest_fitness
                        all_mean_fitness_algo1[run, generation - 1] = mean_fitness
                    else:
                        all_max_fitness_algo2[run, generation - 1] = highest_fitness
                        all_mean_fitness_algo2[run, generation - 1] = mean_fitness

    # Calculate mean and std deviation for both EAs
    mean_max_fitness_algo1, std_max_fitness_algo1 = compute_mean_stdev(all_max_fitness_algo1)
    mean_mean_fitness_algo1, std_mean_fitness_algo1 = compute_mean_stdev(all_mean_fitness_algo1)

    mean_max_fitness_algo2, std_max_fitness_algo2 = compute_mean_stdev(all_max_fitness_algo2)
    mean_mean_fitness_algo2, std_mean_fitness_algo2 = compute_mean_stdev(all_mean_fitness_algo2)

    # Normalize the fitness values with max fitness per fitness function
    max_fitness_algo1, min_fitness_algo1 = np.max(all_max_fitness_algo1), np.min(all_mean_fitness_algo1)
    max_fitness_algo2, min_fitness_algo2 = np.max(all_max_fitness_algo2), np.min(all_mean_fitness_algo2)

    plot_dir = os.path.join(directory, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot Regular Fitness
    group_name = 'All Enemies' if group == 'all_enemies' else '{2, 3, 5, 8}'
    plt.figure(figsize=(12, 8))
    plt.title(f'Fitness Comparison Plot: Enemy Group {group_name}')
    # Colorblind-friendly color palette
    colors = [
        '#db69db',
        '#db69db',
        '#bc8ae3',
        '#bc8ae3',
        '#69DB69',
        '#69DB69',
        '#9FCC7C',
        '#9FCC7C',
    ]

    # Plot Algorithm 1 Regular Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_max_fitness_algo1, label="EA1 Mean Max Fitness", color=colors[0], marker='o')
    plt.fill_between(range(n_generations), mean_max_fitness_algo1 - std_max_fitness_algo1,
                     mean_max_fitness_algo1 + std_max_fitness_algo1, color=colors[1], alpha=0.3,
                     label="EA1 Max Fitness Std Dev")

    # Plot Algorithm 1 Regular Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_mean_fitness_algo1, label="EA1 Mean Avg Fitness", color=colors[2],
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_mean_fitness_algo1 - std_mean_fitness_algo1,
                     mean_mean_fitness_algo1 + std_mean_fitness_algo1, color=colors[3], alpha=0.3,
                     label="EA1 Avg Fitness Std Dev")

    # Plot Algorithm 2 Regular Max Fitness with Std Dev
    plt.plot(range(n_generations), mean_max_fitness_algo2, label="EA2 Mean Max Fitness", color=colors[4], marker='o')
    plt.fill_between(range(n_generations), mean_max_fitness_algo2 - std_max_fitness_algo2,
                     mean_max_fitness_algo2 + std_max_fitness_algo2, color=colors[5], alpha=0.3,
                     label="EA2 Max Fitness Std Dev")

    # Plot Algorithm 2 Regular Mean Fitness with Std Dev
    plt.plot(range(n_generations), mean_mean_fitness_algo2, label="EA2 Mean Avg Fitness", color=colors[6],
             linestyle='--', marker='x')
    plt.fill_between(range(n_generations), mean_mean_fitness_algo2 - std_mean_fitness_algo2,
                     mean_mean_fitness_algo2 + std_mean_fitness_algo2, color=colors[7], alpha=0.3,
                     label="EA2 Avg Fitness Std Dev")

    # Labels and legend
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(plot_dir, f'fitness_comparison_{group}.png'))
    plt.show()

    # # Plot Normalized Fitness
    # plt.figure(figsize=(10, 6))
    # plt.title(f'Normalized Fitness Comparison Plot: Enemy {group_name}')
    #
    # # Plot Algorithm 1 Normalized Max Fitness with Std Dev
    # plt.plot(range(n_generations), mean_norm_max_fitness_algo1, label="EA1 Norm Max Fitness", color="blue",
    #          marker='o')
    # plt.fill_between(range(n_generations), mean_norm_max_fitness_algo1 - std_norm_max_fitness_algo1,
    #                  mean_norm_max_fitness_algo1 + std_norm_max_fitness_algo1, color='lightblue', alpha=0.5,
    #                  label="EA1 Norm Max Fitness Std Dev")
    #
    # # Plot Algorithm 1 Normalized Mean Fitness with Std Dev
    # plt.plot(range(n_generations), mean_norm_mean_fitness_algo1, label="EA1 Norm Avg Fitness", color="green",
    #          linestyle='--', marker='x')
    # plt.fill_between(range(n_generations), mean_norm_mean_fitness_algo1 - std_norm_mean_fitness_algo1,
    #                  mean_norm_mean_fitness_algo1 + std_norm_mean_fitness_algo1, color='lightgreen', alpha=0.5,
    #                  label="EA1 Norm Avg Fitness Std Dev")
    #
    # # Plot Algorithm 2 Normalized Max Fitness with Std Dev
    # plt.plot(range(n_generations), mean_norm_max_fitness_algo2, label="EA2 Norm Max Fitness", color="red", marker='o')
    # plt.fill_between(range(n_generations), mean_norm_max_fitness_algo2 - std_norm_max_fitness_algo2,
    #                  mean_norm_max_fitness_algo2 + std_norm_max_fitness_algo2, color='lightcoral', alpha=0.5,
    #                  label="EA2 Norm Max Fitness Std Dev")
    #
    # # Plot Algorithm 2 Normalized Mean Fitness with Std Dev
    # plt.plot(range(n_generations), mean_norm_mean_fitness_algo2, label="EA2 Norm Avg Fitness", color="orange",
    #          linestyle='--', marker='x')
    # plt.fill_between(range(n_generations), mean_norm_mean_fitness_algo2 - std_norm_mean_fitness_algo2,
    #                  mean_norm_mean_fitness_algo2 + std_norm_mean_fitness_algo2, color='navajowhite', alpha=0.5,
    #                  label="EA2 Norm Avg Fitness Std Dev")
    #
    # # Labels and legend
    # plt.ylim(0, 1)
    # plt.yticks(np.arange(0, 1.1, 0.2))
    # plt.xlabel('Generation')
    # plt.ylabel('Normalized Fitness')
    # plt.legend(fontsize=16)
    # plt.savefig(plot_dir)
    # plt.show()
