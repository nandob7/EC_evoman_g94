import os
import matplotlib.pyplot as plt
import pandas as pd

show_plots = False
print_fitness_stats = False
random_start = True

# Loop over both EA
for ea in [1, 2]:
    # Loop over each enemy
    for enemy in [2, 3, 5]:
        # Loop for the 10 experiment runs
        for i in range(10):
            directory = f'runs/{"random/" if random_start else ""}ea{ea}/enemy{enemy}/test_4_{i + 1}_100pop_30gen_enemy{enemy}'
            csv_file_path = os.path.join(directory, 'all_statistics.csv')
            df = pd.read_csv(csv_file_path)

            # Extract relevant columns to numpy arrays for easier manipulation
            generations = df['Generation'].to_numpy()
            highest_fitness_values = df['Highest Fitness'].to_numpy()
            mean_fitness_values = df['Mean Fitness'].to_numpy()
            std_dev_values = df['Std Dev Fitness'].to_numpy()
            player_wins_values = df['Player Wins'].to_numpy()
            enemy_wins_values = df['Enemy Wins'].to_numpy()
            mean_player_energy_values = df['Mean Player Energy'].to_numpy()
            mean_enemy_energy_values = df['Mean Enemy Energy'].to_numpy()
            mean_play_times = df['Mean Play Time'].to_numpy()
            min_play_times = df['Min Play Time'].to_numpy()
            max_play_times = df['Max Play Time'].to_numpy()

            # If you want to print fitness stats
            if print_fitness_stats:
                print(f'Mean Fitness: {mean_fitness_values.mean()}')
                print(f'Highest fitness: {highest_fitness_values.max()}')
                print(f'Standard Deviation: {std_dev_values.mean()}')

            # Create Fitness Plot
            plt.figure(figsize=(10, 6))
            plt.plot(generations, mean_fitness_values, label="Mean Fitness", color="blue", marker='o')
            plt.plot(generations, highest_fitness_values, label="Highest Fitness", color="green", linestyle='--')

            # Plot bounds for std deviation (mean ± std_dev)
            upper_bound = mean_fitness_values + std_dev_values
            lower_bound = mean_fitness_values - std_dev_values
            plt.fill_between(generations, lower_bound, upper_bound, color='lightblue', alpha=0.5,
                             label="Std Dev Bounds")

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'Fitness Plot: EA {ea}, Enemy {enemy}')
            plt.legend()
            plt.savefig(os.path.join(directory, f'ea{ea}_enemy{enemy}_run{i + 1}_fitness_plot.png'))

            if show_plots:
                plt.show()

            # Plot Player Wins vs Enemy Wins over Generations
            plt.figure(figsize=(10, 6))

            plt.plot(generations, player_wins_values, label="Player Wins", color="blue", marker='o')
            plt.plot(generations, enemy_wins_values, label="Enemy Wins", color="red", marker='x')

            plt.xlabel('Generation')
            plt.ylabel('Wins')
            plt.title(f'Wins Plot: : EA {ea}, Enemy {enemy}')
            plt.legend()
            plt.savefig(os.path.join(directory, f'ea{ea}_enemy{enemy}_run{i + 1}_wins_plot.png'))

            if show_plots:
                plt.show()

            # Plot Mean Energy (Player vs Enemy) over Generations
            plt.figure(figsize=(10, 6))

            plt.plot(generations, mean_player_energy_values, label="Mean Player Energy", color="blue", marker='o')
            plt.plot(generations, mean_enemy_energy_values, label="Mean Enemy Energy", color="red", marker='x')

            plt.xlabel('Generation')
            plt.ylabel('Energy')
            plt.title(f'Energy Plot: : EA {ea}, Enemy {enemy}')
            plt.legend()
            plt.savefig(os.path.join(directory, f'ea{ea}_enemy{enemy}_run{i + 1}_energy_plot.png'))

            if show_plots:
                plt.show()

            # Plot Mean Play Time with Min and Max Play Times over Generations
            plt.figure(figsize=(10, 6))
            plt.plot(generations, mean_play_times, label="Mean Play Time", color="purple", marker='o')
            plt.fill_between(generations, min_play_times, max_play_times, color='violet', alpha=0.3,
                             label="Min-Max Play Time Range")

            plt.xlabel('Generation')
            plt.ylabel('Play Time (seconds)')
            plt.title('Mean Play Time with Min and Max over Generations')
            plt.legend()
            plt.savefig(os.path.join(directory, f'ea{ea}_enemy{enemy}_run{i + 1}_play_time_plot.png'))

            if show_plots:
                plt.show()

            plt.close('all')

        print(f'Generated plots for ea{ea}, enemy {enemy}')
