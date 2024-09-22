import os
import matplotlib.pyplot as plt
import csv
import numpy as np

# Path to the CSV file
directory = 'test_1_100pop_30gen_enemy3'
csv_file_path = os.path.join(directory, 'all_statistics.csv')

# Lists to store generation data, fitness, wins, energy, and play times
generations = []
highest_fitness_values = []
mean_fitness_values = []
std_dev_values = []
player_wins_values = []
enemy_wins_values = []
mean_player_energy_values = []
mean_enemy_energy_values = []
mean_play_times = []
min_play_times = []
max_play_times = []

# Read the CSV file and extract data
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Skip the header
    next(csvreader)

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

        # Append the data to lists
        generations.append(generation)
        highest_fitness_values.append(highest_fitness)
        mean_fitness_values.append(mean_fitness)
        std_dev_values.append(std_dev)
        player_wins_values.append(player_wins)
        enemy_wins_values.append(enemy_wins)
        mean_player_energy_values.append(mean_player_energy)
        mean_enemy_energy_values.append(mean_enemy_energy)
        mean_play_times.append(mean_play_time)
        min_play_times.append(min_play_time)
        max_play_times.append(max_play_time)

# Convert lists to numpy arrays for easier manipulation
generations = np.array(generations)
mean_fitness_values = np.array(mean_fitness_values)
std_dev_values = np.array(std_dev_values)
highest_fitness_values = np.array(highest_fitness_values)
player_wins_values = np.array(player_wins_values)
enemy_wins_values = np.array(enemy_wins_values)
mean_player_energy_values = np.array(mean_player_energy_values)
mean_enemy_energy_values = np.array(mean_enemy_energy_values)
mean_play_times = np.array(mean_play_times)
min_play_times = np.array(min_play_times)
max_play_times = np.array(max_play_times)

# Plot Mean Fitness with Standard Deviation
plt.figure(figsize=(10, 6))

# Plot mean fitness
plt.plot(generations, mean_fitness_values, label="Mean Fitness", color="blue", marker='o')

# Plot the highest fitness
plt.plot(generations, highest_fitness_values, label="Highest Fitness", color="green", linestyle='--')

# Plot bounds for std deviation (mean ± std_dev)
upper_bound = mean_fitness_values + std_dev_values
lower_bound = mean_fitness_values - std_dev_values
plt.fill_between(generations, lower_bound, upper_bound, color='lightblue', alpha=0.5, label="Std Dev Bounds")

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Mean and Highest Fitness with Standard Deviation over Generations')
plt.legend()
plt.savefig(os.path.join(directory, 'fitness_plot.png'))

# Show the plot
plt.show()

# Plot Player Wins vs Enemy Wins over Generations
plt.figure(figsize=(10, 6))

plt.plot(generations, player_wins_values, label="Player Wins", color="blue", marker='o')
plt.plot(generations, enemy_wins_values, label="Enemy Wins", color="red", marker='x')

plt.xlabel('Generation')
plt.ylabel('Wins')
plt.title('Player Wins vs Enemy Wins over Generations')
plt.legend()
plt.savefig(os.path.join(directory, 'wins_plot.png'))

plt.show()

# Plot Mean Energy (Player vs Enemy) over Generations
plt.figure(figsize=(10, 6))

plt.plot(generations, mean_player_energy_values, label="Mean Player Energy", color="blue", marker='o')
plt.plot(generations, mean_enemy_energy_values, label="Mean Enemy Energy", color="red", marker='x')

plt.xlabel('Generation')
plt.ylabel('Energy')
plt.title('Mean Player Energy vs Mean Enemy Energy over Generations')
plt.legend()
plt.savefig(os.path.join(directory, 'energy_plot.png'))

plt.show()

# Plot Mean Play Time with Min and Max Play Times over Generations
plt.figure(figsize=(10, 6))

# Plot mean play time
plt.plot(generations, mean_play_times, label="Mean Play Time", color="purple", marker='o')

# Plot bounds for min and max play time
plt.fill_between(generations, min_play_times, max_play_times, color='violet', alpha=0.3, label="Min-Max Play Time Range")

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Play Time (seconds)')
plt.title('Mean Play Time with Min and Max over Generations')
plt.legend()
plt.savefig(os.path.join(directory, 'play_time_plot.png'))

plt.show()
