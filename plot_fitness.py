import os
import matplotlib.pyplot as plt

# Directory where the .txt files are stored
directory_path = 'test_1_100pop_30gen_enemy3/'

# Lists to store generation data, mean fitness values, and standard deviations
generations = []
mean_fitness_values = []
std_dev_values = []

# Loop through the files and read the data
for generation in range(1, 51):  # Assuming 50 generations
    file_path = os.path.join(directory_path, f"fitness_stats_generation_{generation}.txt")

    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extract the required data from the file
        generation_num = int(lines[0].strip().split()[-1])
        highest_fitness = float(lines[1].strip().split()[-1])
        mean_fitness = float(lines[2].strip().split()[-1])
        std_dev = float(lines[3].strip().split()[-1])

        # Append the data to lists
        generations.append(generation_num)
        mean_fitness_values.append(mean_fitness)
        std_dev_values.append(std_dev)

# Convert lists to numpy arrays (optional, if you prefer working with numpy)
import numpy as np

mean_fitness_values = np.array(mean_fitness_values)
std_dev_values = np.array(std_dev_values)

# Plotting the data
plt.figure(figsize=(10, 6))

# Mean fitness line plot
plt.plot(generations, mean_fitness_values, label="Mean Fitness", color="blue", marker='o')

# Upper and lower bounds (mean ± std_dev)
upper_bound = mean_fitness_values + std_dev_values
lower_bound = mean_fitness_values - std_dev_values

# Fill between the upper and lower bounds
plt.fill_between(generations, lower_bound, upper_bound, color='lightblue', alpha=0.5, label="Std Dev Bounds")

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Mean Fitness with Standard Deviation over Generations')
plt.legend()

# Show the plot
plt.show()
