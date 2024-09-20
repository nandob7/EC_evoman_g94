import os
import matplotlib.pyplot as plt

# Directory where the .txt files are stored
directory_path = 'test_1_100pop_30gen_enemy3/'

# Lists to store generation data, player energy, and enemy energy
generations = []
player_energy_values = []
enemy_energy_values = []

# Loop through all files in the directory that match the pattern
for file_name in os.listdir(directory_path):
    if file_name.startswith("energy_stats_generation_") and file_name.endswith(".txt"):
        # Extract generation number from the filename
        generation_num = int(file_name.split("_")[-1].split(".")[0])

        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)

        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Extract the required data from the file
            player_energy = float(lines[1].strip().split()[-1])
            enemy_energy = float(lines[2].strip().split()[-1])

            # Append the data to lists
            generations.append(generation_num)
            player_energy_values.append(player_energy)
            enemy_energy_values.append(enemy_energy)

# Sort the data by generation (to ensure it's in the correct order)
sorted_data = sorted(zip(generations, player_energy_values, enemy_energy_values))
generations, player_energy_values, enemy_energy_values = zip(*sorted_data)

# Convert lists to numpy arrays (optional, if you prefer working with numpy)
import numpy as np

player_energy_values = np.array(player_energy_values)
enemy_energy_values = np.array(enemy_energy_values)

# Plotting the data
plt.figure(figsize=(10, 6))

# Mean fitness line plot
plt.plot(generations, player_energy_values, label="Player energy", color="blue")
plt.plot(generations, enemy_energy_values, label="Enemy energy", color="red")
# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Energy')
plt.title('Average energy of player and enemy per Generation')
plt.legend()

# Show the plot
plt.show()
