import os
import matplotlib.pyplot as plt

# Directory where the .txt files are stored
directory_path = 'test_1_100pop_30gen_enemy3/'

# Lists to store generation data, mean fitness values, and standard deviations
generations = []
player_won = []
enemy_won = []

# Loop through all files in the directory that match the pattern
for file_name in os.listdir(directory_path):
    if file_name.startswith("winner_stats_generation_") and file_name.endswith(".txt"):
        # Extract generation number from the filename
        generation_num = int(file_name.split("_")[-1].split(".")[0])

        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)

        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Extract the required data from the file
            player_wins = float(lines[1].strip().split()[-1])
            enemy_wins = float(lines[2].strip().split()[-1])

            # Append the data to lists
            generations.append(generation_num)
            player_won.append(player_wins)
            enemy_won.append(enemy_wins)

# Sort the data by generation (to ensure it's in the correct order)
sorted_data = sorted(zip(generations, player_won, enemy_won))
generations, player_won, enemy_won = zip(*sorted_data)

# Convert lists to numpy arrays (optional, if you prefer working with numpy)
import numpy as np

player_won = np.array(player_won)
enemy_won = np.array(enemy_won)

# Plotting the data
plt.figure(figsize=(10, 6))

# Wins line plot

bar_width = 0.35 # Set the positions of the bars on the x-axis
r1 = np.arange(len(generations)) # Positions for player wins bars
r2 = [x + bar_width for x in r1] # Offset for enemy wins bars
plt.bar(r1, player_won, color='blue', width=bar_width, edgecolor='grey', label='Player wins') # Create the bars for player and enemy wins
plt.bar(r2, enemy_won, color='red', width=bar_width, edgecolor='grey', label='Enemy wins') # Add labels and title

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Wins')
plt.title('Wins per Generation')
plt.xticks([r + bar_width/2 for r in r1], generations) # Add xticks on the middle of the bars
plt.legend()

# Show the plot
plt.show()
