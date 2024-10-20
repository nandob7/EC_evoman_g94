import numpy as np
import os
import csv

from evoman.environment import Environment
from demo_controller import player_controller

# Parameters
desired_generation = 750
number_of_hidden_neurons = 10
experiment_name = f'runs/competition/run_1'
input_size = 20  # Hardcoded number of sensors

# Create experiment directory if it doesn't exist
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize the environment with the hardcoded controller
env_test = Environment(
    experiment_name=experiment_name,
    playermode="ai",
    player_controller=player_controller(number_of_hidden_neurons),  # Pass the controller directly
    enemymode="static",
    level=2,
    randomini='no',
    savelogs='no',
    # speed='normal',
    # visuals=True
    # multiplemode='no'
    )

def load_best_genome_from_csv(csv_file_path, generation):
    """
    Load the best genome from a given generation in the CSV file.
        csv_file_path: Path to the CSV file containing saved genomes and fitness values.
        generation: The generation number for which to load the best genome.

        The best genome (as a numpy array) and its fitness value.
    """
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the header
        next(csvreader)

        for row in csvreader:
            gen_num = int(row[0])
            if gen_num == generation:
                genome_str = row[1]
                fitness = float(row[2])

                # Convert genome string back into a numpy array
                genome = np.fromstring(genome_str, sep=' ')
                return genome, fitness
        # If no genome is found for the given generation, return None
        return None, None


def calc_cust_fitness(player_life, enemy_life, time):
    return (0.8 * (100 - enemy_life)) + (
        0.4 * player_life if player_life > 75 else 0.2 * player_life if player_life > 50 else 0.1 * player_life) - np.log(
        time)


# File path to the CSV
csv_file_path = os.path.join(experiment_name, 'all_parents.csv')


# Load the best genome from the specified generation
best_genome, best_fitness = load_best_genome_from_csv(csv_file_path, desired_generation)

# init health dictionary for each enemy
health = {f'{key}': [] for key in range(1, 9)}

# Play the environment using the best genome and display its fitness
for en in range(1, 9):
    # Update the enemy
    env_test.update_parameter('enemies', [en])

    fitness, player_life, enemy_life, play_time = env_test.play(best_genome)
    health[f'{en}'].append([player_life, enemy_life])

# LaTex table output
player_string = ' & '.join([f'{value[0][0]}' for key, value in health.items()])
enemy_string = ' & '.join([f'{value[0][1]}' for key, value in health.items()])
print(f'Player health & {player_string}')
print(f'Enemy health & {enemy_string}')

# Save genome as np.array in txt
np.savetxt('94.txt', best_genome)
