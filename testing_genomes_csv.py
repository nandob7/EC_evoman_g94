import numpy as np
import os
import csv
from evoman.environment import Environment
from evoman.controller import Controller

# Parameters
number_of_hidden_neurons = 10
experiment_name = 'test_1_100pop_30gen_enemy3'
input_size = 20  # Hardcoded number of sensors

# Create experiment directory if it doesn't exist
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize a neural controller
neural_controller = Controller(input_size, number_of_hidden_neurons)

# Initialize the environment with the hardcoded controller
env_test = Environment(
    experiment_name=experiment_name,
    playermode="ai",
    player_controller=neural_controller,  # Pass the controller directly
    speed="normal",
    enemymode="static",
    level=2,
    visuals=True,
    enemies=[3]
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


# File path to the CSV
csv_file_path = os.path.join(experiment_name, 'all_parents.csv')

# Input the desired generation number to load the best genome
desired_generation = 15

# Load the best genome from the specified generation
best_genome, best_fitness = load_best_genome_from_csv(csv_file_path, desired_generation)

if best_genome is not None:
    # Play the environment using the best genome and display its fitness
    fitness, _, _, _ = env_test.play(best_genome)
    print(f"Tested genome from generation {desired_generation}, achieved fitness: {fitness}")
else:
    print(f"No genome found for generation {desired_generation}")
