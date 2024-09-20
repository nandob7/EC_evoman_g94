import numpy as np
import os
from evoman.environment import Environment
from evoman.controller import Controller
import random

# Parameters
number_of_hidden_neurons = 10
population_size_per_gen = 100
number_of_gen = 30
mutation_chance = 0.2
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
    enemies=[2]
)

def load_genomes_from_file(file_path):
    """
    Load genomes from a .txt file.
    file_path: Path to the file containing saved genomes.
    Returns: A list of genomes (each genome is a numpy array).
    """
    genomes = []
    with open(file_path, 'r') as f:
        for line in f:
            genome = np.fromstring(line.strip(), sep=' ')
            genomes.append(genome)
    return genomes


file_path = 'test_1_100pop_30gen_enemy3/parents_generation_15.txt'
genomes=load_genomes_from_file(file_path)

best_genome = genomes[0]

fitness, _, _, _ = env_test.play(best_genome)
print(fitness)