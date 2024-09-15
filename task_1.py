import numpy as np
import os
from evoman.environment import Environment
from evoman.controller import Controller

# Parameters
number_of_hidden_neurons = 10
population_size_per_gen = 100
number_of_gen = 50
mutation_chance = 0.2
experiment_name = 'controller_task1_test'
input_size = 20  # Hardcoded number of sensors

# Create experiment directory if it doesn't exist
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize a neural controller
neural_controller = Controller(input_size, number_of_hidden_neurons)

# Initialize the environment with the hardcoded controller
env = Environment(
    experiment_name=experiment_name,
    playermode="ai",
    player_controller=neural_controller,  # Pass the controller directly
    speed="normal",
    enemymode="static",
    level=2,
    visuals=True
)

# Calculate genome size for the given controller
genome_size = neural_controller.genome_size()


# Function to create a random genome (weights)
def create_random_genome(genome_size):
    return np.random.uniform(-1, 1, genome_size)


# Function to evaluate the fitness of a controller with a given genome
def evaluate_genome(genome):
    # Set the genome for the neural controller
    neural_controller.set(genome, input_size)

    # Play the environment using this controller and get the fitness score
    fitness, _, _, _ = env.play(genome)

    return fitness


# Create the initial population (a list of random genomes)
population = [create_random_genome(genome_size) for _ in range(population_size_per_gen)]


#TO-DO:
#Calculate the fitness of each instance in a population
#Rank the genomes
#choose parents
#children
#mutation



