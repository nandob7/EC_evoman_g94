import numpy as np
import os
from evoman.environment import Environment
from evoman.controller import Controller
import random
import time
import itertools

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

def sort_by_fitness(parents_with_fitness):
    """
    Sort the parents based on their fitness values in descending order.
    parents_with_fitness: A list of (genome, fitness) tuples.
    Returns: A sorted list of (genome, fitness) tuples, with the highest fitness first.
    """
    return sorted(parents_with_fitness, key=lambda x: x[1], reverse=True)

def parent_selection(sorted_population_with_fitness, num_elite, k=8):
    """
    This function creates the next generation of genomes.
    - Keeps the top `num_elite` individuals (elitism).
    - Uses `k-member tournament selection` to fill the remaining spots.
    """
    # Step 1: Select top `num_elite` individuals (elitism)
    elite = sorted_population_with_fitness[:num_elite]

    # Step 2: Use tournament selection to fill the remaining population
    tournament_offspring = []
    num_tournament_offspring = population_size_per_gen - num_elite
    for _ in range(num_tournament_offspring):
        winner = k_member_tournament(sorted_population_with_fitness, k)
        tournament_offspring.append(winner)

    # Combine elite individuals and tournament-selected offspring
    parents = elite + tournament_offspring
    return parents

def k_member_tournament(sorted_fitness_tuple, k=8):
    """
    Perform k-member tournament selection and return the selected individual (genome, fitness).
    """
    # Randomly select k individuals from the population
    tournament_contestants = random.sample(sorted_fitness_tuple, k)
    # Sort the selected individuals by fitness (descending order) to choose the best one
    winner = sorted(tournament_contestants, key=lambda x: x[1], reverse=True)[0]
    return winner


def crossover(parents, N=2, crossover_probability=0.7, mutation_rate=0.2):
    """
    Perform N-point crossover between two parents with a certain probability.
    If no crossover happens, the parents are passed directly as offspring.

    parents: A list or tuple of two genomes (numpy arrays).
    N: The number of crossover points.
    crossover_probability: The probability of performing crossover.
    mutation_rate: Probability of mutating each gene in the offspring.
    """
    # Unpack the parents
    parent1, parent2 = parents
    # Ensure both parents have the same length genome
    assert len(parent1) == len(parent2), "Parents must have the same genome size"

    genome_length = len(parent1)

    # Step 1: Check if crossover should happen based on the crossover_probability
    if np.random.rand() < crossover_probability:
        # Perform N-point crossover
        crossover_points = np.sort(np.random.choice(range(1, genome_length), N, replace=False))

        # Start with copies of the parents
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)

        # Alternate segments between parents at the crossover points
        for i in range(len(crossover_points)):
            if i % 2 == 0:
                # Swap the segments between parents
                offspring1[crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None] = \
                    parent2[crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None]
                offspring2[crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None] = \
                    parent1[crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None]
    else:
        # No crossover: Offspring are just clones of the parents
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)

    # Step 2: Apply mutation to both offspring
    offspring1 = mutate(offspring1, mutation_rate)
    offspring2 = mutate(offspring2, mutation_rate)

    return offspring1, offspring2

def mutate(genome, mutation_rate, sigma=0.5, mutation_percentage=0.1, mutation_step=0.05):
    """
    Perform mutation on the genome.
    A subset of the genome is either scrambled, mutated using Gaussian mutation,
    or mutated using box mutation. The mutation function is randomly chosen from a list.

    mutation_rate: Probability of selecting a portion of the genome to mutate.
    sigma: Standard deviation for the normal distribution used in the mutation.
    mutation_percentage: The fraction of the genome to undergo mutation.
    mutation_step: Maximum step size (range) for box mutation.
    """
    def scramble_mutation():
        # Select two random points in the genome to define the scramble range
        start_idx = np.random.randint(0, genome_length)
        end_idx = np.random.randint(start_idx, genome_length)
        # Scramble (shuffle) the selected range of the genome
        subset_to_scramble = genome[start_idx:end_idx]
        np.random.shuffle(subset_to_scramble)
        # Replace the selected range with the shuffled version
        genome[start_idx:end_idx] = subset_to_scramble

    def gaussian_mutation():
        num_mutations = int(mutation_percentage * genome_length)
        indices_to_mutate = np.random.choice(genome_length, num_mutations, replace=False)
        for idx in indices_to_mutate:
            random_value = np.random.normal(0, sigma)
            genome[idx] += random_value

    def box_mutation():
        # Apply box mutation (small perturbation)
        num_mutations = int(mutation_percentage * genome_length)
        indices_to_mutate = np.random.choice(genome_length, num_mutations, replace=False)
        for idx in indices_to_mutate:
            # Generate a small random change within [-mutation_step, mutation_step]
            random_step = np.random.uniform(-mutation_step, mutation_step)
            genome[idx] += random_step
    genome_length = len(genome)

    application_list = [scramble_mutation, gaussian_mutation, box_mutation]

    if np.random.rand() < mutation_rate:
        # Randomly pick a mutation method from the list and apply it
        chosen_mutation = random.choice(application_list)
        chosen_mutation()  # Apply the selected mutation function
    return genome


def calculate_selection_probabilities(sorted_parents_with_fitness):
    """
    Calculate the probability of each parent being chosen based on their fitness.
    Probability is calculated as (fitness of parent - min_fitness + epsilon) / total_adjusted_fitness,
    where epsilon is a small constant to ensure all probabilities are positive.
    sorted_parents_with_fitness: A sorted list of (genome, fitness) tuples.
    Returns: A list of tuples (genome, fitness, probability) with probabilities based on adjusted fitness.
    """
    # Extract fitness values
    fitness_values = np.array([fitness for _, fitness in sorted_parents_with_fitness])

    # Shift fitness values to be non-negative
    min_fitness = fitness_values.min()
    epsilon = 1e-8  # Small constant to prevent zero probabilities
    adjusted_fitness_values = fitness_values - min_fitness + epsilon

    # Calculate probabilities
    total_adjusted_fitness = adjusted_fitness_values.sum()
    probabilities = adjusted_fitness_values / total_adjusted_fitness

    # Combine the probabilities with the tuples
    parents_with_probabilities = [
        (genome, fitness, prob) for (genome, fitness), prob in zip(sorted_parents_with_fitness, probabilities)
    ]

    return parents_with_probabilities

def sample_parents_for_crossover(parents_with_probabilities, num_children):
    """
    Sample parents based on their selection probabilities to form couples for crossover.
    parents_with_probabilities: A list of tuples (genome, fitness, probability).
    num_children: Number of children (offspring) to create.
    Returns: A list of tuples, each containing two sampled parent genomes for crossover.
    """
    # Extract genomes and probabilities
    genomes = [genome for genome, _, _ in parents_with_probabilities]
    probabilities = [prob for _, _, prob in parents_with_probabilities]

    # Convert genomes to indices
    num_parents = len(genomes)
    indices = np.arange(num_parents)

    # Sample indices based on their probabilities
    sampled_indices = np.random.choice(indices, size=(num_children * 2), p=probabilities, replace=True)

    # Retrieve the sampled genomes
    sampled_genomes = [genomes[idx] for idx in sampled_indices]

    # Pair the sampled genomes
    parent_pairs = [(sampled_genomes[i], sampled_genomes[i + 1]) for i in range(0, len(sampled_genomes), 2)]

    return parent_pairs


def save_fitness_statistics(generation, population_fitness, experiment_name):
    """
    Save the highest fitness, mean fitness, and standard deviation of fitness for a generation.
    generation: The current generation number.
    population_fitness: A list of fitness values for the population.
    experiment_name: The directory where the file should be saved.
    """
    # Calculate statistics
    highest_fitness = np.max(population_fitness)
    mean_fitness = np.mean(population_fitness)
    std_dev_fitness = np.std(population_fitness)

    # Define file path
    stats_file_path = os.path.join(experiment_name, f"fitness_stats_generation_{generation + 1}.txt")

    # Save statistics to the file
    with open(stats_file_path, 'w') as f:
        f.write(f"Generation {generation + 1}\n")
        f.write(f"Highest fitness: {highest_fitness}\n")
        f.write(f"Mean fitness: {mean_fitness}\n")
        f.write(f"Standard deviation: {std_dev_fitness}\n")

    print(f"Saved fitness statistics to {stats_file_path}")


def save_genomes_to_file(parents, generation, experiment_name):
    """
    Save the top parents to a .txt file for a given generation.
    parents: The current parents (list of (genome, fitness) tuples).
    generation: The current generation number.
    experiment_name: The directory where the file should be saved.
    """
    file_path = os.path.join(experiment_name, f"parents_generation_{generation + 1}.txt")
    with open(file_path, 'w') as f:
        for genome, fitness in parents:
            genome_str = ' '.join(map(str, genome))
            f.write(f"{genome_str} {fitness}\n")  # Include fitness value in the file for reference
    print(f"Saved parents to {file_path}")

# Record the start time
start_time = time.time()

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Define parameter ranges
hidden_neuron_configs = [10, 20]
population_configs = [50, 100]
generation_configs = [50, 100]
mutation_chance_configs = [0.05, 0.1, 0.2]
input_size = 20  # Hardcoded number of sensors

# Initialize results dictionary to store fitness results
results = {}

# Grid search loop
for num_hidden_neurons, population_size_per_gen, num_generations, mutation_chance in itertools.product(
        hidden_neuron_configs, population_configs, generation_configs, mutation_chance_configs):

    # Set experiment name for each config
    experiment_name = f"test_n{num_hidden_neurons}_p{population_size_per_gen}_g{num_generations}_m{mutation_chance}_e3"

    # Create experiment directory if it doesn't exist
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize a neural controller with the current number of hidden neurons
    neural_controller = Controller(input_size, num_hidden_neurons)

    # Initialize the environment with the current configuration
    env = Environment(
        experiment_name=experiment_name,
        playermode="ai",
        player_controller=neural_controller,
        speed="fastest",
        enemymode="static",
        level=2,
        visuals=True,
        enemies=[3]
    )

    # Calculate genome size for the given controller
    genome_size = neural_controller.genome_size()

    # Initialize the population
    population = [create_random_genome(genome_size) for _ in range(population_size_per_gen)]

    # Main loop for multiple generations
    for generation in range(num_generations):
        print(f"Experiment: {experiment_name}, Generation {generation + 1}/{num_generations}")

        # Evaluate fitness for each genome
        population_fitness = []
        for i, genome in enumerate(population):
            fitness = evaluate_genome(genome)
            population_fitness.append(fitness)

        # Create a list of tuples (genome, fitness)
        population_with_fitness = list(zip(population, population_fitness))

        # Sort the tuples based on the fitness (second element of the tuple)
        sorted_population_with_fitness = sort_by_fitness(population_with_fitness)

        # Save fitness statistics for this generation
        save_fitness_statistics(generation, population_fitness, experiment_name)

        # Select parents for the next generation
        parents = parent_selection(sorted_population_with_fitness, num_elite=10, k=8)

        # Calculate selection probabilities
        parents_with_probabilities = calculate_selection_probabilities(parents)

        # Sample parents and form pairs
        parent_pairs = sample_parents_for_crossover(parents_with_probabilities, num_children=len(parents))

        # Apply crossover to each pair and generate offspring
        offspring = []
        for parent1, parent2 in parent_pairs:
            offspring1, offspring2 = crossover(
                (parent1, parent2), crossover_probability=0.7, mutation_rate=mutation_chance
            )
            offspring.extend([offspring1, offspring2])

        # Ensure the new population size matches the original population size
        population = offspring[:population_size_per_gen]

        # Save the top parents (not offspring) of the current generation to a file
        save_genomes_to_file(sorted_population_with_fitness, generation, experiment_name)

        # Log both the best fitness and the best genome for this generation
        best_fitness = sorted_population_with_fitness[0][1]
        best_genome = sorted_population_with_fitness[0][0]

        print(f"Best fitness in generation {generation + 1}: {best_fitness}")

        # Save both fitness and genome for later use
        if experiment_name not in results:
            results[experiment_name] = {'fitness': [], 'genomes': []}
        results[experiment_name]['fitness'].append(best_fitness)
        results[experiment_name]['genomes'].append(best_genome)

    # End of experiment for this configuration
    print(f"Finished experiment {experiment_name}")

# After all experiments, you can save or print the results
print("Grid search results:", results)