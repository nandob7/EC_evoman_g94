import numpy as np
from evoman.environment import Environment
import random
from demo_controller import player_controller
import optuna

# Parameters
number_of_hidden_neurons = 10
input_size = 20  # Hardcoded number of sensors
population_size_per_gen = 100
number_of_gen = 15
enemies = [1, 2, 3, 4, 5, 6, 7, 8]

custom_fitness = False
random_start = False

params_ea1 = {
    "crossover_chance": 0.7,
    "n_crossover_points": 2,
    "mutation_chance": 0.01,
    "num_elite": 20,
    "k_members": 3,
    "custom_fitness": False,
    "random_start": False,
    "n_ea": 1
}

params_ea2 = {
    "crossover_chance": 0.8,
    "n_crossover_points": 5,
    "mutation_chance": 0.2,
    "num_elite": 10,
    "k_members": 5,
    "custom_fitness": True,
    "random_start": False,
    "n_ea": 2
}


def create_random_genome(size):
    """
    Generate a random genome
    param size: size of the genome
    Returns: A np array randomly generated genome
    """
    return np.random.uniform(-1, 1, size)


# Function to define and calculate custom fitness function
def calc_cust_fitness(player_life, enemy_life, play_time):
    """
    Calculates the custom fitness
    player_life: The energy level of the player
    enemy_life: The energy level of the enemy
    play_time: The play time surpassed
    Returns: A float value of the computed custom fitness function
    """
    return (0.8 * (100 - enemy_life)) + (0.4 * player_life if player_life > 75 else 0.2 * player_life if player_life > 50 else 0.1 * player_life) - np.log(play_time)


def evaluate_genome(genome):
    """
    Evaluates the genome, playing it in the environment and retrieving statistics
    genome: A np array of floats representing the genome
    Returns: The fitness value and statistics of the evaluated genome.
    """
    # Set the genome for the neural controller
    # neural_controller.set(genome, input_size)
    # Play the environment using this controller and get the fitness score
    fitness, player_life, enemy_life, play_time = env.play(genome)

    if play_time <= 0:
        play_time = 1e-8

    if custom_fitness:
        fitness = calc_cust_fitness(player_life, enemy_life, play_time)
    return fitness, player_life, enemy_life, play_time


def sort_by_fitness(parents_with_fitness):
    """
    Sort the parents based on their fitness values in descending order.
    parents_with_fitness: A list of (genome, fitness) tuples.
    Returns: A sorted list of (genome, fitness) tuples, with the highest fitness first.
    """
    return sorted(parents_with_fitness, key=lambda x: x[1], reverse=True)


def parent_selection(population, num_elite, k):
    """
    This function creates the next generation of genomes.
    - Keeps the top `num_elite` individuals (elitism).
    - Uses `k-member tournament selection` to fill the remaining spots.
    """
    # Step 1: Select top `num_elite` individuals (elitism)
    elite = population[:num_elite]

    # Step 2: Use tournament selection to fill the remaining population
    tournament_offspring = []
    num_tournament_offspring = population_size_per_gen - num_elite
    for _ in range(num_tournament_offspring):
        winner = k_member_tournament(population, k)
        tournament_offspring.append(winner)

    # Combine elite individuals and tournament-selected offspring
    parents = elite + tournament_offspring
    return parents


def k_member_tournament(sorted_fitness_tuple, k):
    """
    Perform k-member tournament selection and return the selected individual (genome, fitness).
    """
    # Randomly select k individuals from the population
    tournament_contestants = random.sample(sorted_fitness_tuple, k)
    fittest_probability = 1
    if np.random.rand() < fittest_probability:
        # Sort the selected individuals by fitness (descending order) to choose the best one
        winner = sorted(tournament_contestants, key=lambda x: x[1], reverse=True)[0]
    else:
        winner = random.choice(tournament_contestants)

    return winner


def crossover(parents, N, crossover_probability, mutation_rate):
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
                offspring1[
                crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None] = \
                    parent2[
                    crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None]
                offspring2[
                crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None] = \
                    parent1[
                    crossover_points[i]:crossover_points[i + 1] if i + 1 < len(crossover_points) else None]
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
    # application_list = [box_mutation]

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
        (genome, fitness, prob) for (genome, fitness), prob in
        zip(sorted_parents_with_fitness, probabilities)
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


def objective(trial):
    crossover_chance = trial.suggest_float("crossover_chance", 0, 1)
    mutation_chance = trial.suggest_float("mutation_chance", 0, 0.5)
    num_elite = trial.suggest_int("num_elite", 0, 30)
    k_members = trial.suggest_int("k_members", 2, 10)
    n_crossover_points = trial.suggest_int("n_crossover_points", 1, 5)

    env = Environment(
        experiment_name=f'optuna_trial_{trial.number}',  # Unique name per trial
        playermode="ai",
        player_controller=player_controller(number_of_hidden_neurons),
        enemymode="static",
        level=2,
        randomini='no',
        savelogs='no',
        visuals=False,
        enemies=enemies,
        multiplemode='yes'
    )

    # Initialize the population

    population = [create_random_genome(265) for _ in range(population_size_per_gen)]

    best_genome = None
    best_fitness = float('-inf')  # Initialize best fitness as the lowest possible value

    # Evolutionary loop (similar to your original)
    for generation in range(number_of_gen):
        # Evaluate all genomes in the population
        population_fitness = []
        for genome in population:
            fitness, player_life, enemy_life, play_time = evaluate_genome(genome)
            population_fitness.append(fitness)

        # Sort by fitness and get the best one
        population_with_fitness = list(zip(population, population_fitness))
        sorted_population_with_fitness = sort_by_fitness(population_with_fitness)

        # Track the best fitness of this generation
        if sorted_population_with_fitness[0][1] > best_fitness:
            best_fitness = sorted_population_with_fitness[0][1]
            best_genome = sorted_population_with_fitness[0][0]

        # Perform the selection, crossover, and mutation to form the next generation
        parents = parent_selection(sorted_population_with_fitness, num_elite=num_elite, k=k_members)
        parents_with_probabilities = calculate_selection_probabilities(parents)

        parent_pairs = sample_parents_for_crossover(parents_with_probabilities, len(parents))
        offspring = []
        for parent1, parent2 in parent_pairs:
            offspring1, offspring2 = crossover((parent1, parent2), n_crossover_points, crossover_chance,
                                               mutation_chance)
            offspring.extend([offspring1, offspring2])

        population = offspring[:population_size_per_gen]  # Keep the population size consistent

    # Return the best fitness score found in this trial
    return best_fitness

env = Environment(
    experiment_name=f'optuna_trial',
    playermode="ai",
    player_controller=player_controller(10),  # Pass the controller directly
    enemymode="static",
    level=2,
    randomini='no',
    savelogs='no',
    # speed='normal',
    visuals=False, #disabling this massively speeds up training
    enemies=[1,2,3,4,5,6,7,8],
    multiplemode='yes'
)

# Define the study and optimize the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# You can print or save the best found parameters
print(f"Best parameters: {study.best_params}")
print(f"Best fitness: {study.best_value}")