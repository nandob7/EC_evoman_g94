import numpy as np
import os
import csv
from evoman.environment import Environment
from evoman.controller import Controller
import random
import time

# Parameters
number_of_hidden_neurons = 10
population_size_per_gen = 100
number_of_gen = 30
crossover_chance = 0.7
n_crossover_points = 2
mutation_chance = 0.15
num_elite = 10
experiment_name = 'test_1_100pop_30gen_enemy3'
input_size = 20  # Hardcoded number of sensors
k_members = 3

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    speed="fastest",
    enemymode="static",
    level=2,
    visuals=True,
    enemies=[3]
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
    fitness, player_life, enemy_life, play_time = env.play(genome)
    return fitness, player_life, enemy_life, play_time


def sort_by_fitness(parents_with_fitness):
    """
    Sort the parents based on their fitness values in descending order.
    parents_with_fitness: A list of (genome, fitness) tuples.
    Returns: A sorted list of (genome, fitness) tuples, with the highest fitness first.
    """
    return sorted(parents_with_fitness, key=lambda x: x[1], reverse=True)


def parent_selection(sorted_population_with_fitness, num_elite, k):
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


def save_genomes_to_csv(parents, generation, experiment_name, csv_file_name='all_parents.csv'):
    """
    Save the top parents to a .csv file for a given generation.
        parents: List of (genome, fitness) tuples representing the current parents.
        generation: The current generation number.
        experiment_name: Directory where the CSV file should be saved.
        csv_file_name: Name of the CSV file where the parents will be logged (default is 'best_parents.csv').
    """

    # Path to the CSV file
    csv_file_path = os.path.join(experiment_name, csv_file_name)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Check if the file is empty (add headers if empty)
        if csvfile.tell() == 0:
            csvwriter.writerow(['Generation', 'Genome', 'Fitness'])

        # Write each parent's genome and fitness to the CSV file
        for genome, fitness in parents:
            genome_str = ' '.join(map(str, genome))  # Convert the genome list to a string
            csvwriter.writerow([generation + 1, genome_str, fitness])

    print(f"Saved parents to {csv_file_path}")


def save_all_statistics(generation, population_fitness, player_wins, enemy_wins, player_energy, enemy_energy,
                        play_times, experiment_name):
    """
    Save all relevant statistics (fitness, wins, energy, play time) for a generation into a CSV file.
    """
    # Define the file path
    file_path = os.path.join(experiment_name, "all_statistics.csv")

    # Check if the file exists, if not, initialize with headers
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Highest Fitness", "Mean Fitness", "Std Dev Fitness",
                             "Player Wins", "Enemy Wins", "Mean Player Energy", "Mean Enemy Energy",
                             "Mean Play Time", "Min Play Time", "Max Play Time"])

    # Calculate statistics
    highest_fitness = np.max(population_fitness)
    mean_fitness = np.mean(population_fitness)
    std_dev_fitness = np.std(population_fitness)
    mean_player_energy = np.mean(player_energy)
    mean_enemy_energy = np.mean(enemy_energy)
    mean_play_time = np.mean(play_times)
    min_play_time = np.min(play_times)
    max_play_time = np.max(play_times)

    fitness_stats = [highest_fitness, mean_fitness, std_dev_fitness]
    win_stats = [player_wins, enemy_wins]
    energy_stats = [mean_player_energy, mean_enemy_energy]
    time_stats = [mean_play_time, min_play_time, max_play_time]

    # Combine everything into a single row
    combined_stats = [generation + 1] + fitness_stats + win_stats + energy_stats + time_stats

    # Append data to the CSV
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(combined_stats)

    print(f"Saved statistics for generation {generation + 1} to {file_path}")


def save_experiment_parameters():
    """
    Saves the experiment parameters to a log file.
    """
    log_file_path = os.path.join(experiment_name, 'experiment_log.txt')

    # Create directory if it doesn't exist
    os.makedirs(experiment_name, exist_ok=True)

    # Open the log file and write the parameters
    with open(log_file_path, 'w') as f:
        f.write("Experiment Parameters\n")
        f.write("=====================\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Number of Hidden Neurons: {number_of_hidden_neurons}\n")
        f.write(f"Population Size per Generation: {population_size_per_gen}\n")
        f.write(f"Number of Generations: {number_of_gen}\n")
        f.write(f"Crossover Chance: {crossover_chance}\n")
        f.write(f"Number of Crossover Points: {n_crossover_points}\n")
        f.write(f"Mutation Chance: {mutation_chance}\n")
        f.write(f"Number of Elite Individuals: {num_elite}\n")
        f.write(f"Tournament Selection K-Members: {k_members}\n")
        f.write("=====================\n")

    print(f"Experiment parameters saved to {log_file_path}")


# Record the start time
start_time = time.time()

# Log experiment parameter setup
save_experiment_parameters()

# Initialize the population
population = [create_random_genome(genome_size) for _ in range(population_size_per_gen)]

# Main loop for multiple generations
for generation in range(number_of_gen):
    print(f"Generation {generation + 1}/{number_of_gen}")

    # Record the start time for generation evaluation
    gen_start_time = time.time()

    # Evaluate fitness for each genome
    population_fitness = []
    player_wins = 0
    enemy_wins = 0
    player_energy = []
    enemy_energy = []
    play_times = []
    for i, genome in enumerate(population):
        fitness, player_life, enemy_life, play_time = evaluate_genome(genome)
        population_fitness.append(fitness)
        player_energy.append(player_life)
        enemy_energy.append(enemy_life)
        play_times.append(play_time)

        if player_life < enemy_life:
            enemy_wins += 1
        elif player_life > enemy_life:
            player_wins += 1
        # print(f"Evaluated {i + 1} genomes")

    # # Create a list of tuples (genome, player energy, enemy energy)
    # population_with_energy = list(zip(population, player_energy, enemy_energy))
    # # Sort the tuples based on the fitness (second element of the tuple)
    # sorted_population_with_energy = sort_by_fitness(population_with_energy)

    # Create a list of tuples (genome, fitness)
    population_with_fitness = list(zip(population, population_fitness))
    # Sort the tuples based on the fitness (second element of the tuple)
    sorted_population_with_fitness = sort_by_fitness(population_with_fitness)

    # Save generation statistics
    save_all_statistics(generation, population_fitness, player_wins, enemy_wins, player_energy, enemy_energy,
                        play_times,
                        experiment_name)

    # Select parents for the next generation
    parents = parent_selection(sorted_population_with_fitness, num_elite=num_elite, k=k_members)

    # Calculate selection probabilities
    parents_with_probabilities = calculate_selection_probabilities(parents)

    # Number of children to create (should be the same as the number of parent pairs)
    num_children = len(parents)  # Ensure the number of children matches the number of pairs

    # Sample parents and form pairs
    parent_pairs = sample_parents_for_crossover(parents_with_probabilities, num_children)

    # Apply crossover to each pair and generate offspring
    offspring = []
    for parent1, parent2 in parent_pairs:
        offspring1, offspring2 = crossover((parent1, parent2), n_crossover_points,
                                           crossover_probability=crossover_chance, mutation_rate=mutation_chance)
        offspring.extend([offspring1, offspring2])

    # Ensure the new population size matches the original population size
    population = offspring[:population_size_per_gen]

    # Save the top parents (not offspring) of the current generation to a file
    save_genomes_to_csv(sorted_population_with_fitness, generation, experiment_name)

    # You may want to save or log the best genome of this generation
    best_genome = sorted_population_with_fitness[0][0]
    print(f"Best fitness in generation {generation + 1}: {sorted_population_with_fitness[0][1]}")

    # Record the end time for generation evaluation
    gen_end_time = time.time()

    # Calculate the time it took to evaluate this generation
    gen_time = gen_end_time - gen_start_time
    print(f"Time taken to evaluate generation {generation + 1}: {gen_time:.2f} seconds")

# Calculate and print the total time trained
end_time = time.time()
total_time = end_time - start_time
print(f"Total time trained: {total_time:.2f} seconds")
print("Evolution finished!")
