import numpy as np
import os
import pandas as pd

from evoman.environment import Environment
from evoman.controller import Controller

# Parameters
number_of_hidden_neurons = 10
csv_path = f'runs/best_genomes.csv'
input_size = 20  # Hardcoded number of sensors
random_start = True

# Initialize a neural controller
neural_controller = Controller(input_size, number_of_hidden_neurons)

best_genomes = pd.read_csv(csv_path)

# Initialize the columns with empty lists for each row
best_genomes['fitness'] = [[] for _ in range(len(best_genomes))]
best_genomes['gain'] = [[] for _ in range(len(best_genomes))]


def calc_cust_fitness(player_life, enemy_life, time):
    return (0.8 * (100 - enemy_life)) + (
        0.4 * player_life if player_life > 75 else 0.2 * player_life if player_life > 50 else 0.1 * player_life) - np.log(
        time)


for enemy in [2, 3, 5]:
    for ea in [1, 2]:
        current_genome = best_genomes[(best_genomes['ea'] == ea) & (best_genomes['enemy'] == enemy)]
        experiment_name = f'test_4_ea{ea}_e{enemy}'

        for i in range(5):
            # Initialize the environment with the hardcoded controller
            env_test = Environment(
                experiment_name=experiment_name,
                playermode="ai",
                player_controller=neural_controller,  # Pass the controller directly
                enemymode="static",
                level=2,
                randomini='yes' if random_start else 'no',
                savelogs='no',
                speed='normal',
                visuals=True,
                enemies=[enemy]
            )

            # Play the environment using the best genome and display its fitness
            fitness, player_life, enemy_life, play_time = env_test.play(np.fromstring(current_genome['genome'].iloc[0], sep=' '))

            # Append the values to the corresponding lists in the DataFrame
            current_genome['fitness'].item().append(fitness if ea == 1 else calc_cust_fitness(player_life, enemy_life, play_time))
            current_genome['gain'].item().append(player_life - enemy_life)

output_csv_path = f'runs/best_genomes_{"random" if random_start else "fixed"}.csv'
best_genomes.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
