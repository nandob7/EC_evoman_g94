import numpy as np
import pandas as pd

from evoman.environment import Environment
from demo_controller import player_controller

# Parameters
number_of_hidden_neurons = 10
input_size = 20
random_trained = False
csv_path = f'runs/generalist/best_genomes.csv'
random_start = False
eas = [1, 2]
groups = ['all_enemies', '2358']

best_genomes = pd.read_csv(csv_path)

# Initialize the columns with empty lists for each row
best_genomes['gain'] = [[] for _ in range(len(best_genomes))]

for group in groups:
    for ea in eas:
        relevant_genomes = best_genomes[(best_genomes['ea'] == ea) & (best_genomes['group'] == group)]
        for _, genome in relevant_genomes.iterrows():
            experiment_name = f'test_ea{ea}_{group}_r{int(genome["run"])}'

            for enemy in range(1, 9):
                # Initialize the environment with the hardcoded controller
                env_test = Environment(
                    experiment_name=experiment_name,
                    playermode="ai",
                    player_controller=player_controller(number_of_hidden_neurons),  # Pass the controller directly
                    enemymode="static",
                    level=2,
                    randomini='yes' if random_start else 'no',
                    savelogs='no',
                    # speed='normal',
                    # visuals=True,
                    enemies=[enemy]
                )

                # Play the environment using the best genome and display its fitness
                fitness, player_life, enemy_life, play_time = env_test.play(
                    np.fromstring(genome['genome'], sep=' '))

                print(f'EA {ea}, enemy {enemy}, run {int(genome["run"])}: {"Player" if enemy_life == 0 else "Enemy"} won, Gain: {player_life - enemy_life}')

                # Append gain to the corresponding lists in the DataFrame
                genome['gain'].append((player_life - enemy_life))

output_csv_path = f'runs/generalist/tested_best_genomes.csv'
best_genomes.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
