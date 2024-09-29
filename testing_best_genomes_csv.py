import numpy as np
import pandas as pd

from evoman.environment import Environment
from evoman.controller import Controller

# Parameters
number_of_hidden_neurons = 10
input_size = 20  # Hardcoded number of sensors
random_trained = False
csv_path = f'runs/{"random/" if random_trained else ""}best_genomes.csv'
random_start = False
eas = [1, 2]
enemies = [2, 3, 5]

# Initialize a neural controller
neural_controller = Controller(input_size, number_of_hidden_neurons)

best_genomes = pd.read_csv(csv_path)

# Initialize the columns with empty lists for each row
best_genomes['gain'] = [[] for _ in range(len(best_genomes))]

for enemy in enemies:
    for ea in eas:
        relevant_genomes = best_genomes[(best_genomes['ea'] == ea) & (best_genomes['enemy'] == enemy)]
        for j, genome in relevant_genomes.iterrows():
            experiment_name = f'test_4_ea{ea}_e{enemy}_r{int(genome["run"])}'

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
                    # speed='normal',
                    # visuals=True,
                    enemies=[enemy]
                )

                # Play the environment using the best genome and display its fitness
                fitness, player_life, enemy_life, play_time = env_test.play(
                    np.fromstring(genome['genome'], sep=' '))

                print(f'EA {ea}, enemy {enemy}, run {int(genome["run"])}, test {i + 1}: {"Player" if player_life > enemy_life else "Enemy"} won')

                # Append gain to the corresponding lists in the DataFrame
                genome['gain'].append(player_life - enemy_life)

output_csv_path = f'runs/{"random/" if random_trained else ""}/best_genomes_{"random" if random_start else "fixed"}.csv'
best_genomes.to_csv(output_csv_path, index=False)

print(f"Best genomes saved to {output_csv_path}")
