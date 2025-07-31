import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Rastrigin function
def rastrigin(position):
    x, y = position
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# Parameters
POP_SIZE = 50
DIM = 2
F = 0.8
CR = 0.9
NUM_GENERATIONS = 200
BOUNDS = [-5.12, 5.12]

# Initialize population
def init_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, DIM))

# DE main loop
def run_de():
    population = init_population()
    best_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        new_population = []

        for i in range(POP_SIZE):
            indices = list(range(0, i)) + list(range(i+1, POP_SIZE))
            r1, r2, r3 = population[np.random.choice(indices, 3, replace=False)]
            mutant = r1 + F * (r2 - r3)
            mutant = np.clip(mutant, BOUNDS[0], BOUNDS[1])

            trial = np.copy(population[i])
            j_rand = np.random.randint(DIM)
            for j in range(DIM):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            if rastrigin(trial) < rastrigin(population[i]):
                new_population.append(trial)
            else:
                new_population.append(population[i])

        population = np.array(new_population)
        best_fitness = np.min([rastrigin(ind) for ind in population])
        best_fitness_over_time.append(best_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.6f}")

    return best_fitness_over_time

if __name__ == "__main__":
    history = run_de()

    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    with open("data/de_fitness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best Fitness"])
        for i, val in enumerate(history):
            writer.writerow([i+1, val])

    # Plot and save
    os.makedirs("plots", exist_ok=True)
    plt.plot(history, label="DE")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("DE Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/de_convergence.png")
    plt.show()

