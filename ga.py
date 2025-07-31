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
NUM_GENERATIONS = 200
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
BOUNDS = [-5.12, 5.12]

# Initialize population
def init_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, 2))

# Selection
def select(pop, fitness):
    idx = np.random.randint(0, POP_SIZE, 2)
    return pop[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else pop[idx[1]]

# Crossover
def crossover(p1, p2):
    if np.random.rand() < CROSSOVER_RATE:
        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2
    return p1.copy()

# Mutation
def mutate(ind):
    if np.random.rand() < MUTATION_RATE:
        ind += np.random.normal(0, 0.5, size=2)
        ind = np.clip(ind, BOUNDS[0], BOUNDS[1])
    return ind

# GA main loop
def run_ga():
    population = init_population()
    best_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        fitness = np.array([rastrigin(ind) for ind in population])
        new_population = []

        for _ in range(POP_SIZE):
            p1 = select(population, fitness)
            p2 = select(population, fitness)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = np.array(new_population)
        best_fitness = np.min([rastrigin(ind) for ind in population])
        best_fitness_over_time.append(best_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.6f}")

    return best_fitness_over_time

if __name__ == "__main__":
    history = run_ga()

    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    with open("data/ga_fitness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best Fitness"])
        for i, val in enumerate(history):
            writer.writerow([i+1, val])

    # Plot and save
    os.makedirs("plots", exist_ok=True)
    plt.plot(history, label="GA")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ga_convergence.png")
    plt.show()

