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
NUM_GENERATIONS = 200
BOUNDS = [-5.12, 5.12]
w = 0.7
c1 = 1.5
c2 = 1.5

# Initialize swarm
def init_swarm():
    pos = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, DIM))
    vel = np.zeros((POP_SIZE, DIM))
    pbest = pos.copy()
    gbest = pos[np.argmin([rastrigin(p) for p in pos])]
    return pos, vel, pbest, gbest

# PSO main loop
def run_pso():
    pos, vel, pbest, gbest = init_swarm()
    pbest_fitness = np.array([rastrigin(p) for p in pbest])
    best_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        for i in range(POP_SIZE):
            r1, r2 = np.random.rand(2)
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest[i] - pos[i])
                + c2 * r2 * (gbest - pos[i])
            )
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], BOUNDS[0], BOUNDS[1])

            fit = rastrigin(pos[i])
            if fit < pbest_fitness[i]:
                pbest[i] = pos[i]
                pbest_fitness[i] = fit

        gbest = pbest[np.argmin(pbest_fitness)]
        best_fitness = rastrigin(gbest)
        best_fitness_over_time.append(best_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.6f}")

    return best_fitness_over_time

if __name__ == "__main__":
    history = run_pso()

    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    with open("data/pso_fitness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best Fitness"])
        for i, val in enumerate(history):
            writer.writerow([i+1, val])

    # Plot and save
    os.makedirs("plots", exist_ok=True)
    plt.plot(history, label="PSO")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("PSO Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/pso_convergence.png")
    plt.show()

