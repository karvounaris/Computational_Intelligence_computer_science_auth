import numpy as np
import matplotlib.pyplot as plt
import os
import heapq

def genetic_algorithm(pop_size, chromo_length, generations, crossover_rate, mutation_rate):
    # Initializes the population with randomly generated binary chromosomes
    def initialize_population(size, length):
        return np.random.randint(2, size=(size, length))

    # Fitness function to evaluate each chromosome
    def fitness(chromosome):
        return np.sum(chromosome)  # Summing up all 1's in the chromosome

    # Selects chromosomes to form the next generation based on their fitness proportionate probabilities
    def roulette_selection(population, fitnesses):
        fitnesses = np.array(fitnesses, dtype=np.float64)
        total_fitness = np.sum(fitnesses)
        selection_probs = fitnesses / total_fitness
        selected_indices = np.random.choice(population.shape[0], size=population.shape[0], replace=True, p=selection_probs)
        return population[selected_indices]

    # Perform single point crossover between two parents
    def single_point_crossover(parent1, parent2, P_CROSSOVER):
        if np.random.rand() < P_CROSSOVER:
            crossover_point = np.random.randint(1, chromo_length)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    # Mutates chromosomes by flipping bits with a given probability
    def mutation(chromosome, rate):
        for i in range(len(chromosome)):
            if np.random.rand() < rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def find_top_4_values_and_indices(data):
        # Create a min-heap with up to 4 elements, storing both value and index
        min_heap = []
        for index, value in enumerate(data):
            if len(min_heap) < 4:
                heapq.heappush(min_heap, (value, index))
            elif value > min_heap[0][0]:  # Only push if the value is larger than the smallest in the heap
                heapq.heapreplace(min_heap, (value, index))

        # Sort the heap to get the top 4 values in descending order
        top_values = sorted(min_heap, reverse=True, key=lambda x: x[0])
        values = [v[0] for v in top_values]
        indices = [v[1] for v in top_values]

        return values, indices

    population = initialize_population(pop_size, chromo_length)
    best_fitness_over_time = []
    average_fitness_over_time = []

    # Main loop of the genetic algorithm
    for generation in range(generations):
        fitnesses = np.array([fitness(chromosome) for chromosome in population])
        best_fitness_over_time.append(np.max(fitnesses))
        average_fitness_over_time.append(np.mean(fitnesses))

        new_population = []

        values, indices = find_top_4_values_and_indices(fitnesses)
        parent1 = population[indices[0]]
        parent2 = population[indices[1]]
        child1, child2 = single_point_crossover(parent1, parent2, crossover_rate)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2])

        parent1 = population[indices[2]]
        parent2 = population[indices[3]]
        child1, child2 = single_point_crossover(parent1, parent2, crossover_rate)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2])

        for _ in range((pop_size // 2) - 2):
            parent1 = roulette_selection(population, fitnesses)[0]
            parent2 = roulette_selection(population, fitnesses)[0]
            child1, child2 = single_point_crossover(parent1, parent2, crossover_rate)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = np.array(new_population)  # Ensure the new population is a numpy array

    return best_fitness_over_time, average_fitness_over_time

# Parameters setup for the genetic algorithm
POP_SIZES = [50, 100, 150, 200, 300]
CROSSOVER_RATES = [0, 0.3, 0.5, 0.7, 0.9]
MUTATION_RATES = [0.0005, 0.001, 0.005, 0.01]
CHROMO_LENGTH = 100
GENERATIONS = 100
OUTPUT_DIR = 'plots_genetikos_9_2'

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Running the genetic algorithm with different parameter settings and saving the results
for pop_size in POP_SIZES:
    for crossover_rate in CROSSOVER_RATES:
        for mutation_rate in MUTATION_RATES:
            best_fitness, avg_fitness = genetic_algorithm(pop_size, CHROMO_LENGTH, GENERATIONS, crossover_rate, mutation_rate)
            plt.figure(figsize=(10, 5))
            plt.plot(best_fitness, label='Best Fitness')
            plt.plot(avg_fitness, label='Average Fitness')
            title = f'Pop size: {pop_size}, Crossover rate: {crossover_rate}, Mutation rate: {mutation_rate}'
            plt.title(title)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()

            # Save the plot with a descriptive filename
            filename = f"{title}.png".replace(", ", "_").replace(": ", "-").replace(".", "")
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(filepath)
            plt.close()
