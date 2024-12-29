import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import heapq


# Constants for the Genetic Algorithm
POP_SIZE = 100  # Population size
GENOME_LENGTH = 100  # Length of the chromosome
CROSSOVER_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GENERATION_LIMIT = 1e4  # Making sure it's an integer

# Fitness function to evaluate each chromosome
def fitness(chromosome):
    return np.sum(chromosome)  # Summing up all 1's in the chromosome

# Initialize population with random chromosomes
def initialize_population():
    return np.random.randint(2, size=(POP_SIZE, GENOME_LENGTH))

def find_top_4_values_and_indices(data):
    # Create a min-heap with up to 5 elements, storing both value and index
    min_heap = []
    for index, value in enumerate(data):
        if len(min_heap) < 4:
            heapq.heappush(min_heap, (value, index))
        elif value > min_heap[0][0]:  # Only push if the value is larger than the smallest in the heap
            heapq.heapreplace(min_heap, (value, index))

    # Sort the heap to get the top 5 values in descending order
    # Sorting on the first item of each tuple (the value) and extracting the values and indices
    top_values = sorted(min_heap, reverse=True, key=lambda x: x[0])
    values = [v[0] for v in top_values]
    indices = [v[1] for v in top_values]

    return values, indices

# Select parent for reproduction based on fitness proportionate selection (Roulette wheel selection)
def roulette_wheel_selection(population, fitnesses):
    total_fitness = np.sum(fitnesses)
    selection_probs = fitnesses / total_fitness
    selected_index = np.random.choice(np.arange(POP_SIZE), p=selection_probs)
    return population[selected_index]

# Perform single point crossover between two parents
def single_point_crossover(parent1, parent2, P_CROSSOVER):
    if np.random.rand() < P_CROSSOVER:
        crossover_point = np.random.randint(1, GENOME_LENGTH)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

# Mutation function with dynamic mutation rate based on fuzzy logic
def mutation(chromosome, fitness_score):
    # Determine mutation rate based on fitness score
    if fitness_score < 70:
        P_MUTATION = 0.01  # High mutation rate for low fitness
    elif fitness_score < 85:
        P_MUTATION = 0.005  # Medium mutation rate for medium fitness
    elif fitness_score < 95:
        P_MUTATION = 0.001  # Medium mutation rate for medium fitness    
    else:
        P_MUTATION = 0.0005  # Low mutation rate for high fitness

    # P_MUTATION = 0.001

    for i in range(GENOME_LENGTH):
        if np.random.rand() < P_MUTATION:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit
    return chromosome

# Main genetic algorithm function
def genetic_algorithm(P_CROSSOVER):
    population = initialize_population()
    best_fitness = 0
    generation = 0 

    while best_fitness < GENOME_LENGTH and generation < GENERATION_LIMIT:
        fitnesses = np.empty(len(population))
        for i, individual in enumerate(population):
            fitnesses[i] = fitness(individual)
        new_population = []
        
        values, indices = find_top_4_values_and_indices(fitnesses)
        parent1 = population[indices[0]]
        parent2 = population[indices[1]]
        child1, child2 = single_point_crossover(parent1, parent2, P_CROSSOVER)
        child1_fitness = fitness(child1)
        child2_fitness = fitness(child2)
        child1 = mutation(child1, child1_fitness)
        child2 = mutation(child2, child2_fitness)
        new_population.extend([child1, child2])

        parent1 = population[indices[2]]
        parent2 = population[indices[3]]
        child1, child2 = single_point_crossover(parent1, parent2, P_CROSSOVER)
        child1_fitness = fitness(child1)
        child2_fitness = fitness(child2)
        child1 = mutation(child1, child1_fitness)
        child2 = mutation(child2, child2_fitness)
        new_population.extend([child1, child2])

        for _ in range((POP_SIZE // 2 ) - 2):
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)
            child1, child2 = single_point_crossover(parent1, parent2, P_CROSSOVER)
            child1_fitness = fitness(child1)
            child2_fitness = fitness(child2)
            child1 = mutation(child1, child1_fitness)
            child2 = mutation(child2, child2_fitness)
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
        current_best = np.max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
        
        print(f"Generation is: {generation}, best_fitness is: {best_fitness} and P_CROSSOVER is: {P_CROSSOVER}")
        generation += 1
        
    return generation

# Experiment with different P_CROSSOVER values
results = {p: [] for p in CROSSOVER_VALUES}
for p in CROSSOVER_VALUES:
    for i in range(20):
        gen_count = genetic_algorithm(p)
        results[p].append(gen_count)
    print(f"Results for P_CROSSOVER = {p}: {results[p]}")

# Setup plot folder path and ensure directory exists
plot_folder = os.path.expanduser('~/Documents/university/deep_reinforcement_learning/genetikos/plots_genetikos_9_1_fuzzy_logic')
os.makedirs(plot_folder, exist_ok=True)

# Plot results for different P_CROSSOVER values
plt.figure(figsize=(12, 8))
for p, gens in results.items():
    plt.plot(range(1, 21), gens, marker='o', linestyle='-', label=f'P_CROSSOVER = {p}')
plt.xlabel('Run')
plt.ylabel('Number of Generations')
plt.title('Number of Generations to Find Optimal Solution by P_CROSSOVER')
plt.legend()
plt.grid(True)
plt_path = os.path.join(plot_folder, 'Number of Generations to Find Optimal Solution by P_CROSSOVER.png')
plt.savefig(plt_path)

# Calculate means and plot the average number of generations for different P_CROSSOVER values
means = {p: np.mean(gens) for p, gens in results.items()}
colors = list(mcolors.TABLEAU_COLORS.values())
plt.figure(figsize=(10, 6))
bar_container = plt.bar(means.keys(), means.values(), color=colors[:len(means)], width=0.1)
plt.xlabel('P_CROSSOVER Value')
plt.ylabel('Average Number of Generations')
plt.title('Average Number of Generations to Find Optimal Solution by P_CROSSOVER')
plt.xticks(list(means.keys()))
plt.grid(True, linestyle='--', alpha=0.6)
for bar in bar_container:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontweight='bold')
plt_path = os.path.join(plot_folder, 'Average Number of Generations to Find Optimal Solution by P_CROSSOVER.png')
plt.savefig(plt_path)






