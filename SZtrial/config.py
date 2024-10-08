import os
import numpy as np

num_generations = 10

sol_per_pop = 20
select_rate = 0.3
num_parents_mating = int(sol_per_pop * select_rate)

num_genes = 215  # number of wind turbines

# wind farm size, cells
rows = 58
cols = 73

cell_width = 77.0 * 2  # unit : m
init_range_high = rows * cols - 1

parent_selection_type = 'rank'
keep_parents = -1

elite_rate = 0.2
keep_elitism = int(sol_per_pop * elite_rate)

crossover_type = 'single_point'
# crossover_type = None
crossover_probability = 0.3  # this is the selection rate for crossover

mutation_type = 'random'
# mutation_type = None
mutation_probability = 0.01  # this is the mutation rate but applies to gene
mutation_by_replacement = True

gene_space = range(rows * cols)  # this should be manually set later

parallel_processing = None
# parallel_processing = ['process', 10]

'''
================================================================
'''


theta = np.array([0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 5 * np.pi / 4.0, 3 * np.pi / 2.0,
                  7 * np.pi / 4.0], dtype=np.float32)
velocity = np.array([4.5] * 8, dtype=np.float32)


'''
originally 200 and 30, set 10 and 3 for test
'''
# pop_size = 200  # population size, number of individuals in a population
# iteration = 30  # number of genetic algorithm iterations
pop_size = 10
iteration = 3

# all data will be saved in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

init_pops_data_folder = "data/init_pops"
if not os.path.exists(init_pops_data_folder):
    os.makedirs(init_pops_data_folder)
'''
originally 60, set 10 for test
'''
# n_init_pops : number of initial populations
# n_init_pops = 60
n_init_pops = 10

hub_height = 80.0  # unit (m)
surface_roughness = 0.25 * 0.001
entrainment_const = 0.5 / np.log(hub_height / surface_roughness)
rotor_radius = 77.0 / 2
f_theta_v = np.array([0.2, 0.04, 0.15, 0.06, 0.2, 0.1, 0.18, 0.07], dtype=np.float32)
