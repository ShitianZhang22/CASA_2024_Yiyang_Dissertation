import os

# parameters for the genetic algorithm
elite_rate = 0.2
cross_rate = 0.8
random_rate = 0.3
mutate_rate = 0.01

# wind farm size, cells
rows = 58
cols = 73
cell_width = 77.0 * 2  # unit : m

N = 215  # number of wind turbines
pop_size = 200  # population size, number of individuals in a population
iteration = 30  # number of genetic algorithm iterations

# all data will be saved in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

init_pops_data_folder = "data/init_pops"
if not os.path.exists(init_pops_data_folder):
    os.makedirs(init_pops_data_folder)
# n_init_pops : number of initial populations
n_init_pops = 60

# wind distribution
wds_data_folder = "data/wds"
if not os.path.exists(wds_data_folder):
    os.makedirs(wds_data_folder)
# mc : monte-carlo
n_mc_samples = 1000
