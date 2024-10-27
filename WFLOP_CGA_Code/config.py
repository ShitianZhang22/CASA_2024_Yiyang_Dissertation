import os
import numpy as np
import time

'''
set fixed seed for debug
'''
# random number generator
# rng = np.random.default_rng(seed=int(time.time()))
rng = np.random.default_rng(seed=0)


def reset_random_seed():
    global rng
    rng = np.random.default_rng(seed=int(time.time()))


theta = np.array([0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 5 * np.pi / 4.0, 3 * np.pi / 2.0,
                  7 * np.pi / 4.0], dtype=np.float32)


# parameters for the genetic algorithm
elite_rate = 0.2
select_rate = 0.3
mutate_rate = 0.01

# wind farm size, cells
rows = 58
cols = 73
cell_width = 77.0 * 2  # unit : m

N = 215  # number of wind turbines
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

'''
originally 1000, set 10 for test
'''
# wind distribution
wds_data_folder = "data/wds"
if not os.path.exists(wds_data_folder):
    os.makedirs(wds_data_folder)
# mc : monte-carlo
# n_mc_samples = 1000
n_mc_samples = 10

'''
originally it is 3 times but 1 for test
'''
# n_run_times = 3  # number of genetic run times
n_run_times = 1
