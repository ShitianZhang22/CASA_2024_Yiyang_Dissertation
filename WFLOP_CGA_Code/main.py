import numpy as np
import pandas as pd
import MARS  # MARS (Multivariate Adaptive Regression Splines) regression class
import WindFarmGeneticToolbox  # wind farm layout optimization using genetic algorithms classes
from datetime import datetime
import os
import pickle
from config import *

# create an object of WindFarmGenetic
wfg = WindFarmGeneticToolbox.WindFarmGenetic()

# set wind distribution
# wind distribution is discrete (number of wind speeds) by (number of wind directions)

'''
max and min are two data sets (Jan and July)
'''
# wfg.init_8_direction_1_WeatherSpark_Max()
wfg.init_8_direction_1_WeatherSpark_Min()

'''
the first gen of layouts in init_pop
'''

################################################
# generate initial populations
################################################

for i in range(n_init_pops):
    wfg.gen_init_pop()
    wfg.save_init_pop("{}/init_{}.dat".format(init_pops_data_folder,i))

#############################################
# generate wind distribution surface
#############################################

# each layout is binary list and the length of the list is (rows*cols)
# 1 indicates there is a wind turbine in that cell
# 0 indicates there is no wind turbine in the cell
# in "mc_layout.dat", there are 'n_mc_samples' line and each line is a layout.

'''
create random layouts again, and then save and load?
Monte-Carlo method here seems not be used later, just for cross referencing series
'''

# generate 'n_mc_samples' layouts and save it in 'mc_layout.data' file
WindFarmGeneticToolbox.LayoutGridMCGenerator.gen_mc_grid(rows=rows, cols=cols, n=n_mc_samples, N=N,
                                                         lofname="{}/{}".format(wds_data_folder, "mc_layout.dat"))
# read layouts from 'mc_layout.dat' file
layouts = np.genfromtxt("{}/{}".format(wds_data_folder,"mc_layout.dat"), delimiter="  ", dtype=np.int32)

# generate dataset to build wind farm distribution surface
wfg.mc_gen_xy(rows=rows, cols=cols, layouts=layouts, n=n_mc_samples, N=N, xfname="{}/{}".format(wds_data_folder, "x.dat"),
              yfname="{}/{}".format(wds_data_folder, "y.dat"))

'''
the following is MARS, check later
'''
# parameters for MARS regression method
n_variables = 2
n_points = rows * cols
n_candidate_knots = [rows, cols]
n_max_basis_functions = 100
n_max_interactions = 4
difference = 1.0e-3

'''
x and y here are just cross ref
'''
x_original = pd.read_csv("{}/{}".format(wds_data_folder,"x.dat"), header=None, nrows=n_points, delim_whitespace=True)
x_original = x_original.values

y_original = pd.read_csv("{}/{}".format(wds_data_folder,"y.dat"), header=None, nrows=n_points, delim_whitespace=True)
y_original = y_original.values

mars = MARS.MARS(n_variables=n_variables, n_points=n_points, x=x_original, y=y_original,
                 n_candidate_knots=n_candidate_knots, n_max_basis_functions=n_max_basis_functions,
                 n_max_interactions=n_max_interactions, difference=difference)
mars.MARS_regress()
# save wind distribution model to 'wds.mars'
mars.save_mars_model_to_file()
with open("{}/{}".format(wds_data_folder,"wds.mars"), "wb") as mars_file:
    pickle.dump(mars, mars_file)

# results folder
# adaptive_best_layouts_N60_9_20190422213718.dat : best layout for AGA of run index 9
# result_CGA_20190422213715.dat : run time and best eta for CGA method
results_data_folder = "data/results"
if not os.path.exists(results_data_folder):
    os.makedirs(results_data_folder)

# result_arr stores the best conversion efficiency of each run
result_arr = np.zeros((n_run_times, 2), dtype=np.float32)

'''
genetic method, not related to Monte-Carlo
'''
# CGA method
CGA_results_data_folder = "{}/CGA".format(results_data_folder)
if not os.path.exists(CGA_results_data_folder):
    os.makedirs(CGA_results_data_folder)
for i in range(0, n_run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop("{}/init_{}.dat".format(init_pops_data_folder, i))
    run_time, eta = wfg.conventional_genetic_alg(ind_time=i, result_folder=CGA_results_data_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_CGA_{}.dat".format(CGA_results_data_folder, time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
