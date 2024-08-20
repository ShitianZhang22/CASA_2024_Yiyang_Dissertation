import numpy as np
import pandas as pd
import MARS  # MARS (Multivariate Adaptive Regression Splines) regression class
import WindFarmGeneticToolbox  # wind farm layout optimization using genetic algorithms classes
from datetime import datetime
import os
import pickle

# parameters for the genetic algorithm
elite_rate = 0.2
cross_rate = 0.8
random_rate = 0.3
mutate_rate = 0.01

# wind farm size, cells
rows = 58
cols = 73
cell_width = 77.0 * 2 # unit : m

#
N = 215  # number of wind turbines 
pop_size = 200  # population size, number of inidividuals in a population
iteration = 30  # number of genetic algorithm iterations

# all data will be save in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# create an object of WindFarmGenetic
wfg = WindFarmGeneticToolbox.WindFarmGenetic(rows=rows, cols=cols, N=N, pop_size=pop_size,
                                             iteration=iteration, cell_width=cell_width, elite_rate=elite_rate,
                                             cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)
# set wind distribution
# wind distribution is discrete (number of wind speeds) by (number of wind directions)
#wfg.init_1_direction_1_SSW_WindFinder_Max()
#wfg.init_4_direction_1_WeatherSpark_Max()

#wfg.init_8_direction_1_WeatherSpark_Max()
wfg.init_8_direction_1_WeatherSpark_Min()

################################################
# generate initial populations
################################################

init_pops_data_folder = "data/init_pops"
if not os.path.exists(init_pops_data_folder):
    os.makedirs(init_pops_data_folder)
# n_init_pops : number of initial populations
n_init_pops = 60
for i in range(n_init_pops):
    wfg.gen_init_pop()
    wfg.save_init_pop("{}/init_{}.dat".format(init_pops_data_folder,i))


#############################################
# generate wind distribution surface
#############################################
wds_data_folder = "data/wds"
if not os.path.exists(wds_data_folder):
    os.makedirs(wds_data_folder)
# mc : monte-carlo
n_mc_samples = 1000

# each layout is binary list and the length of the list is (rows*cols)
# 1 indicates there is a wind turbine in that cell
# 0 indicates there is no wind turbine in the cell
# in "mc_layout.dat", there are 'n_mc_samples' line and each line is a layout.

# generate 'n_mc_samples' layouts and save it in 'mc_layout.data' file
WindFarmGeneticToolbox.LayoutGridMCGenerator.gen_mc_grid(rows=rows, cols=cols, n=n_mc_samples, N=N,
                                                         lofname="{}/{}".format(wds_data_folder, "mc_layout.dat"))
# read layouts from 'mc_layout.dat' file
layouts = np.genfromtxt("{}/{}".format(wds_data_folder,"mc_layout.dat"), delimiter="  ", dtype=np.int32)



# generate dataset to build wind farm distribution surface
wfg.mc_gen_xy(rows=rows, cols=cols, layouts=layouts, n=n_mc_samples, N=N, xfname="{}/{}".format(wds_data_folder, "x.dat"),
              yfname="{}/{}".format(wds_data_folder, "y.dat"))

# parameters for MARS regression method
n_variables = 2
n_points = rows * cols
n_candidate_knots = [rows, cols]
n_max_basis_functions = 100
n_max_interactions = 4
difference = 1.0e-3

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

n_run_times = 3  # number of run times
# result_arr stores the best conversion efficiency of each run
result_arr = np.zeros((n_run_times, 2), dtype=np.float32)

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


#======== New trial on existing layout=======#

#original_layout = 

#wfg.load_init_pop("Turbines/turbine_0.dat")



# 读取 .dat 文件内容并转换为 numpy 数组
#dat_file_path = 'Turbines/turbine_0.dat'
#existing_layout = np.loadtxt(dat_file_path)

# 创建 WindFarmGenetic 类的实例
#wind_farm_original = WindFarmGenetic(rows=dat_content.shape[0], cols=dat_content.shape[1], N=215, pop_size=1, iteration=iteration, cell_width=cell_width, elite_rate=elite_rate,
#                                             cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)
#wind_farm_original.init_4_direction_1_WeatherSpark_Max()

# 将读取的数据赋值给实例的 init_pop 属性
#wind_farm_original.init_pop = dat_content
#wind_farm_original.init_pop_nonezero_indices = np.nonzero(dat_content)

# 调用计算总功率率的方法
#P_rate_total = wind_farm_original.cal_P_rate_total()
#velocity = np.array([6.56], dtype=np.float32)
#layout_power = wind_farm_original.layout_power(velocity, N)
#power_order = np.zeros((pop_size, N),dtype=np.int32)
#fitness_original = wind_farm_original.conventional_fitness(dat_content, rows=dat_content.shape[0], cols=dat_content.shape[1],
#                                                           pop_size=1, N=N, po=power_order)

# 打印结果
#print(f'The calculated eta value is: {eta}')


"""
import numpy as np
from WindFarmGeneticToolbox import WindFarmGenetic
import matplotlib.pyplot as plt

dat_file_path = 'Turbines/turbine_0.dat'
existing_layout = np.loadtxt(dat_file_path)

wind_farm_original = WindFarmGeneticToolbox.WindFarmGenetic(rows=rows, cols=cols, N=215, pop_size=1, iteration=iteration, cell_width=cell_width, elite_rate=elite_rate,
                                             cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)

#wind_farm_original.init_4_direction_1_WeatherSpark_Max()
#wind_farm_original.init_1_direction_1_SSW_WindFinder_Max()
wind_farm_original.init_8_direction_1_WeatherSpark_Max()
wind_farm_original.init_8_direction_1_WeatherSpark_Min()

# 调用 calculate_eta 函数，并分别存储 eta, fitness_value[0], 和 P_rate_total
eta_original, fitness_value_original, P_rate_total_original = wind_farm_original.calculate_eta(existing_layout)

# 现在你可以分别使用这三个变量
print(f"Eta: {eta_original}")
print(f"Fitness Value: {fitness_value_original}")
print(f"P Rate Total: {P_rate_total_original}")

"""