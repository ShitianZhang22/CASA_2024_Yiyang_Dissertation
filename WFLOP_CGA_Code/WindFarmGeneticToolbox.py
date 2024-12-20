import math
from datetime import datetime
from config import *

'''
version: 1.0.0
'''


# # reset_random_seed()
class WindFarmGenetic:

    # constructor of class WindFarmGenetic
    def __init__(self):
        self.turbine = GE_1_5_sleTurbine()
        self.cell_width_half = cell_width * 0.5

        self.init_pop = None
        self.init_pop_nonezero_indices = None

    '''
    the following is wind data called in initialisation
    max and min are Jan and July
    not make sense with uniform wind speed
    f_theta_v is proportion of wind in 8 directions
    '''
    def init_8_direction_1_WeatherSpark_Max(self):
        self.velocity = np.array([6.67], dtype=np.float32)  # 1
        self.f_theta_v = np.array([[0.14], [0.06], [0.2], [0.09], [0.22], [0.08], [0.16], [0.05]], dtype=np.float32)
    
    def init_8_direction_1_WeatherSpark_Min(self):
        self.velocity = np.array([4.5], dtype=np.float32)  # 1
        self.f_theta_v = np.array([[0.2], [0.04], [0.15], [0.06], [0.2], [0.1], [0.18], [0.07]], dtype=np.float32)

    '''
    generate the 1st gen of layouts (via gen_pop method)
    then update init_pop_nonezero_indices, with each row a layout.
    Non-zero coordinates (turbines) are recorded within. (process can be realised via numpy built-in methods)
    '''
    def gen_init_pop(self):
        for i in range(n_init_pops):
            self.init_pop = LayoutGridMCGenerator.gen_pop(rows=rows, cols=cols, n=pop_size, N=N)
            self.init_pop_nonezero_indices = np.zeros((pop_size, N), dtype=np.int32)
            for ind_init_pop in range(pop_size):
                ind_indices = 0
                for ind in range(rows * cols):
                    if self.init_pop[ind_init_pop, ind] == 1:
                        self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                        ind_indices += 1
            np.savetxt("{}/init_{}.dat".format(init_pops_data_folder,i), self.init_pop, fmt='%d', delimiter="  ")

    def load_init_pop(self, fname):
        self.init_pop = np.genfromtxt(fname, delimiter="  ", dtype=np.int32)
        self.init_pop_nonezero_indices = np.zeros((pop_size, N), dtype=np.int32)
        for ind_init_pop in range(pop_size):
            ind_indices = 0
            for ind in range(rows * cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1

    '''
    calculate denominator of formula 1, or ideal power generation
    since only one wind speed is used here, the proportions of directions are also useless
    '''
    def cal_P_rate_total(self):
        f_p = 0.0
        for ind_t in range(len(theta)):
            for ind_v in range(len(self.velocity)):
                f_p += self.f_theta_v[ind_t, ind_v] * self.turbine.P_i_X(self.velocity[ind_v])
        return N * f_p

    def layout_power(self, velocity, N):
        power = np.zeros(N, dtype=np.float32)
        for i in range(N):
            power[i] = self.turbine.P_i_X(velocity[i])
        return power

    '''
    initialising some storage space and no calculation?
    '''
    # generate dataset to build the wind distribution surface
    def mc_gen_xy(self, rows, cols, layouts, n, N, xfname, yfname):
        layouts_cr = np.zeros((rows * cols, 2), dtype=np.int32)  # layouts column row index
        n_copies = np.sum(layouts, axis=0)
        layouts_power = np.zeros((n, rows * cols), dtype=np.float32)
        self.mc_fitness(pop=layouts, rows=rows, cols=cols, pop_size=n, N=N, lp=layouts_power)
        sum_layout_power = np.sum(layouts_power, axis=0)
        mean_power = np.zeros(rows * cols, dtype=np.float32)
        for i in range(rows * cols):
            mean_power[i] = sum_layout_power[i] / n_copies[i]
        # print(n_copies)
        # print(sum_layout_power)
        # print(mean_power)
        # print(n_copies)
        '''
        1-D coordinate to 2-D and stored in layouts_cr (2 cols)
        just cross ref again, can get directly
        '''
        for ind in range(rows * cols):
            r_i = np.floor(ind / cols)
            c_i = np.floor(ind - r_i * cols)
            layouts_cr[ind, 0] = c_i
            layouts_cr[ind, 1] = r_i
        np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        np.savetxt(yfname, mean_power, fmt='%f', delimiter="  ")

    '''
    fitness evaluation for each layouts
    iterate with rows at the outer for loop
    '''
    def mc_fitness(self, pop, rows, cols, pop_size, N, lp):
        for i in range(pop_size):
            '''
            regenerate the centroid of grids as the coordinates of turbines and stored in 2 rows * N cols
            '''
            print("layout {}...".format(i))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(theta)):
                '''
                fix wind direction and rotate the layout (7 times) for different direction of winds
                '''
                for ind_v in range(len(self.velocity)):
                    trans_matrix = np.array(
                        [[np.cos(theta[ind_t]), -np.sin(theta[ind_t])],
                         [np.sin(theta[ind_t]), np.cos(theta[ind_t])]],
                        np.float32)
                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    '''
                    wake calculation
                    check later
                    '''
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                 N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            lp[i, ind_position] = lp_power_accum

    '''
    this and the next are to calculate wakes with formula in the paper
    '''
    def wake_calculate(self, trans_xy_position, N):
        # print(-trans_xy_position)
        sorted_index = np.argsort(-trans_xy_position[1, :])  # y value descending
        wake_deficiency = np.zeros(N, dtype=np.float32)
        # print(1-wake_deficiency)
        wake_deficiency[sorted_index[0]] = 0
        for i in range(1, N):
            for j in range(i):
                xdis = np.absolute(trans_xy_position[0, sorted_index[i]] - trans_xy_position[0, sorted_index[j]])
                ydis = np.absolute(trans_xy_position[1, sorted_index[i]] - trans_xy_position[1, sorted_index[j]])
                d = self.cal_deficiency(dx=xdis, dy=ydis, r=self.turbine.rator_radius,
                                        ec=self.turbine.entrainment_const)
                wake_deficiency[sorted_index[i]] += d ** 2

            wake_deficiency[sorted_index[i]] = np.sqrt(wake_deficiency[sorted_index[i]])
            # print(trans_xy_position[0, sorted_index[i]])
            # print(trans_xy_position[0, sorted_index[j]])
            # print(xdis)
        # print(trans_xy_position)
        # print(v)
        return wake_deficiency
    

    # ec : entrainment_const
    def cal_deficiency(self, dx, dy, r, ec):
        if dy == 0:
            return 0
        R = r + ec * dy
        inter_area = self.cal_interaction_area(dx=dx, dy=dy, r=r, R=R)
        d = 2.0 / 3.0 * (r ** 2) / (R ** 2) * inter_area / (np.pi * r ** 2)
        return d

    def cal_interaction_area(self, dx, dy, r, R):
        if dx >= r + R:
            return 0
        elif dx >= np.sqrt(R ** 2 - r ** 2):
            alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
            beta = np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
            A1 = alpha * R ** 2
            A2 = beta * r ** 2
            A3 = R * dx * np.sin(alpha)
            return A1 + A2 - A3
        elif dx >= R - r:
            alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
            beta = np.pi - np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
            A1 = alpha * R ** 2
            A2 = beta * r ** 2
            A3 = R * dx * np.sin(alpha)
            return np.pi * r ** 2 - (A2 + A3 - A1)
        else:
            return np.pi * r ** 2

    '''
    genetic method
    '''
    def conventional_genetic_alg(self, ind_time, result_folder):  # conventional genetic algorithm
        """
        name of variants or methods and contents:
        fitness_value: calculate the power generated by each turbine in each layout, sorted (stored in indices) from
        the lowest to the highest
        power_order: reverse the above and store the value; index in sorted_index
        """
        P_rate_total = self.cal_P_rate_total()
        '''
        the above is formula 1 for ideal power generation
        '''
        start_time = datetime.now()
        print("conventional genetic algorithm starts....")
        fitness_generations = np.zeros(iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((iteration, rows * cols),
                                           dtype=np.int32)  # best layout in each generation
        power_order = np.zeros((pop_size, N),
                               dtype=np.int32)  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(pop_size * elite_rate))  # elite number
        rN = int(int(np.floor(pop_size * mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = pop_size - eN - mN  # crossover number

        # 新增数组来存储每代最优个体的eta值历史
        best_eta_history = np.zeros(iteration, dtype=np.float32)
        '''
        power for each turbine in a layout is stored in fitness_value
        '''

        for gen in range(iteration):
            print("generation {}...".format(gen))
            fitness_value = self.conventional_fitness(pop=pop, rows=rows, cols=cols, pop_size=pop_size,
                                                      N=N,
                                                      po=power_order)
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]
            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]

            # 记录每代最优个体的eta值
            best_eta_history[gen] = fitness_value[sorted_index[0]] * (1.0 / P_rate_total)

            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
            '''
            select, crossover and mutate
            '''
            n_parents, parent_layouts, parent_pop_indices = self.conventional_select(pop=pop, pop_indices=pop_indices,
                                                                                     pop_size=pop_size,
                                                                                     elite_rate=elite_rate,
                                                                                     select_rate=select_rate)
            self.conventional_crossover(N=N, pop=pop, pop_indices=pop_indices, pop_size=pop_size,
                                        n_parents=n_parents,
                                        parent_layouts=parent_layouts, parent_pop_indices=parent_pop_indices)
            self.conventional_mutation(rows=rows, cols=cols, N=N, pop=pop, pop_indices=pop_indices,
                                       pop_size=pop_size,
                                       mutation_rate=mutate_rate)
        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        eta_generations = np.copy(fitness_generations)
        eta_generations = eta_generations * (1.0 / P_rate_total)
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = "{}/conventional_eta_N{}_{}_{}.dat".format(result_folder, N, ind_time, time_stamp)
        np.savetxt(filename, eta_generations, fmt='%f', delimiter="  ")
        filename = "{}/conventional_best_layouts_N{}_{}_{}.dat".format(result_folder, N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_generations, fmt='%d', delimiter="  ")
        print("conventional genetic algorithm ends.")
        filename = "{}/conventional_runtime.txt".format(result_folder)  # time used to run the method in seconds
        f = open(filename, "a+")
        f.write("{}\n".format(run_time))
        f.close()

        filename = "{}/conventional_eta.txt".format(result_folder)  # all best etas
        f = open(filename, "a+")
        f.write("{}\n".format(eta_generations[iteration - 1]))
        f.close()

        # 保存每代最优个体的eta值历史
        filename = "{}/best_eta_history_N{}_{}_{}.dat".format(result_folder, N, ind_time, time_stamp)
        np.savetxt(filename, best_eta_history, fmt='%f', delimiter="  ")

        return run_time, eta_generations[iteration - 1]
    '''
    mutation
    move a random turbine to a vacant position
    '''
    def conventional_mutation(self, rows, cols, N, pop, pop_indices, pop_size, mutation_rate):
        # reset_random_seed()
        for i in range(pop_size):
            if rng.uniform() > mutation_rate:
                continue
            while True:
                turbine_pos = rng.integers(0, cols * rows)
                if pop[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = rng.integers(0, cols * rows)
                if pop[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1
            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])

    '''
    crossover
    randomly select two parents and use 1st to xth turbines of one parent and x+1th to Nth turbines of another
    problem: no duplicate check
    '''
    def conventional_crossover(self, N, pop, pop_indices, pop_size, n_parents,
                               parent_layouts, parent_pop_indices):
        n_counter = 0
        # reset_random_seed()  # init random seed
        while n_counter < pop_size:
            male = rng.integers(0, n_parents)
            female = rng.integers(0, n_parents)
            if male != female:
                cross_point = rng.integers(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]
                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1

    def conventional_select(self, pop, pop_indices, pop_size, elite_rate, select_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        # reset_random_seed()  # init random seed
        for i in range(n_elite, pop_size):
            if rng.uniform() < select_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_pop_indices

    '''
    no input?
    the following method is nearly the same with mc_fitness with only differences in last several rows
    lp_power_accum stores power of each turbine in each layout
    then the total power of each layout is stored in the returned fitness_val
    po stores the indices of turbines sorted by power low to high, but no return?
    '''
    def conventional_fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            '''
            above coordinate transfer again
            '''
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(theta)):
                for ind_v in range(len(self.velocity)):
                    # print(theta[ind_t])
                    # print(np.cos(theta[ind_t]))
                    trans_matrix = np.array(
                        [[np.cos(theta[ind_t]), -np.sin(theta[ind_t])],
                         [np.sin(theta[ind_t]), np.cos(theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)

                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                 N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]

            fitness_val[i] = np.sum(lp_power_accum)
        return fitness_val


'''
turbine class
also surface roughness
'''
class GE_1_5_sleTurbine:
    hub_height = 80.0  # unit (m)
    rator_diameter = 77.0  # unit m
    surface_roughness = 0.25 * 0.001  # unit mm surface roughness
    # surface_roughness = 0.25  # unit mm surface roughness
    rator_radius = 0

    entrainment_const = 0

    def __init__(self):
        self.rator_radius = self.rator_diameter / 2
        '''
        the following is alpha in formula 4
        '''
        self.entrainment_const = 0.5 / np.log(self.hub_height / self.surface_roughness)

    '''
    power generation according to wind speed
    formula 7
    '''
    # power curve
    def P_i_X(self, v):
        if v < 2.0:
            return 0
        elif v < 12.8:
            return 0.3 * v ** 3
        elif v < 18:
            return 629.1
        else:
            return 0


class LayoutGridMCGenerator:
    def __init__(self):
        return

    # rows : number of rows in wind farm
    # cols : number of columns in wind farm
    # n : number of layouts
    # N : number of turbines
    # lofname : layouts file name
    '''
    another layout set again?
    '''

    @staticmethod
    def gen_mc_grid(rows, cols, n, N, lofname):  # generate monte carlo wind farm layout grids
        # reset_random_seed()  # init random seed
        layouts = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout
        # layouts_cr = np.zeros((n*, 2), dtype=np.float32)  # layouts column row index
        positionX = rng.integers(0, cols, size=(N * n * 2))
        positionY = rng.integers(0, rows, size=(N * n * 2))
        ind_rows = 0  # index of layouts from 0 to n-1
        ind_pos = 0  # index of positionX, positionY from 0 to N*n*2-1
        # ind_crs = 0
        while ind_rows < n:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
            if np.sum(layouts[ind_rows, :]) == N:
                # for ind in range(rows * cols):
                #     if layouts[ind_rows, ind] == 1:
                #         r_i = np.floor(ind / cols)
                #         c_i = np.floor(ind - r_i * cols)
                #         layouts_cr[ind_crs, 0] = c_i
                #         layouts_cr[ind_crs, 1] = r_i
                #         ind_crs += 1
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        # filename = "positions{}by{}by{}N{}.dat".format(rows, cols, n, N)
        np.savetxt(lofname, layouts, fmt='%d', delimiter="  ")
        # np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        return layouts

    '''
    generate random layout (may duplicate) according to grids and number of turbines
    returns a set of layouts with each row as a layout (1-D)
    '''
    # generate population
    def gen_pop(rows, cols, n,
                N):  # generate population very similar to gen_mc_grid, just without saving layouts to a file
        # reset_random_seed()
        layouts = np.zeros((n, rows * cols), dtype=np.int32)
        positionX = rng.integers(0, cols, size=(N * n * 2))
        positionY = rng.integers(0, rows, size=(N * n * 2))
        ind_rows = 0
        ind_pos = 0

        while ind_rows < n:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
            if np.sum(layouts[ind_rows, :]) == N:
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        return layouts
