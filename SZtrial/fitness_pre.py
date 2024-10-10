"""
This is the fitness function.
"""

from config import *

'''
xy position initialisation
from 1-D index to xy position
'''
xy = np.zeros((rows, cols, 2), dtype=np.int32)
for i in range(rows):
    xy[i, :, 1] = i
for i in range(cols):
    xy[:, i, 0] = i
xy = xy.reshape(rows * cols, 2)
xy = xy.transpose()

rotate = np.array([
    [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]],
], dtype=np.int32)

trans_xy = np.zeros((len(theta), 2, rows * cols), dtype=np.float32)
for i in range(len(theta)):
    trans_xy[i] = np.matmul(rotate[i // 2], xy)

rc = max(rows, cols)
data = np.zeros((2, rc, rc), dtype=np.float32)
data[0] = np.loadtxt(r'data/wake0.txt', dtype='float', delimiter=',', encoding='utf-8')
data[1] = np.loadtxt(r'data/wake1.txt', dtype='float', delimiter=',', encoding='utf-8')


def fitness_func(ga_instance, solution, solution_idx):
    fitness = 0  # a specific layout power accumulate
    for ind_t in range(len(theta) // 2):
        pos = trans_xy[ind_t, :, solution].transpose()
        sorted_index = np.argsort(pos[1, :])
        wake_deficiency0 = np.zeros(num_genes, dtype=np.float32)
        wake_deficiency1 = np.zeros(num_genes, dtype=np.float32)
        for j in range(num_genes):
            for k in range(j):
                dx = int(np.abs(pos[0, sorted_index[j]] - pos[0, sorted_index[k]]))
                dy = int(pos[1, sorted_index[j]] - pos[1, sorted_index[k]])
                temp = data[ind_t % 2, dx, dy]
                wake_deficiency0[sorted_index[k]] += temp
                wake_deficiency1[sorted_index[j]] += temp

        actual_velocity = (1 - np.sqrt(wake_deficiency0)) * velocity[ind_t]
        lp_power = layout_power(actual_velocity)  # total power of a specific layout specific wind speed specific theta
        fitness += lp_power.sum() * f_theta_v[ind_t]
        actual_velocity = (1 - np.sqrt(wake_deficiency1)) * velocity[ind_t]
        lp_power = layout_power(actual_velocity)  # total power of a specific layout specific wind speed specific theta
        fitness += lp_power.sum() * f_theta_v[ind_t + 4]
    return fitness


def layout_power(v):
    power = np.zeros(num_genes, dtype=np.float32)
    for j in range(num_genes):
        if 2.0 <= v[j] < 18:
            if v[j] < 12.8:
                power[j] = 0.3 * v[j] ** 3
            else:
                power[j] = 629.1
    return power


if __name__ == '__main__':
    a = fitness_func(None, [3349, 2685, 3663, 896, 2268, 4090, 266, 3303, 1824, 3428, 964, 163, 2391, 1111, 738, 1044, 3098, 2460, 1804, 2833], 0)
    print(a)
