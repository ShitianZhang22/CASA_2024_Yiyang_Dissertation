"""
This is the fitness function.
"""

from config import *

'''
xy position initialisation
from 1-D index to xy position
'''
xy = np.zeros((rows, cols, 2), dtype=np.float32)
for i in range(rows):
    xy[i, :, 1] = i
for i in range(cols):
    xy[:, i, 0] = i
xy = xy.reshape(rows * cols, 2)
xy = xy * cell_width + cell_width / 2
xy = xy.transpose()

trans_matrix = np.zeros((len(theta), 2, 2), dtype=np.float32)
trans_xy = np.zeros((len(theta), 2, rows * cols), dtype=np.float32)
for i in range(len(theta)):
    trans_matrix[i] = np.array(
        [[np.cos(theta[i]), -np.sin(theta[i])],
         [np.sin(theta[i]), np.cos(theta[i])]],
        dtype=np.float32)
    trans_xy[i] = np.matmul(trans_matrix[i], xy)


def fitness_func(ga_instance, solution, solution_idx):
    xy_position = xy[:, solution]
    fitness = 0  # a specific layout power accumulate
    for ind_t in range(len(theta)):
        # need an extra transpose. the indices will auto trans once
        trans_xy_position = trans_xy[ind_t, :, solution].transpose()

        speed_deficiency = wake(trans_xy_position)

        actual_velocity = (1 - speed_deficiency) * velocity[ind_t]
        lp_power = layout_power(actual_velocity)  # total power of a specific layout specific wind speed specific theta
        fitness += lp_power.sum() * f_theta_v[ind_t]
    return fitness


def wake(trans_xy_position):
    # y value increasing, making theta correspondent to the wind direction
    # to do: we can directly sort genes
    sorted_index = np.argsort(trans_xy_position[1, :])
    wake_deficiency = np.zeros(num_genes, dtype=np.float32)
    for i in range(num_genes):
        for j in range(i):
            dx = np.absolute(trans_xy_position[0, sorted_index[i]] - trans_xy_position[0, sorted_index[j]])
            dy = np.absolute(trans_xy_position[1, sorted_index[i]] - trans_xy_position[1, sorted_index[j]])
            d = cal_deficiency(dx=dx, dy=dy)
            wake_deficiency[sorted_index[i]] += d ** 2
    return np.sqrt(wake_deficiency)


def cal_deficiency(dx, dy):
    r_wake = rotor_radius + entrainment_const * dy
    if dx >= rotor_radius + r_wake:
        intersection = 0
    elif dx > r_wake - rotor_radius:
        alpha = np.arccos((r_wake ** 2 + dx ** 2 - rotor_radius ** 2) / (2 * r_wake * dx))
        beta = np.arccos((rotor_radius ** 2 + dx ** 2 - r_wake ** 2) / (2 * rotor_radius * dx))
        intersection = alpha * r_wake ** 2 + beta * rotor_radius ** 2 - r_wake * dx * np.sin(alpha)
    else:
        intersection = np.pi * rotor_radius ** 2
    return 2.0 / 3.0 * intersection / (np.pi * r_wake ** 2)


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
    a = fitness_func(None, [9, 1, 27, 16, 2, 14, 23, 21, 28, 17], 0)
    print(a)