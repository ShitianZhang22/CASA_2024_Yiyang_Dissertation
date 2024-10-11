"""
This is the precomputation file.
If there is no data/wake0.txt or ata/wake1.txt, please run this file first.
"""

from config import *


def cal_deficiency(dx, dy):
    """
    This is to calculate the wake effect deficiency factor.
    :param dx: distance perpendicular to the wind direction
    :param dy: distance along the wind direction
    :return: the wake effect deficiency factor of a turbine from another
    """
    r_wake = rotor_radius + entrainment_const * dy
    if dx >= rotor_radius + r_wake or dy == 0:
        intersection = 0
    elif dx > r_wake - rotor_radius:
        alpha = np.arccos((r_wake ** 2 + dx ** 2 - rotor_radius ** 2) / (2 * r_wake * dx))
        beta = np.arccos((rotor_radius ** 2 + dx ** 2 - r_wake ** 2) / (2 * rotor_radius * dx))
        intersection = alpha * r_wake ** 2 + beta * rotor_radius ** 2 - r_wake * dx * np.sin(alpha)
    else:
        intersection = np.pi * rotor_radius ** 2
    return 2.0 / 3.0 * intersection / (np.pi * r_wake ** 2)


if __name__ == '__main__':
    rc = max(rows, cols)
    xy = np.zeros((rc, rc, 2), dtype=np.float32)
    data = np.zeros((2, rc ** 2), dtype=np.float32)
    for i in range(rc):
        xy[i, :, 1] = i
        xy[:, i, 0] = i
    xy = xy.reshape(rc ** 2, 2)
    xy *= cell_width
    xy = xy.transpose()
    for i in range(2):
        trans_matrix = np.array(
            [[np.cos(theta[i]), -np.sin(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i])]],
            dtype=np.float32)
        trans_xy = np.matmul(trans_matrix, xy)
        for j in range(rc ** 2):
            d = cal_deficiency(np.abs(trans_xy[0, j]), np.abs(trans_xy[1, j]))
            data[i, j] = d ** 2  # notice that we have already squared
    data = data.reshape((2, rc, rc))
    data[0] = data[0].transpose()
    data[1] = data[1].transpose()
    np.savetxt(r'data/wake0.txt', data[0], fmt='%f', delimiter=',', encoding='utf-8')
    np.savetxt(r'data/wake1.txt', data[1], fmt='%f', delimiter=',', encoding='utf-8')
