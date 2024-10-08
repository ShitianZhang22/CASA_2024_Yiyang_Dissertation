from config import *


def cal_deficiency(dx, dy):
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


rc = max(rows, cols)

xy = np.zeros((rc, rc, 2), dtype=np.float32)
data1 = np.zeros(rc ** 2, dtype=np.float32)
for i in range(rc):
    xy[i, :, 1] = i
    xy[:, i, 0] = i
xy = xy.reshape(rc ** 2, 2)
for i in range(rc ** 2):
    x, y = xy[i] * cell_width
    d = cal_deficiency(x, y)
    data1[i] = d ** 2

np.savetxt(r'data/wake1.txt', data1, fmt='%f', delimiter=',')
