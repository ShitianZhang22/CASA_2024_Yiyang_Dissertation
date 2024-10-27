"""
This file is for testing the validation of 'main' in small cases.
"""

from config import *
from fitness import fitness_func

best_layout = np.zeros(num_genes, dtype=np.int32)
best_fit = 0.0
current = np.zeros(num_genes, dtype=np.int32)
print('layout, fit, best layout, best fit')


def test(layout, depth):
    global best_layout
    global best_fit
    if depth == num_genes:
        depth -= 1
        for i in range(depth, rows * cols):
            layout[depth] = i
            test(layout, depth)
    elif depth > 0:
        depth -= 1
        for i in range(layout[depth+1]):
            layout[depth] = i
            test(layout, depth)
    else:
        fit = fitness_func(None, layout, 0)
        print(layout, fit, best_layout, best_fit)
        if fit > best_fit:
            best_fit = fit
            best_layout = layout.copy()


test(current, num_genes)

print(best_layout)
print(best_fit)
