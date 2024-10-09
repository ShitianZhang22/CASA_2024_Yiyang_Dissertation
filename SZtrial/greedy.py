from config import *
from fitness import fitness_func

best_layout = [rows * cols - 1 for i in range(num_genes)]
layout = []
used = [best_layout[0]]
best_fit = 0

for i in range(num_genes-1):
    layout = best_layout.copy()
    for j in range(num_genes-1, -1, -1):
        if j not in used:
            layout[i] = j
            fit = fitness_func(None, layout, 0)
            if fit > best_fit:
                best_layout = layout
                best_fit = fit
    used.append(best_layout[i])
    print(best_layout, best_fit)



print(best_layout, best_fit)
