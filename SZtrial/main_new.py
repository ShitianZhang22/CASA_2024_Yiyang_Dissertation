"""
This is a new trial on GA wind farm optimisation using PyGAD library.
Primarily we set the spatial range as 5 rows * 6 cols, or 30 grids.
Expanding from left to right, then from top to bottom
"""

import pygad
from config import *
from fitness import *

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=int,
                       init_range_low=0,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       mutation_by_replacement=mutation_by_replacement,
                       gene_space=gene_space,
                       on_generation=on_generation,
                       allow_duplicate_genes=False,
                       stop_criteria=None,
                       parallel_processing=None,
                       random_seed=None,
                       logger=None,
                       )
ga_instance.run()
ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
