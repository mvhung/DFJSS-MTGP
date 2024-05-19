
import math
from deap import base, creator, gp, tools
import numpy as np
import operator
import itertools
import random
import copy

from tools import evaluate,cxOnePoint
# tap function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
def max_operator(left, right):
    return max(left, right)
def min_operator(left, right):
    return min(left, right)
pset = gp.PrimitiveSet("MAIN", 10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(max_operator, 2)
pset.addPrimitive(min_operator, 2)

pset.renameArguments(ARG0='TIS')
pset.renameArguments(ARG1='W')
pset.renameArguments(ARG2='NOR')
pset.renameArguments(ARG3='WKR')
pset.renameArguments(ARG4='rDD')
pset.renameArguments(ARG5='PT')
pset.renameArguments(ARG6='OWT')
pset.renameArguments(ARG7='NPT')
pset.renameArguments(ARG8='NIQ')
pset.renameArguments(ARG9='MIQ')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)  # Tạo cây ngẫu nhiên
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  # Tạo cá thể từ cây
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Tạo quần thể từ các cá thể
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def select_tournament(population,offspring,fitness, k = 4):
    individuals_with_fitness = list(zip(population + offspring, fitness))
    selected = []
    fitness_res = []
    population_size = len(population)
    for i in range(population_size):
        tournament = random.sample(individuals_with_fitness, k)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
        fitness_res.append(winner[1])


    return selected, fitness_res

def roulette_wheel_selection(population,offspring,fitness):
    fitness_sum = sum(evaluate(toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1])) for x in population)  # Tính tổng fitness của tất cả các cá thể trong quần thể
    selection_probs = [evaluate(toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1])) / fitness_sum for x in population]  # Tính xác suất chọn lọc cho mỗi cá thể
    individuals_with_fitness = list(zip(population + offspring, fitness))
    selected_parents = []
    
    for _ in range(len(population)):
        rand_num = random.uniform(0, 1)  # Sinh số ngẫu nhiên trong khoảng từ 0 đến 1
        cumulative_prob = 0
        index = 0
        
        while cumulative_prob < rand_num:
            cumulative_prob += selection_probs[index]
            index += 1
        
        selected_parents.append(population[index - 1])  # Chọn cá thể tương ứng với vùng mà số ngẫu nhiên dừng lại
    
    return selected_parents

def normal_selection(population, offspring,fitness):
    individuals_with_fitness = list(zip(population + offspring, fitness))
    
    # Sắp xếp danh sách theo thứ tự giảm dần của fitness
    sorted_individuals = sorted(individuals_with_fitness, key=lambda x: x[1], reverse=True)
    
    # Chọn top những cá thể có fitness cao nhất
    top_individuals = []
    top_fitness = []

    for individual, fitness in sorted_individuals[:len(population)]:
        top_fitness.append(fitness)
        top_individuals.append(individual)

    return top_individuals ,top_fitness


def genetic_programing(jobs,operations,population,mutate_rate,crossover_rate, generation, is_tardiness= False):
    pop = copy.deepcopy(population)
    avg_fitnesses = [0]*generation
    fitness = [1/evaluate(jobs,operations,toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1]),is_tardiness = is_tardiness) for x in pop]
    for g in range(generation):
        progress = g / generation * 100
    
        print('\rProgress: [{:<100}] {:0.2f}%'.format('#' * math.floor(progress), progress), end='', flush=True)
        
        offspring = copy.deepcopy(pop)
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                a = random.randint(0, 1)
                b = (a+1) % 2
                child1[a], child2[a] = toolbox.mate(toolbox.clone(child1[a]), toolbox.clone(child2[a]))
                if a == 1:
                    child1[b], child2[b] = child2[b], child1[b]
                    
        for mutant in offspring:
            if random.random() < mutate_rate:
                index = random.randint(0, 1)
                mutant[index] = toolbox.mutate(mutant[index])[0]
        
        new_fitness = [1/evaluate(jobs,operations,toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1]),is_tardiness = is_tardiness) for x in offspring]
        
        pop,fitness = normal_selection(pop, offspring,fitness+new_fitness)
        # fitness = new_fitness
        for i in range(len(pop)):
        #   print( 1/fitness[i])
          avg_fitnesses[g]+= 1/fitness[i]/len(pop)

    best_individual = max(pop, key=lambda x: evaluate(jobs,operations,toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1]),is_tardiness = is_tardiness))
    return best_individual, avg_fitnesses

def get_best_ind(population, fitness):
    individuals_with_fitness = list(zip(population, fitness))
    best_individual = max(individuals_with_fitness, key=lambda x: x[1])
    return best_individual[0]

def genetic_programing_cc(jobs,operations,population1,population2,mutate_rate,crossover_rate, generation, is_tardiness= False):
    pop1 = copy.deepcopy(population1)
    pop2 = copy.deepcopy(population2)
    best_individual = [pop1[0] , pop2[0]]

    fitness1 = [1/evaluate(jobs,operations,toolbox.compile(expr=x), toolbox.compile(expr=pop2[0]),is_tardiness = is_tardiness) for x in pop1]
    fitness2 = [1/evaluate(jobs,operations,toolbox.compile(expr=pop1[0]), toolbox.compile(expr=x),is_tardiness = is_tardiness) for x in pop2]
    best_individual[0] = get_best_ind(pop1, fitness1)
    best_individual[1] = get_best_ind(pop2, fitness2)

    avg_fitnesses1 = [0]*generation
    avg_fitnesses2 = [0]*generation

    for g in range(generation):
        progress = g / generation * 100
    
        print('\rProgress: [{:<100}] {:0.2f}%'.format('#' * math.floor(progress), progress), end='', flush=True)
        
        offspring1 = copy.deepcopy(pop1)
        offspring2 = copy.deepcopy(pop2)
        
        for i in range(0, len(offspring1) - 1, 2):
            if random.random() < crossover_rate:
                offspring1[i], offspring1[i+1]=toolbox.mate(toolbox.clone(offspring1[i]), toolbox.clone(offspring1[i+1])) 
            if random.random() < crossover_rate:
                offspring2[i], offspring2[i+1]=toolbox.mate(toolbox.clone(offspring2[i]), toolbox.clone(offspring2[i+1])) 
            if random.random() > mutate_rate:
                offspring1[i] = toolbox.mutate(offspring1[i])[0]
            if random.random() > mutate_rate:
                offspring2[i+1] = toolbox.mutate(offspring2[i+1])[0]
            if random.random() > mutate_rate:
                offspring1[i+1] = toolbox.mutate(offspring1[i+1])[0]
            if random.random() > mutate_rate:
                offspring2[i] = toolbox.mutate(offspring2[i])[0]
   
        new_fitness1 = [1/evaluate(jobs,operations,toolbox.compile(expr=x), toolbox.compile(expr=best_individual[1]),is_tardiness = is_tardiness) for x in offspring1]
        new_fitness2 = [1/evaluate(jobs,operations,toolbox.compile(expr=best_individual[0]), toolbox.compile(expr=x),is_tardiness = is_tardiness) for x in offspring2]
        # new_fitness = [1/evaluate(jobs,operations,toolbox.compile(expr=x[0]), toolbox.compile(expr=x[1]),is_tardiness = is_tardiness) for x in offspring]
        
        pop1,fitness1 = normal_selection(pop1, offspring1,fitness1+new_fitness1)
        pop2,fitness2 = normal_selection(pop2, offspring2,fitness2+new_fitness2)
        best_individual[0] = get_best_ind(pop1, fitness1)
        best_individual[1] = get_best_ind(pop2, fitness2)
        # fitness = new_fitness
        for i in range(len(pop1)):
        #   print( 1/fitness[i])
          avg_fitnesses1[g]+= 1/fitness1[i]/len(pop1)
          avg_fitnesses2[g]+= 1/fitness2[i]/len(pop2)

    return best_individual, avg_fitnesses1,avg_fitnesses2

