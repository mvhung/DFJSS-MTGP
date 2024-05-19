import time
import pandas as pd
import numpy as np
import random
import ast
import math
import copy
from entity import Job, Operation
from gen_data import NUM_MACHINES, NUM_JOBS
from tools import  read_convert_data
from deap import base, creator, gp, tools
import operator
import itertools
from GP import genetic_programing_cc, toolbox, genetic_programing
import pickle

jobs_train , operations_train = read_convert_data("jobs_data_200.csv", "operations_data_200.csv")

POPULATION_SIZE = 200
MUTATE_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATION = 50
pop_sequencing = toolbox.population(n=POPULATION_SIZE)
pop_routing = toolbox.population(n=POPULATION_SIZE)
pop = []
for i in range(POPULATION_SIZE):
  pop.append([pop_sequencing[i], pop_routing[i]])

start_time = time.time()
best_individual_flow, avg_fitnesses_flow = genetic_programing(jobs_train,operations_train,pop,MUTATE_RATE,CROSSOVER_RATE,GENERATION,is_tardiness=False)
end_time = time.time()
print(f"Thời gian thực hiện thuật toán mtgp: {end_time - start_time} giây")

pop_sequencing = toolbox.population(n=int(POPULATION_SIZE/2))
pop_routing = toolbox.population(n=int(POPULATION_SIZE/2))
start_time = time.time()
best_individual_flow_cc, avg_fitnesses_flow_cc1 ,avg_fitnesses_flow_cc2 = genetic_programing_cc(jobs_train,operations_train,pop_sequencing,pop_routing,MUTATE_RATE,CROSSOVER_RATE,GENERATION,is_tardiness=False)
end_time = time.time()
print(f"Thời gian thực hiện thuật toán ccgp: {end_time - start_time} giây")

filename = "avg_fitnesses_flow.txt"

filename1 = "avg_fitnesses_flow_cc1.txt"
filename2 = "avg_fitnesses_flow_cc2.txt"

with open(filename1, "w") as file:
    for element in avg_fitnesses_flow_cc1:
        file.write(str(element) + "\n")
with open(filename2, "w") as file:
    for element in avg_fitnesses_flow_cc2:
        file.write(str(element) + "\n")
# Lưu cá thể vào tệp tin
with open('sequencing_rule_flow_cc.pkl', 'wb') as file:
    pickle.dump(best_individual_flow_cc[0], file)

with open('routing_rule_flow_cc.pkl', 'wb') as file:
    pickle.dump(best_individual_flow_cc[1], file)

with open(filename, "w") as file:
    for element in avg_fitnesses_flow:
        file.write(str(element) + "\n")
# Lưu cá thể vào tệp tin
with open('sequencing_rule_flow.pkl', 'wb') as file:
    pickle.dump(best_individual_flow[0], file)

with open('routing_rule_flow.pkl', 'wb') as file:
    pickle.dump(best_individual_flow[1], file)