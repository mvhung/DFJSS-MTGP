import pickle
import random
import matplotlib.pyplot as plt
from gen_data import NUM_JOBS, NUM_MACHINES
from tools import evaluate, evaluate_for_plot, read_array_from_txt, read_convert_data
from GP import toolbox
import numpy as np

best_individual_pth1= ['sequencing_rule_flow.pkl', 'routing_rule_flow.pkl']
best_individual_pth2= ['sequencing_rule_flow_cc.pkl', 'routing_rule_flow_cc.pkl']
avg_pth = "avg_fitnesses_flow.txt"
avg_pth2 = "avg_fitnesses_flow_cc1.txt"
avg_pth3 = "avg_fitnesses_flow_cc2.txt"
jobs , operations = read_convert_data("jobs_data_test_200.csv", "operations_data_test_200.csv")
num_resources = 10
is_tardiness = False
best_individual1 = [None, None]
best_individual2 = [None, None]
avg_fitnesses1 = read_array_from_txt(avg_pth)
avg_fitnesses2 = read_array_from_txt(avg_pth2)
avg_fitnesses3 = read_array_from_txt(avg_pth3)

# with open('sequencing_rule.pkl', 'rb') as file:
#     best_individual[0] = pickle.load( file)
with open(best_individual_pth1[0], 'rb') as file:
    best_individual1[0] = pickle.load( file)

with open(best_individual_pth1[1], 'rb') as file:
    best_individual1[1] = pickle.load( file)

with open(best_individual_pth2[0], 'rb') as file:
    best_individual2[0] = pickle.load( file)

with open(best_individual_pth2[1], 'rb') as file:
    best_individual2[1] = pickle.load( file)

indices1 = list(range(len(avg_fitnesses1)))
indices2 = list(range(len(avg_fitnesses2)))
indices3 = list(range(len(avg_fitnesses3)))

plt.figure(figsize=(10, 6))  # Điều chỉnh kích thước biểu đồ

plt.plot(indices1, avg_fitnesses1,  label='MTGP')
plt.plot(indices2, avg_fitnesses2,  label='Sequencing-CCGP')
plt.plot(indices3, avg_fitnesses3,  label='Routing-CCGP')

plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Average Fitness over Generations')
plt.legend()  # Hiển thị chú giải
plt.grid(True)
plt.show()

time, schedule = evaluate_for_plot(jobs,operations,toolbox.compile(expr=best_individual1[0]), toolbox.compile(expr=best_individual1[1]),is_tardiness=is_tardiness)
time1, schedule1 = evaluate_for_plot(jobs,operations,toolbox.compile(expr=best_individual2[0]), toolbox.compile(expr=best_individual2[1]),is_tardiness=is_tardiness)



# Tạo hình và 2 trục con
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Vẽ lịch trình 1
colors1 = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(jobs))]
for resource, start, duration, job, label in schedule:
    ax1.broken_barh([(start, duration)], (resource-0.4, 0.8), facecolors=colors1[job])
    ax1.text(start + duration/2, resource, label, ha='center', va='center', color='white')
ax1.set_title('Lịch trình 1')
ax1.set_yticks(np.arange(0, num_resources))
ax1.set_yticklabels([f'M{i}' for i in range(1, num_resources+1)])
ax1.set_xlabel('Thời gian')
ax1.set_ylabel('Máy')

# Vẽ lịch trình 2
# colors2 = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(jobs))]
for resource, start, duration, job, label in schedule1:
    ax2.broken_barh([(start, duration)], (resource-0.4, 0.8), facecolors=colors1[job])
    ax2.text(start + duration/2, resource, label, ha='center', va='center', color='white')
ax2.set_title('Lịch trình 2')
ax2.set_yticks(np.arange(0, num_resources))
ax2.set_yticklabels([f'M{i}' for i in range(1, num_resources+1)])
ax2.set_xlabel('Thời gian')
ax2.set_ylabel('Máy')

# Hiển thị biểu đồ
plt.show()