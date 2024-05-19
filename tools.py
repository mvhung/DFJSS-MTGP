# helper functions
import pandas as pd
import numpy as np
import random
import ast
import math
import copy
from entity import Job, Operation
from gen_data import NUM_MACHINES,NUM_JOBS,NUM_JOBS_TEST,LEVEL_WEIGHT,MIN_WORKLOAD,MAX_WORKLOAD,DELAY,MAX_NUM_OPERATION,POISSON_LAMDA
from collections import defaultdict

def get_job_by_id(job_id, jobs):
  for job in jobs:
    if job.id == job_id:
      return job
  return None

def get_operation_by_id(job_id, operation_id, operations):

  for operation in operations[job_id]:
    if operation.id == operation_id:
      return operation
  return None

def get_operations_by_job_id(job_id,operations):
  return operations[job_id]

def get_num_job_in_queue(operation_queue):
  job_count = []
  for op in operation_queue:
    if op.job_id not in job_count:
      job_count.append(op.job_id)
  return len(job_count)

def get_next_operation(job_id, operation_id,operations):
  for operation in operations[job_id]:
    if operation.id == operation_id + 1:
      return operation
  return None

def get_jobs_by_timespan(timespan, jobs):
  new_jobs = []
  for job in  jobs:
    if job.arrival_time == timespan:
      new_jobs.append(job)
  return new_jobs

def show_operation_in_queue(operation_queue):
  print("[", end=" ")
  print(" ".join(str(op.id) + str(op.job_id) for op in operation_queue), end=" ")
  print("]")
def read_processing_rate(str):
  return np.fromstring(str.strip('[]'), sep=' ')

def read_array_from_txt(file_path):
    array = []
    with open(file_path, 'r') as file:
        # Đọc nội dung của file
        content = file.read()

        # Chuyển đổi chuỗi thành mảng
        array = content.split()

        # Chuyển đổi các phần tử thành kiểu số nguyên
        array = [float(x) for x in array]
    return array


def read_convert_data(job_file, operation_file):
    jobs_df_read = pd.read_csv(job_file)
    operations_df_read = pd.read_csv(operation_file)
    NUM_JOBS = len(jobs_df_read)
    jobs_train = []
    operations_train = [None] * NUM_JOBS

    for i in range(NUM_JOBS):
        operations_train[i] = []

    for job in jobs_df_read.iterrows():
        job_id = job[1]['Job_ID']
        arrival_time = job[1]['Arrival_Time']
        due_date = job[1]['Due_Date']
        weight = job[1]['Weight']
        num_operation = job[1]['Num_Operation']
        jobs_train.append(Job(job_id, arrival_time, due_date, weight, num_operation))
    for operation in operations_df_read.iterrows():
        operation_id = operation[1]['Operation_ID']
        job_id = operation[1]['Job_ID']
        workload = operation[1]['Work_Load']
        processing_rate = read_processing_rate(operation[1]['Processing_Rate'])
        operations_train[job_id].append(Operation(operation_id, job_id, workload, processing_rate))
    
    return jobs_train, operations_train

def get_all_current_terminal(jobs, operations,job_id,op_id,machine_id, job_queue, operation_queue, operation_in_machine_queue, operations_ready_time,time):
  job = get_job_by_id(job_id,jobs)
  operation = get_operation_by_id(job_id, op_id, operations)
  operations_in_job = get_operations_by_job_id(job_id,operations)
  TIS = time - job.arrival_time # thời gian ở trong xưởng của công việc
  W = job.weight # trọng số công việc
  NOR = job.num_operation # số thao tác còn lại của công việc
  WKR = len(job_queue) # số công việc còn lại (trong hàng đợi job)
  rDD = job.due_date - time #số ngày trước hạn
  PT = operation.get_procesing_time(machine_id) # thời gian thực thi thao tác
  OWT = time - operations_ready_time[job_id][op_id] # thời gian đợi của thao tác
  NPT = 0 #thời gian thực thi trung bình của các thao tác
  for op in operations_in_job:
    NPT += op.workload/len(operations_in_job)
  NIQ = len(operation_in_machine_queue[machine_id]) # số thao tác trên hàng đợi máy
  MIQ = get_num_job_in_queue(operation_in_machine_queue[machine_id]) # số công việc trên hàng đợi máy


  return TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ

def evaluate(jobs_converted,operations_converted,sequencing_rule, routing_rule,toolbox=None, is_tardiness = False):
  if(toolbox != None):
    sequencing_rule = toolbox.compile(expr=sequencing_rule)
    routing_rule = toolbox.compile(expr=routing_rule)
    
  jobs = copy.deepcopy(jobs_converted)
  NUM_JOBS = len(jobs)
  operations = copy.deepcopy(operations_converted)
  operations_queue = []
  job_queue = []
  operations_in_machines_queue = [None]*NUM_MACHINES
  for i in range(NUM_MACHINES):
    operations_in_machines_queue[i] = []

  machines_ready_time = [0]*NUM_MACHINES
  operations_ready_time = [None]*NUM_JOBS
  for i in range(NUM_JOBS):
    operations_ready_time[i] = [0]*len(operations[i])

  time = 0
  tardiness = [None]*NUM_JOBS
    
# hàm của máy kiểm soát các thao tác trên hàng đợi của máy
  def machine_thread(machine_id):
    operations_queue_in_machine = operations_in_machines_queue[machine_id]
    max_prio = None
    next_operation = None
    if time >= machines_ready_time[machine_id]:
      for operation in operations_queue_in_machine:
        TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ = get_all_current_terminal(jobs, operations,operation.job_id, operation.id, machine_id, job_queue, operations_queue, operations_in_machines_queue,operations_ready_time, time)
        priority = sequencing_rule(TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ)
        if max_prio == None:
          max_prio = priority
          next_operation = operation
        elif priority >= max_prio:
          max_prio = priority
          next_operation = operation
    if next_operation != None:
      machines_ready_time[machine_id] = time + next_operation.get_procesing_time(machine_id)
      if len(operations[next_operation.job_id]) > next_operation.id + 1:
        operations_ready_time[next_operation.job_id][next_operation.id+1] = machines_ready_time[machine_id]
      else :
        job = get_job_by_id(next_operation.job_id,jobs)
        tardiness[next_operation.job_id] = machines_ready_time[machine_id] - job.due_date
      operations_in_machines_queue[machine_id].remove(next_operation)

  def job_coming():
    new_jobs = get_jobs_by_timespan(time,jobs)
    if len(new_jobs) != 0:
      job_queue.extend(new_jobs)
      for job in new_jobs:
        operations_queue.extend(operations[job.id])


  def routing_controller():
    for op in operations_queue:
      if (op.id != 0 and operations_ready_time[op.job_id][op.id] == 0):
        continue
      if operations_ready_time[op.job_id][op.id] > time :
        continue
     
      max_prio = None
      machine_selected = None
      list_machine = np.floor(op.processing_rate).astype(int)
      for machine_id in list_machine:
        TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ = get_all_current_terminal(jobs, operations,op.job_id, op.id, machine_id, job_queue, operations_queue, operations_in_machines_queue, operations_ready_time,time)
        priority = routing_rule(TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ)
        if max_prio == None:
          max_prio = priority
          machine_selected = machine_id
        elif priority >= max_prio:
          max_prio = priority
          machine_selected = machine_id
      if machine_selected != None:
        operations_queue.remove(op)
        operations_in_machines_queue[machine_selected].append(op)
        # print(f"Operation {op.id} of job {op.job_id} is routing to machine {machine_selected} at time {time}")

  def is_done():
    if len(job_queue) != NUM_JOBS:
      return False
    if (len(operations_queue) != 0):
      return False
    for i in range(NUM_MACHINES):
      # print(f"there are {len(operations_in_machines_queue[i])} left in machine {i}")
      if len(operations_in_machines_queue[i]) != 0:
        return False
    return True

  while is_done() == False:
    job_coming()
    routing_controller()
    for i in range(NUM_MACHINES):
      machine_thread(i)
    time += 1
  
  if is_tardiness:
    return max(tardiness)
  else :
    return time

def evaluate_for_plot(jobs_converted,operations_converted,sequencing_rule, routing_rule,is_tardiness = False):
  schedule = []
  jobs = copy.deepcopy(jobs_converted)
  NUM_JOBS = len(jobs)
  operations = copy.deepcopy(operations_converted)
  operations_queue = []
  job_queue = []
  operations_in_machines_queue = [None]*NUM_MACHINES
  for i in range(NUM_MACHINES):
    operations_in_machines_queue[i] = []

  machines_ready_time = [0]*NUM_MACHINES
  operations_ready_time = [None]*NUM_JOBS
  for i in range(NUM_JOBS):
    operations_ready_time[i] = [0]*len(operations[i])

  time = 0
  tardiness = [None]*NUM_JOBS

  def machine_thread(machine_id):
    operations_queue_in_machine = operations_in_machines_queue[machine_id]
    max_prio = None
    next_operation = None
    if time >= machines_ready_time[machine_id]:
      for operation in operations_queue_in_machine:
        TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ = get_all_current_terminal(jobs, operations,operation.job_id, operation.id, machine_id, job_queue, operations_queue, operations_in_machines_queue,operations_ready_time, time)
        priority = sequencing_rule(TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ)
        if max_prio == None:
          max_prio = priority
          next_operation = operation
        elif priority >= max_prio:
          max_prio = priority
          next_operation = operation
    if next_operation != None:
      machines_ready_time[machine_id] = time + next_operation.get_procesing_time(machine_id)
      
      if len(operations[next_operation.job_id]) > next_operation.id + 1:
        operations_ready_time[next_operation.job_id][next_operation.id+1] = machines_ready_time[machine_id]
      else :
        job = get_job_by_id(next_operation.job_id,jobs)
        tardiness[next_operation.job_id] = machines_ready_time[machine_id] - job.due_date
      
      operations_in_machines_queue[machine_id].remove(next_operation)
      schedule.append((machine_id, time,machines_ready_time[machine_id]-time,next_operation.job_id,f"{next_operation.job_id}-{next_operation.id}"))

  def job_coming():
    new_jobs = get_jobs_by_timespan(time,jobs)
    if len(new_jobs) != 0:
      job_queue.extend(new_jobs)
      for job in new_jobs:
        operations_queue.extend(operations[job.id])


  def routing_controller():
    for op in operations_queue:
      if (op.id != 0 and operations_ready_time[op.job_id][op.id] == 0):
        continue
      if operations_ready_time[op.job_id][op.id] > time :
        continue
      max_prio = None
      machine_selected = None
      list_machine = np.floor(op.processing_rate).astype(int)
      for machine_id in list_machine:
        TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ = get_all_current_terminal(jobs, operations,op.job_id, op.id, machine_id, job_queue, operations_queue, operations_in_machines_queue, operations_ready_time,time)
        priority = routing_rule(TIS, W, NOR, WKR, rDD, PT, OWT, NPT, NIQ, MIQ)
        if max_prio == None:
          max_prio = priority
          machine_selected = machine_id
        elif priority >= max_prio:
          max_prio = priority
          machine_selected = machine_id
      if machine_selected != None:
        operations_queue.remove(op)
        operations_in_machines_queue[machine_selected].append(op)

  def is_done():
    if len(job_queue) != NUM_JOBS:
      return False
    if (len(operations_queue) != 0):
      return False
    for i in range(NUM_MACHINES):
      if len(operations_in_machines_queue[i]) != 0:
        return False
    return True

  while is_done() == False:
    job_coming()
    routing_controller()
    for i in range(NUM_MACHINES):
      machine_thread(i)
    time += 1
  if is_tardiness :
    return max(tardiness), schedule
  else:
    return time, schedule


__type__ = object
def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2
