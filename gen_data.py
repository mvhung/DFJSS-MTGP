import pandas as pd
import numpy as np
import random
import ast
import math
import copy
from entity import Job, Operation

# global variables
NUM_MACHINES = 10
NUM_JOBS = 200
NUM_JOBS_TEST = 40
LEVEL_WEIGHT = 4
MIN_WORKLOAD = 2
MAX_WORKLOAD = 5
DELAY = 20
MAX_NUM_OPERATION = 8
POISSON_LAMDA = 4

def random_processing_rate():
    num_machines = np.random.randint(1,NUM_MACHINES+1)
    processing_rate = np.random.uniform(0.5, 1, size=num_machines)
    numbers = np.arange(0, NUM_MACHINES)
    random_array = np.random.permutation(numbers)[:num_machines]

    processing_rate = processing_rate + random_array
    return processing_rate

def gen_arrival_time_with_poisson(time_interval = 1, num_jobs = 10):
  arrival_times_poisson = np.random.poisson(POISSON_LAMDA, size=num_jobs)

  arrival_times = np.cumsum(arrival_times_poisson) * time_interval
  return arrival_times
def generate_dfjss_data(num_jobs = NUM_JOBS, poisson_lambda = POISSON_LAMDA, max_num_operation = MAX_NUM_OPERATION):
    arrival_times = gen_arrival_time_with_poisson(num_jobs=num_jobs)
    weights = np.random.randint(0, LEVEL_WEIGHT, size=num_jobs)
    num_operations = np.random.randint(1, max_num_operation, size=num_jobs)
    jobs = []
    operations = []
    for i in range(num_jobs):
      arrival_time = arrival_times[i]
      weight = weights[i]
      num_operation = num_operations[i]
      due_date = arrival_time
      for j in range(num_operation):
          workload = np.random.randint(MIN_WORKLOAD, MAX_WORKLOAD)
          processing_rate = random_processing_rate()
          operation = Operation(id = j, job_id = i,workload=workload, processing_rate=processing_rate)
          operations.append(operation)
          due_date = due_date + workload
      delay_can_accept = np.random.randint(0,DELAY)
      due_date = due_date + delay_can_accept

      job = Job(id = i ,arrival_time=arrival_time, due_date=due_date, weight=weight, num_operation=num_operation)
      jobs.append(job)

    jobs_df = pd.DataFrame({
    'Job_ID': [job.id for job in jobs],
    'Arrival_Time': [job.arrival_time for job in jobs],
    'Due_Date': [job.due_date for job in jobs],
    'Weight': [job.weight for job in jobs],
    'Num_Operation': [job.num_operation for job in jobs]
    })

    # Tạo DataFrame cho các thao tác
    operations_df = pd.DataFrame({
        'Operation_ID':[op.id for op in operations],
        'Job_ID':[op.job_id for op in operations],
        'Work_Load': [op.workload for op in operations],
        'Processing_Rate': [op.processing_rate for op in operations]
    })

    return jobs_df, operations_df

# jobs_df, operations_df = generate_dfjss_data()

# jobs_df.to_csv(f"jobs_data_{NUM_JOBS}.csv", index=False)
# operations_df.to_csv(f"operations_data_{NUM_JOBS}.csv", index=False)

# jobs_df, operations_df = generate_dfjss_data(num_jobs = NUM_JOBS_TEST)

# jobs_df.to_csv(f"jobs_data_test_{NUM_JOBS}.csv", index=False)
# operations_df.to_csv(f"operations_data_test_{NUM_JOBS}.csv", index=False)