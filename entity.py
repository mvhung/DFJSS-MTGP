import math

class Job:
    def __init__(self, id,arrival_time, due_date, weight, num_operation):
        self.id = id
        self.arrival_time = arrival_time
        self.due_date = due_date
        self.weight = weight
        self.num_operation = num_operation

    def to_string(self):
      return f"Job ID: {self.id}, Arrival Time: {self.arrival_time}, Due Date: {self.due_date}, Weight: {self.weight}, Num Operation: {self.num_operation}"

class Operation:
    def __init__(self, id, job_id,workload, processing_rate):
        self.ready_time = 0
        self.id = id
        self.job_id = job_id
        self.workload = workload
        self.processing_rate = processing_rate

    def set_ready_time(self, current_time):
      self.ready_time = current_time

    def get_procesing_time(self, machine_id):
      for i in self.processing_rate:
        if math.floor(i) == machine_id:
          return math.ceil(self.workload / (i -  math.floor(i)))
      return None

    def to_string(self):
      return f"Operation ID: {self.id}, Job ID: {self.job_id}, Workload: {self.workload}, Processing Rate: {self.processing_rate}, ready_time: {self.ready_time}"
