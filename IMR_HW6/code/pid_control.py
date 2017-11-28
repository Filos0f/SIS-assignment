
from time import time

class PIDControl(object):
  """Dirt simple generic PID Controller Implementation"""

  def __init__(self, kp, ki, kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd

    self.int_error = 0
    self.last_error = 0
    self.last_sample_time = time()

  def set_point(self, sp):
    self.sp = sp


  def update(self, pv, sample_time):
    error = self.sp - pv
    self.int_error += error * (sample_time - self.last_sample_time)
    if (sample_time > self.last_sample_time):
      d_error = (error - self.last_error) / (sample_time - self.last_sample_time)
    else:
      d_error = 0

    self.last_error = error
    self.last_sample_time = sample_time

    return self.kp * error + self.ki * self.int_error + self.kd * d_error

