
import ev3
from time import time
import threading
from Queue import Queue

import pickle

from state import *
from pid_control import PIDControl

class SlamComm(object):



  def __init__(self):
    self.brick = ev3.find_one_brick()

    #set up sensors
    for p in range(4):
      try:
        s = self.brick.get_sensor(p)
        if isinstance(s,ev3.sensor.hitechnic.Compass):
          self.compass = s
        elif isinstance(s,ev3.sensor.hitechnic.Accelerometer):
          self.accel = s
        elif isinstance(s,ev3.sensor.generic.Ultrasonic):
          self.us = s
      except:
        # we assume the only other thing connected is a color sensor
        self.color = ev3.sensor.Color20(self.brick, p)

    #set up motors
    self.m_left = ev3.Motor(self.brick, ev3.PORT_C)
    self.m_right = ev3.Motor(self.brick, ev3.PORT_B)

  def get_sample(self):
    return (
      self.m_left.get_tacho().tacho_count  / -360.0,
      self.m_right.get_tacho().tacho_count  / -360.0,
      self.us.get_distance(),
      self.compass.get_heading(),
      self.color.get_color(),
      time()
    )

  def stop_motors(self):
    for m in (self.m_left, self.m_right):
      m.idle()
      

  def drive_reg(self, power, turn_ratio):
    for m in (self.m_left, self.m_right):
      m.reset_position(True)
      m.turn_ratio = turn_ratio
      m.sync = True
      m.run(-power, True)
      

  def drive_unreg(self, motor_power):
    for m, power in zip((self.m_left, self.m_right), motor_power):
      m.reset_position(True)
      m.sync = False
      m.run(-power, False)

class SlamCommListenerThread(threading.Thread):

  def __init__(self, on_new_data = None, on_status = None):
    super(SlamCommListenerThread,self).__init__()
    self.on_new_data = on_new_data or self._pass
    self.on_status = on_status or self._pass
    self._connected = False
    self._comm_lock = threading.Lock()
    self._logging = False
    self._log = []
    self._last_sample = None
    self._way_point_queue = Queue()
    self._current_way_point = None

    self._left_pid_control = PIDControl(10,1,1)
    self._right_pid_control = PIDControl(10,1,1)
    self._left_pid_control.int_error = 60 # a good starting point
    self._right_pid_control.int_error = 60 # a good starting point

    self._command_queue = Queue()

  def run(self):
    try:
      self.connect()
      with self._comm_lock:
        self._last_sample = self._comm.get_sample()

      while True:
        with self._comm_lock:
          if not self._connected: return
          sample = self._comm.get_sample()
        self._handle_way_points(self._last_sample, sample)
        self.on_new_data(sample)
        
        if not self._command_queue.empty():
          #drain queue of all commands but the last one.
          while not self._command_queue.empty():
            command, args = self._command_queue.get()
          print "Comm: Executing %s with %s" %(command,repr(args))
          getattr(self._comm, command)(*args)
        

        if self._logging:
          self._log.append(sample)

        self._last_sample = sample

    except Exception as e:
      try:
        self.disconnect()
      finally:
        self.on_status("Disconnected: %s" % repr(e))

  def connect(self):
    with self._comm_lock:
      if not self._connected:
        self._comm = SlamComm()
        self._connected = True
        self.on_status("Connected to: %s" % self._comm.brick.get_device_info()[1])

  def disconnect(self):
    with self._comm_lock:
      if self._connected:
        self._comm.brick.sock.close()
        self._comm = None
        self._connected = False
        self.on_status("Disconnected.")

  def start_logging(self):
    if not self._logging:
      self._log = []
      self._logging = True

  def stop_logging(self, filename=None):
    self._logging = False
    if filename:
      with open(filename, "w") as f:
        pickle.dump(self._log, f)
    return self._log

  def drive_reg(self, power, turn_ratio):
    if self._connected:
      self._command_queue.put(('drive_reg',(power, turn_ratio)))

  def drive_unreg(self, motor_power):
    if self._connected:
      self._command_queue.put(('drive_unreg',(motor_power,)))

  def stop_motors(self):
    if self._connected:
      self._command_queue.put(('stop_motors',()))

  def add_way_point(self,odo):
    self._way_point_queue.put((odo.left,odo.right))


  def _handle_way_points(self, last_sample, current_sample):
    if self._current_way_point is None:
      if not self._way_point_queue.empty():
        self._current_way_point = self._way_point_queue.get()
        #reset sample times for pid controller ... avoids wild oscillation
        self._left_pid_control.last_sample_time = current_sample[5]
        self._right_pid_control.last_sample_time = current_sample[5]
      else:
        return

    goal_left, goal_right = self._current_way_point
    err_left = goal_left - current_sample[0] 
    err_right = goal_right - current_sample[1]
    if (abs(err_left) < 0.3 and abs(err_right) < 0.3):
      # We are there!
      self._way_point_queue.task_done()
      if self._way_point_queue.empty():
        self.stop_motors()
        self._current_way_point = None
      else:
        self._current_way_point = self._way_point_queue.get()

    if self._current_way_point:
      self._update_motor_power(last_sample, current_sample)


  def _update_motor_power(self, last_sample, current_sample):
    goal_left, goal_right = self._current_way_point
    err_left = goal_left - current_sample[0] 
    err_right = goal_right - current_sample[1]
    d_time = current_sample[5] - last_sample[5]
    speed_left = abs((current_sample[0] - last_sample[0]) / d_time)
    speed_right = abs((current_sample[1] - last_sample[1]) / d_time)

    self._left_pid_control.set_point(min(abs(err_left/(2*d_time)), 0.5))
    self._right_pid_control.set_point(min(abs(err_right/(2*d_time)), 0.5))
    p_left = self._left_pid_control.update(speed_left, current_sample[5])
    p_right = self._right_pid_control.update(speed_right, current_sample[5])

    p_left = min(100, max(0, p_left))  * (err_left > 0 or -1)
    p_right = min(100, max(0, p_right))  * (err_right > 0 or -1)
    #print goal_left, goal_right, err_left, err_right, s_left, s_right, p_left, p_right
    with self._comm_lock:
      self._comm.drive_unreg((p_left, p_right))

  def cancel_way_points(self):
    while not self._way_point_queue.empty():
      self._way_point_queue.get()
    self._current_way_point = None

  def empty_way_points(self):
    return self._way_point_queue.empty() and (self._current_way_point is None)

  @property
  def logging(self):
    return self._logging

  @property
  def connected(self):
    return self._connected

  def _pass(self, *args):
    print args
    #pass
