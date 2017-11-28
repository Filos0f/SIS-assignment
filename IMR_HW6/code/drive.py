

from ev3slam import wii
from ev3slam import gui
from ev3slam import comm
from ev3slam import particle_filter
from ev3slam.models import *

from time import sleep, time

from matplotlib import pyplot as pp

class SlamPlotter(object):
  def __init__(self, comm):
    self.comm = comm
    self.last_odo = None
    self.pf = particle_filter.SlamParticleFilter(100)
    self.comm.on_new_data = self.handle_sample
    self.last_draw = time()

  def handle_sample(self, sample):
    odo = Odometry(sample[0], sample[1])
    us = sample[2]
    if self.last_odo is not None:
      self.pf.project_given_odometry(odo - self.last_odo)
      self.pf.update_given_obs(us)

    if time() - self.last_draw > 1:
      gui.draw_map(self.pf)
      self.last_draw = time()

    self.last_odo = odo


if __name__ == "__main__":

  pp.ion()

  comm = comm.SlamCommListenerThread()
  wii = wii.WiiCotroller(comm)
  sp = SlamPlotter(comm)


  print "Finding a wii..."
  nxt_wii.connect()
  
  print "Searching for brick..."
  nxt_comm.start()
  nxt_comm.join()




