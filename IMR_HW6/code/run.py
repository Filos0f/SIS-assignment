
import pickle
import sys

from ev3slam import models
from ev3slam import particle_filter
from ev3slam import gui

from matplotlib import pyplot as pp


pp.ion()

if __name__ == "__main__":

  with open(sys.argv[1]) as f:
    samples = pickle.load(f)

  pf = particle_filter.SlamParticleFilter(200)

  for i in xrange(1,len(samples)):
    print("Processing sample %i of %i" % (i, len(samples) -1))
    d_odo = models.Odometry(samples[i][0] - samples[i-1][0], 
                            samples[i][1] - samples[i-1][1])
    us = samples[i][2]     
    
    pf.project_given_odometry(d_odo)
    pf.update_given_obs(us)

    gui.draw_map(pf)

  pp.show()

