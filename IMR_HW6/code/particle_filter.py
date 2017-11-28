

from models import *
import numpy as np
from itertools import izip


class SlamParticleFilter(object):

  def __init__(self, nparticles, initial_pose = Pose(35,35,0)):
    self.motion_model = MotionModel()
    self.us_model = USModel()

    self.nparticles = nparticles

    self.pose_particles =   [initial_pose for x in xrange(nparticles)]
    self.us_map_particles = [USMap() for x in xrange(nparticles)]


  def project_given_odometry(self, d_odo):
    self.pose_particles = [self.motion_model.sample_given_odo(pose,d_odo)
                           for pose in self.pose_particles]



  def update_given_obs(self, us):
    weights = np.ones(self.nparticles) 
    for x in xrange(len(weights)):
      weights[x] = self.us_model.update_map_given_obs(self.us_map_particles[x], 
                                                      self.pose_particles[x], 
                                                      us)

    s = sum(weights)
    if s == 0.0:
      weights[:] = 1/self.nparticles
    else:
      weights = weights/s

    self.resample(weights)


  def resample(self, weights):
    s = np.random.multinomial(self.nparticles, weights)

    self.pose_particles = [particle for i,n in enumerate(s) 
                           for particle in [self.pose_particles[i]]*n]

    # note: we need to copy the maps 
    self.us_map_particles = [particle.copy() for i,n in enumerate(s) 
                           for particle in [self.us_map_particles[i]]*n]

