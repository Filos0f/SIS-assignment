
import math
import numpy as np
import copy
from state import *

class MotionModel(object):

  def __init__(self, wheel_circ = 1, base_circ = 1.3889885942565414, variance_scale = 0.02):
    self.wheel_circ = wheel_circ
    self.base_circ  = base_circ 
    self.variance_scale = variance_scale


  def compute_motion(self, pose, d_odo):
    d_theta = (d_odo.right * self.wheel_circ -
               d_odo.left * self.wheel_circ) / self.base_circ

    d_x = ((d_odo.left * self.wheel_circ +
               d_odo.right * self.wheel_circ) / 2 
           * math.cos(pose.theta+d_theta/2))
    
    d_y = ((d_odo.left * self.wheel_circ +
               d_odo.right * self.wheel_circ) / 2 
           * math.sin(pose.theta+d_theta/2))
    
    return Pose(pose.x + d_x, pose.y + d_y, (pose.theta + d_theta) % (math.pi * 2))

  def sample_given_odo(self, pose, d_odo):
    sd_odo = Odometry(
      np.random.normal(d_odo.left, self.variance_scale * abs(d_odo.left) + 0.02),
      np.random.normal(d_odo.right, self.variance_scale * abs(d_odo.left) + 0.02)
    )
    return self.compute_motion(pose, sd_odo)

  def compute_tacho_from_theta(self, theta):
    d_o = self.base_circ * theta / (2.0*self.wheel_circ)
    return Odometry(d_o, -d_o)

  def compute_tacho_from_dist(self, dist):
    return Odometry(dist/self.wheel_circ, dist/self.wheel_circ)

class CompassModel(object):
  pass

class ColorModel(object):
  pass

class USModel(object):

  FALSE_POSITIVE = 0.01
  FALSE_NEGATIVE = 0.3
  LAMBDA = 0.1

  def __init__(self, scaling_ratio = 0.1, scaling_offset = 1.30790162, std_dev = 2):
    self.scaling_ratio = scaling_ratio
    self.scaling_offset = scaling_offset
    self.std_dev = std_dev
    self.scale = 1.0

    #self.view_angle = 0.2617
    self.view_angle = 0.2
    self.us_max = self.normalize_us(180)

    #precompute polygon for easy computation of viewable features
    self.view_polygon = np.array([
      [0,0,1],
      [math.cos(self.view_angle) * self.us_max, math.sin(self.view_angle) * self.us_max, 1],
      [self.us_max, 0, 1],
      [math.cos(-self.view_angle) * self.us_max, math.sin(-self.view_angle) * self.us_max, 1],
    ]).transpose()


  def _norm_pdf(self,x, mu, std_dev):
    """scipy.stats.norm.pdf is WAY too slow"""
    return (math.exp(-(x-mu)**2.0/ (2*std_dev**2))
            /(math.sqrt(2*math.pi*std_dev**2)))

  def _expon_cdf(self, x, lam):
    return 1.0 - math.exp(-lam * x)

  def update_map_given_obs(self, usmap, pose, us):
    d_us = self.normalize_us(us)

    weight = 1

    dxy = [ (self.distance_to_feature(pose,x,y),x,y) 
           for x,y in self.get_visible_features(usmap, pose)]

    p_map = np.exp(usmap)/(np.exp(usmap)+1)
    dxy.sort()
    #compute range probabilities
    if len(dxy) > 0:
      S = np.zeros(max(us,self.unnormalize_us(dxy[-1][0]))+1)
    else:
      S = np.zeros(us)

    S.fill(self.FALSE_POSITIVE)
    for d,x,y in dxy:
      S[self.unnormalize_us(d)] += p_map[x,y] * self.FALSE_NEGATIVE 

    S = S/S.sum()

    P = np.zeros(S.shape)
    p_sum = 0
    for i,s in enumerate(S):
      P[i] = max(0,S[i] * (1-p_sum))
      p_sum += P[i]


    for d,x,y in dxy:
      if not (d < min(d_us+1.0, self.us_max)):
        # can't see it
        continue
      elif abs(d-d_us) < 1.0:
        p_m_occ = 1 - P[:us].sum()
        #p_m_occ = 1
        p_m_emp = P[us]
      else:
        #probability of measurement given occupied
        p_m_occ = 0.0
        #probability of measurement given empty
        p_m_emp = P[us]
        #p_m_emp = 1 

      p_m_occ = (p_m_occ * (1 - self.FALSE_NEGATIVE) + 
                p_m_emp * self.FALSE_NEGATIVE)

      p_m_emp = (p_m_emp * (1 - self.FALSE_POSITIVE) + 
                p_m_occ * self.FALSE_POSITIVE)


      # add log likelihood ratio
      usmap[x,y] += math.log(p_m_occ/p_m_emp)
      
    if us < P.shape[0]:
      return P[us]
    else:
      return P[-1]


  def distance_to_feature(self, pose, fx, fy):
    d = math.sqrt((fx - pose.x)**2 + (fy - pose.y)**2)
    return d

  def angle_to_feature(self, pose, fx, fy):
    p_vec_x, p_vec_y = pose.norm_vector()
    f_vec_x, f_vec_y = (fx-pose.x, fy-pose.y)
    f_norm = math.sqrt(f_vec_x*f_vec_x + f_vec_y*f_vec_y)
    if f_norm == 0: return np.nan
    dot = (p_vec_x * f_vec_x+p_vec_y * f_vec_y)/f_norm
    th = math.acos(dot)
    return th

  def get_visible_features(self,m, pose):

    # transform the view polygon
    trans_polygon = pose.rigid_trans_matrix().dot(self.view_polygon)

    #initialize the empty feature list
    feature_list = []

    min_x = max(int(round(trans_polygon[0,:].min())), 0)
    max_x = min(int(round(trans_polygon[0,:].max())), m.shape[0]-1)

    for x in xrange(min_x, max_x + 1):
      #construct an intersection list
      y_ints = []

      # we assume without checking that the view polygon is convex
      for vertex in xrange(trans_polygon.shape[1]):
        if (trans_polygon[0,vertex] == trans_polygon[0,vertex-1]):
          y_ints.append(trans_polygon[1,vertex])
          y_ints.append(trans_polygon[1,vertex-1])
        else:
          portion = ((x - trans_polygon[0,vertex]) / 
                     (trans_polygon[0,vertex-1]-trans_polygon[0,vertex]))

          #print x, trans_polygon[0,vertex], trans_polygon[0,vertex-1], portion
          if (0 <= portion <= 1):
            y_ints.append((portion * trans_polygon[1,vertex-1]) + 
                          ((1-portion) * trans_polygon[1,vertex]))

        if len(y_ints) >= 2:
          break

      if len(y_ints) > 0:
        min_y = max(int(round(min(y_ints))),0)
        max_y = min(int(round(max(y_ints))),m.shape[1]-1)

        for y in xrange(min_y, max_y + 1):
          feature_list.append((x,y))

    return feature_list

  def normalize_us(self, us):
    return self.scaling_ratio * us + self.scaling_offset

  def unnormalize_us(self, d):
    return (d - self.scaling_offset) / self.scaling_ratio

  def in_map(self,m,x,y):
    return 0<=x<m.shape[0] and 0<=y<m.shape[1]

