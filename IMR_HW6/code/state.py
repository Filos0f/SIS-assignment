
from math import sin, cos, ceil, floor, sqrt, acos
from scipy.ndimage import map_coordinates
import numpy as np
import itertools

__all__ = ['Pose', 'Odometry', 'CompassMap', 'ColorMap', 'USMap']

class Pose(object):
  __slots__ = ['x','y','theta']

  def __init__(self, x, y, theta):
    self.x, self.y, self.theta = x,y,theta

  def __repr__(self):
    return repr(self.to_nparray())

  def to_nparray(self):
    return np.array([self.x,self.y,self.theta])

  def rigid_trans_matrix(self):
    return np.array([[cos(self.theta), -sin(self.theta), self.x],
                     [sin(self.theta), cos(self.theta), self.y],
                     [0.0, 0.0, 1.0]])

  def norm_vector(self):
    return (cos(self.theta), sin(self.theta))

  def forward(self, d):
    g = np.dot(self.rigid_trans_matrix(), np.array([d,0.,1.]))
    return Pose(g[0],g[1], self.theta)


class Odometry(object):
  __slots__ = ['left','right']

  def __init__(self, left, right):
    self.left, self.right = left, right

  def __sub__(self, other):
    try:
      return Odometry(self.left - other.left, self.right - other.right)
    except:
      return NotImplemented

  def __add__(self, other):
    try:
      return Odometry(self.left + other.left, self.right + other.right)
    except:
      return NotImplemented

class CompassMap(object):
  __slots__ = ['cmap', 'scale']

  def __init__(self, scale = 10, size = (100,100)):
    self.scale = float(scale)
    self.cmap = np.zeros(size)

  def __getitem__(self, pose):
    try:
      x = pose.x / self.scale + self.cmap.shape[0]
      y = pose.y / self.scale + self.cmap.shape[1]

      return map_coordinates(self.cmap, [[x],[y]])

    except Exception as e:
      raise IndexError(e)

class ColorMap(object):
  pass

class USMap(np.ndarray):

  def __new__(cls, shape = (70,70)):
    return super(USMap, cls).__new__(cls,shape)

  def __init__(self):
    super(USMap, self).__init__()
    self.scale = 1.0
    self.fill(0.0)
      
