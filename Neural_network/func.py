import numpy as np
import random

################
def tanh(x, w):
  net  = np.dot(x, w)
  y = 2.0/(1.0 + np.exp(-net)) - 1.0
  grad = 0.5 * (1.0 + y) * (1.0 - y) ## element-wise
  return (y, grad)

def sigmoid(x, w):
  net  = np.dot(x, w)
  y = 1.0/(1.0 + np.exp(-net))
  grad = y * (1.0 - y) ## element-wise
  return (y, grad)

def softmax(x, w):
  net  = np.dot(x, w)
  y    = np.exp(net)
  norm = np.sum(y, axis = 1)
  y = y/norm[:,np.newaxis]
  grad = y - y * y # only for the diagonal part
  #grad = y * (1 - y)
  return (y, grad)

def square_error(t, y):
# t: target
# y: y = h(wx)
  cost = 0.5 * np.vdot(t - y, t-y)
  grad  = -(t-y)
  return (cost, grad)

def cross_entropy(t, y):
  cost = -np.vdot(t, np.log(y))
  grad = -t/y ## element-wise
  return (cost, grad)

