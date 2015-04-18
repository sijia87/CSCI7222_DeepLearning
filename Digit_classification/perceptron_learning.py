# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import timeit
import random

def readData(filename):
  datset = pd.read_csv(filename, header = None, sep = r"\s+")
  return datset

def preProcess(data, portrion):
  data = data[2:].as_matrix()
  random.shuffle(data)
  data = data.astype(np.float)
  
  totrain = int(len(data) * portrion)
  totest  = int(len(data) * 0.75)
  X = np.c_[np.ones(len(data)), data[:,0:2]]

  trainX = X[0:totrain,:]
  testX  = X[totest:,:]
  trainY = data[0:totrain,2]
  testY  = data[totest:,2]
  trainZ = data[0:totrain,-1]
  testZ  = data[totest:,-1]

  return (trainX, testX, trainY, testY, trainZ, testZ)

def least_square_error(x, y, w):
  val  = np.dot(x,w) - y
  cost = np.vdot(val, val)
  return cost/len(y)

def singleclass_classification(x, w):
  res = np.dot(x,w) > 0.0
  return res

def multiclass_classification(x, w):
  xw = np.dot(x,w)
  maxind = np.argmax(xw, axis = 1)
  labels = np.zeros(xw.shape)
  for i, ind in enumerate(maxind):
    labels[i,ind] = 1.0
  return labels  

def linear_regression(w, x, y):
  coef = 1.0/len(y)
  val  = np.dot(x,w) - y
  cost = coef * np.vdot(val, val)
  grad = coef * 2.0 * np.dot(x.T,val)
  return (cost, grad)

def perceptron_learning(w, x, y, fun = singleclass_classification):
  val  = fun(x,w) - y
  cost = np.vdot(val, val)
  grad = np.dot(x.T, val)
  return (cost, grad)

def multiclass_perceptron(w, x, y):
  return perceptron_learning(w, x, y, fun = multiclass_classification)

def gradient_descent(x, y, learning_rate=0.01, maxiter=1000,
                      tol=1e-5, fun_eval=linear_regression, initial_guess=None,batch_size=None):
  if (initial_guess == None):
    w = np.zeros((x.shape[1], y.shape[1]))
  else:
    w = initial_guress

  if (batch_size == None):
    batch_size = x.shape[0]

  err = []
  indices = range(x.shape[0])

  for i in xrange(maxiter):
    random.shuffle(indices)
    for j in range(0, x.shape[0], batch_size):
      batch = indices[j:j+batch_size]
      (cost, grad) = fun_eval(w, x[batch], y[batch])
      w -= learning_rate * grad

    (cost, grad) = fun_eval(w, x, y)
    err.append(cost)
    if (cost < tol):
      break

  return (w, err)

def compute_accuracy(pred, true_sol):
  if (not np.array_equiv(pred.shape, true_sol.shape)):
    print "Error: predition and true solution have different size!"
    print pred.shape, true_sol.shape
    exit(1)

  if (pred.ndim == 1): #vec, shape(n,)
    acc = 1.0 - sum(pred != true_sol)/float(pred.shape[0])
  elif (pred.shape[1] == 1): #vec, shape(n,1)
    acc = 1.0 - sum(pred[:,0] != true_sol[:,0])/float(pred.shape[0])
  else: # pred is a matrix
    count = 0
    for i, row in enumerate(pred):
      if np.array_equiv(row, true_sol[i,:]):
        count += 1
    acc = count/float(pred.shape[0])

  return acc * 100.0
    
