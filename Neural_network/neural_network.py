import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
#from hw1 import *
#from hw2 import *
from func import *

def readData(filename):
  ss = ""
  tt = ""
  ff = []
  with open(filename) as f:
    for line in f:
      if (re.search('-', line) != None):
        s = re.findall('[a-z]+|\d+', line)
        ss += " " + s[1]
      elif (re.search('\d*[.]\d+', line) != None):
        tt +=  line.rstrip() + " "
      else:
        ff.append(tt)
        tt = ""

  l = len(ff[0].rstrip().split(" "))
  feature = np.ndarray(shape=(len(ff),l), dtype = float)
  for i in xrange(0, len(ff)):
    feature[i,:] = np.asarray(ff[i].rstrip().split(" "))

  label = np.fromstring(ss,int,sep=' ')     
  return (feature, label)
         
##############################################
def plot_confusionmatrix(test, pred, alphabet, filename = "confusion_matrix.png"):
  conf = confusion_matrix(test, pred) 
  norm_conf = []
  for i in conf:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
      tmp_arr.append(float(j) / float(a))
    norm_conf.append(tmp_arr)

  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(norm_conf), cmap = plt.cm.jet,
                  interpolation = 'nearest')
  width = len(conf)
  height = len(conf[0])

  for x in xrange(width):
    for y in xrange(height):
      ax.annotate(str(conf[x][y]), xy = (y,x),
          horizontalalignment = 'center',
          verticalalignment = 'center')

  cb = fig.colorbar(res)
#  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  plt.xticks(range(width),alphabet[:width])
  plt.yticks(range(height),alphabet[:height])
  plt.savefig(filename, format = 'png')

##############################################
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

###############################################################
def train_neural_network(x, y, layer_info, learning_rate, alpha, hidden_func, output_func, cost_func): 
# x: matrix, num_samples * num_features, e.g. 2500 * 196 
# y: matrix, num_samples * num_labels, e.g. 2500 * 10 (0-9 digits)
# layer_info: list of dictionaries
# hidden_func: sigmoid or tanh
# output_func: sigmoid or softmax
# cost_func: square_error or cross_entropy

## math ##
### grad_E/grad_theta = (grad_E/grad_O) * (grad_O/grad_net) * (grad_net/grad_theta)
### net: x*theta; O: output from hidden layer (apply sigmoid functions)
### delta = (O-y)*grad_z; or np.dot(delta, theta)*grad_z
### grad_E/grad_theta = delta*O (or a)

  num_hidden_layers = len(layer_info) - 1
# compute forward propagation
  yhidden = x
  for i in xrange(num_hidden_layers):
    layer = layer_info[i]
    theta = layer['theta']
    a = np.insert(yhidden, 0, 1, axis = 1) 
    layer['a'] = a
    yhidden, h_prime = hidden_func(a, theta)
    layer['dhdz'] = h_prime

# output layer: cost and gradient
  a = np.insert(yhidden, 0, 1, axis = 1)
  layer_info[-1]['a'] = a
  theta = layer_info[-1]['theta']
  cost, gradz = cost_output(output_func, cost_func, theta, a, y)
  layer_info[-1]['delta'] = gradz

# back propogation
#  for i in xrange(num_hidden_layers-1, -1, -1):
  for i in reversed(xrange(num_hidden_layers)):  
    theta = layer_info[i+1]['theta']
    gradz = layer_info[i+1]['delta']
    h_prime = layer_info[i]['dhdz']
    grada = np.dot(gradz, theta[1:].T) # ignor bias term
    ### for next layer
    gradz = grada * h_prime 
    layer_info[i]['delta'] = gradz
  
# forward updating weights
  for layer in layer_info:
    a = layer['a']
    gradz = layer['delta']
    gradw = np.dot(a.T, gradz)
#    layer['grad'] = np.dot(a.T, gradz)
    try:
      layer['grad'] = alpha * layer['grad'] - (1.0 - alpha) * learning_rate * gradw
    except KeyError:
      layer['grad'] = -learning_rate * gradw

    layer['theta'] = layer['theta'] + layer['grad']

  return cost

###############################################
def nn_initialize_weights(layer_size, layer_info):
  epsilon = 1.0
  for i in xrange(len(layer_info)):
    m = layer_size[i]+1
    n = layer_size[i+1]
#    w = np.random.normal(0.0, epsilon, (m, n))
    w = np.random.random((layer_size[i]+1, layer_size[i+1])) * 2 * epsilon - epsilon
#    l1 = LA.norm(w, ord=1, axis=0) # row sum
#    layer_info[i]['theta'] = 2.0 * w/l1 # normalization
    layer_info[i]['theta'] = w
  
################################################
def cost_output(output_fun, cost_fun, w, x, t):
  if (output_fun == tanh and cost-fun == cross_entropy):
    raise Exception("Error: invalid combination (%s, %s)" %(output_func._name_, cost_fun._name))

  (y, grady) = output_fun(x, w)
  (cost, gradE) = cost_fun(t, y)
#  grad = grady * gradE
  grad = -(t - y)
  return (cost, grad)

################################################
def neural_network(x, y, layer_size, layer_info, hidden_func, output_func, cost_func, learning_rate = 0.1, alpha = 0.5, maxiter = 100, tol = 1e-5, initial_guess = None, batch_size = None):
  
  if (initial_guess == None):
    nn_initialize_weights(layer_size, layer_info)
  else:
    theta = initial_guess

  if (batch_size == None):
    batch_size = x.shape[0]

  err = []
  indices = range(x.shape[0])

  for i in xrange(maxiter):
    random.shuffle(indices)
    for j in range(0, x.shape[0], batch_size):
      batch = indices[j:j+batch_size]
      cost  = train_neural_network(x[batch], y[batch], layer_info, learning_rate, alpha, hidden_func, output_func, cost_func)

    err.append(cost)
    if (cost < tol):
      break

  # retrieve weights from layer_info for output
  theta = [layer['theta'] for layer in layer_info]
  return (theta, err)

################################################
def nn_predict(x, theta, hidden_func, output_func):
  yhidden = x
  for i in xrange(len(theta)-1):
    yhidden = np.insert(yhidden, 0, 1, axis = 1)
    yhidden = hidden_func(yhidden, theta[i])[0]

  yhidden = np.insert(yhidden, 0, 1, axis = 1)
  output = output_func(yhidden, theta[-1])[0]
  maxind = np.argmax(output, axis = 1)
  labels = np.ravel(maxind) # flatten the array, this is for softmax

  return labels

################################################
def nn_preprocess_label(y, output_func, cost_fun):
  # cost_fun: cross_entropy or least_square_error
  # output_fun: softmax or sigmoid
  digits = np.unique(y)
  digits_len = digits.size
  zz = np.zeros((len(y), digits_len))
  for i in xrange(len(y)):
    zz[i, y[i]] = 1.0

  if (cost_fun == cross_entropy):
    if (output_func == tanh):
      raise Exception("Error: invalid combination (%s, %s)" %(output_func._name_, cost_fun._name))
    else: # for sigmoid or softmax
      yy = zz

  else: # cost_func = least_square_error
    if (output_func == tanh):
      yy = np.empty((len(y), digits_len))
      yy.fill(-1.0)
      for i in xrange(len(y)):
        yy[i, y[i]] = 1.0

    elif (output_func == softmax):
      yy = np.exp(zz)
      norm = np.sum(yy, axis = 1)
      yy = yy/norm[:,np.newaxis]

    else: # output_func = sigmoid
      yy = zz

  return yy     

#############  main function ###########
def main():
  (X,Y) = readData("digits_train.txt")
  (testX, testY) = readData("digits_test.txt")

  digits = np.unique(Y)
  digits_len = digits.size

##################### Neural network classification ###################
  layer_size = [196, 75, 25, 10]  
  num_layers = len(layer_size)-1
  hidden_func = sigmoid
  output_func = softmax 
  cost_func   = cross_entropy
  yy = nn_preprocess_label(Y, output_func, cost_func)
  testyy = nn_preprocess_label(testY, output_func, cost_func)

  layer_info = map(lambda i:{}, xrange(num_layers))

# when do full batch, learning_rate = 0.001, alpha = 0.0005
  (theta, err) = neural_network(X, yy, layer_size, layer_info, hidden_func, output_func, cost_func, learning_rate = 0.001, alpha = 0.0005, maxiter = 100) 
  pred = nn_predict(X, theta, hidden_func, output_func)
  acc = compute_accuracy(pred, Y)
  print "train set accuracy %f" %acc
  pred = nn_predict(testX, theta, hidden_func, output_func)
  acc = compute_accuracy(pred, testY)
  print "test set accuracy %f" %acc

  labels = ["%d" %i for i in xrange(digits_len)]
  plot_confusionmatrix(testY, pred, labels, 'confusion_matrix.png')

#####################################
# run main() as the main function
if __name__ == '__main__':
  main()
