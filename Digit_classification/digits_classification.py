import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from perceptron_learning import *

def readData(filename):
  ss = ""
  tt = ""
  ff = []
  with open(filename) as f:
    for line in f:
      if (re.search('-', line) != None):
        s = re.findall('[a-z]+|\d+', line)
        ss += " " + s[1]
        #if (tt != ""):
        #  feature.append(tt)
        #tt = ""
      elif (re.search('\d*[.]\d+', line) != None):
      #elif (re.search('[.]', line) != None):
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


#######################################
######### main function ###############
def main():
  (X,Y) = readData("digits_train.txt")
  X = np.insert(X, 0, 1, axis = 1) # insert one as bias term
  (testX, testY) = readData("digits_test.txt")
  testX = np.insert(testX, 0, 1, axis = 1)

#### part 2: train a perceptron to classify 2 and not 2 ####
  ind2 = np.nonzero(Y==2)[0] # np.nonzero returns a tuple, [0] gets the array
  y    = np.zeros((Y.size, 1))
  y[ind2] = 1.0

  ind2  = np.nonzero(testY==2)[0]
  testy = np.zeros((testY.size, 1))
  testy[ind2] = 1.0

  [w, err] = gradient_descent(X, y, learning_rate = 0.1, maxiter = 50, 
                              fun_eval = perceptron_learning, batch_size = 1) 
  pred = singleclass_classification(X, w)
  acc  = compute_accuracy(pred, y)
  print "=================================================="
  print "Part 2: train a perceptron to classify 2 and not 2:"
  print "train set accuracy %f" %acc
  pred = singleclass_classification(testX, w)
  acc  = compute_accuracy(pred, testy)
  print "test set accuracy %f" %acc
  plot_confusionmatrix(testy, pred, ['2','not 2'], 'part2.png')

#### part 3: classify 8 from 0 ####
  ind8 = np.nonzero(Y==8)[0]
  ind0 = np.nonzero(Y==0)[0]
  ind08 = np.concatenate((ind8,ind0))
  x = X[ind08]
  y = np.zeros((len(ind8) + len(ind0), 1))
  y[len(ind8)] = 1.0

  ind8  = np.nonzero(testY==8)[0]
  ind0  = np.nonzero(testY==0)[0]
  testx = testX[ind08,:]
  testy = np.zeros((len(ind8) + len(ind0), 1))
  testy[len(ind8)] = 1.0

  [w, err] = gradient_descent(x, y, learning_rate = 0.1, maxiter = 50, 
                fun_eval = perceptron_learning, batch_size = 1) 
  pred = singleclass_classification(x, w)
  acc  = compute_accuracy(pred, y)
  print "=================================================="
  print "Part 3: train a perceptron to classify 8 from 0:"
  print "train set accuracy %f" %acc
  pred = singleclass_classification(testx, w)
  acc  = compute_accuracy(pred, testy)
  print "test set accuracy %f" %acc
  plot_confusionmatrix(testy, pred, ['8','0'], 'part3.png')

#### part 4: classify between all digits 0-9 ####
  # rearrange y to a 2500*10 matrix, each col corresponds to a digits
  digits = np.unique(Y)
  digits_len = digits.size
  yy = np.zeros((len(Y), digits_len))
  for i in xrange(len(Y)):
    yy[i,Y[i]] = 1.0 

  testyy = np.zeros((len(testY), digits_len))  
  for i in xrange(len(testY)):
    testyy[i,testY[i]] = 1.0

  [w, err] = gradient_descent(X, yy, learning_rate = 0.1, maxiter = 50, 
                              fun_eval = multiclass_perceptron, batch_size = 1) 
  pred = multiclass_classification(X, w)
  acc  = compute_accuracy(pred, yy)
  print "=================================================="
  print "Part 4: train a perceptron to classify between all digits 0-9:"
  print "train set accuracy %f" %acc
  pred = multiclass_classification(testX, w)
  acc  = compute_accuracy(pred, testyy)
  print "test set accuracy %f" %acc

  predlabel = np.nonzero(pred)[1]
  labels = ["%d" %i for i in xrange(digits_len)]
  plot_confusionmatrix(testY, predlabel, labels, 'part4.png')

######################################
# run main() as the main function
if __name__ == '__main__':
  main()
