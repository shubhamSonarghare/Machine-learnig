import numpy as np
import functionsBP as BP
import scipy.io as spio

mat = spio.loadmat('MNIST-subset.mat',squeeze_me=True)
#mat2 = spio.loadmat('ex4weights.mat',squeeze_me=True)

#print mat
X=mat['X']
y=mat['y']


[Xtrain, ytrain, Xcv, ycv, Xtest, ytest] = BP.splitDataSets(X, y, seperation = (90,0,10), shuffle =1)

print "Xtrain shape: ", Xtrain.shape
print "ytrain shape: ", ytrain.shape
print "Xcv shape: ", Xcv.shape
print "Ycv shape: ", ycv.shape
print "Xtest shape: ", Xtest.shape
print "ytest shape: ", ytest.shape

lamda = 0

[Cost , Theta1, Theta2] = BP.fit(lamda, Xtrain, ytrain, 400, 25, 10)

print "\nCost: ", Cost

print BP.test(Xtest[0], ytest[0], Theta1, Theta2), ytest[0]

