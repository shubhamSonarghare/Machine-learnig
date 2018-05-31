import numpy as np
import math
from  bigfloat import *
from scipy.special import expit
from scipy.optimize import minimize
import decimal

#decimal.getcontext().prec = 10


def sigmoid(z):
    g = 1.0/(1.0 + np.exp(-z))
    return g

def sigmoid_grad(z):
    grad = sigmoid(z)*(1-sigmoid(z))
    return grad

def randomInitWeights(L_in,L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out,L_in+1)*2*epsilon_init - epsilon_init
    return W

def nncostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    m= float(X.shape[0])
    y.reshape((len(y),1))
    X = np.append(np.ones((int(m),1)),X,axis=1)
    Theta1 = nn_params[0:int(hidden_layer_size*(input_layer_size+1))]
    Theta1 = Theta1.reshape((hidden_layer_size,(input_layer_size+1)),order = 'F')
    Theta2 = nn_params[int(hidden_layer_size*(input_layer_size+1)):]
    Theta2 = Theta2.reshape((num_labels,(hidden_layer_size+1)),order = 'F')
    delta_1_final = 0
    delta_2_final = 0

    Yformat = np.zeros((int(y.max(),)))
    J = float(0)
    cnt = 0
    for i in range(int(m)):
        a1 = X[i,:]
        z2 = np.dot(a1,Theta1.T)
        a2 = sigmoid(z2)
        a2 = a2.reshape((1,len(a2)))
        a2 = np.append(np.ones((a2.shape[0],1)),a2)
        z3 = np.dot(a2, Theta2.T)
        a3 = sigmoid(z3)
        a3 = a3.reshape((1,len(a3)))
        Ycurr = Yformat.copy()
        Ycurr[y[i]-1] = 1
        Ycurr = Ycurr.reshape((1,len(Ycurr)))
        J = J + float(((-1)*np.dot(Ycurr,np.log(a3).T)) - (np.dot((1-Ycurr),np.log(1-a3).T)))
        ##################gradient cal part #####################
        delta_3 = a3 - Ycurr
        delta_2 = np.dot(delta_3,Theta2[:,1:])*sigmoid_grad(z2)
        delta_2_final = delta_2_final + (a2.reshape((len(a2),1))*delta_3)
        delta_1_final = delta_1_final + (a1.reshape((len(a1),1))*delta_2) 

        #while(1):
         #   print delta_1_final.shape
         #   print delta_2_final.shape
         #   break
        #break
    J = (1/m)*J 
    sqTheta1= np.sum(Theta1[:,1:] ** 2)
    sqTheta2 = np.sum(Theta2[:,1:] ** 2)
    RegJ = (lamda/(2*m))*(sqTheta2+sqTheta1)
    Jreg = J + RegJ
    th1,th2 = np.copy(Theta1), np.copy(Theta2)
    th1[:,0], th2[:,0] = 0, 0
    Theta1_Reg = (lamda / m)*th1
    Theta2_Reg = (lamda / m)*th2
    Theta1_grad = (1/m)*(delta_1_final) + Theta1_Reg.T
    Theta2_grad = (1/m)*(delta_2_final) + Theta2_Reg.T
    grad = np.array([Theta1_grad[:].ravel(),Theta2_grad[:].ravel()])
    grad = np.hstack(grad)
    return Jreg,grad
    
def createRandomdata(f_in, f_out):
    W = np.zeros((f_out,1+f_in))
    val = np.sin(range(1,W.size+1)).T
    W = val.reshape((W.shape),order = 'F')/10
    return W

def checkNNgradient(lamda):
    #lamda = 0
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = createRandomdata(input_layer_size, hidden_layer_size)
    Theta2 = createRandomdata(hidden_layer_size, num_labels)
    X = createRandomdata(input_layer_size -1,m)
    y= 1+np.mod(range(1,m+1), num_labels).T
    nn_params = np.array([Theta1[:].ravel(order='F'),Theta2[:].ravel(order='F')])
    nn_params = np.hstack(nn_params)
    costfunc = lambda g : nncostFunction(g,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
    J,grad = costfunc(nn_params)
    print "################################"
    numgrad = computeNumericalGradient(costfunc, nn_params)
    print "Comparison \n", "numgrad: ", numgrad ,"\n grad", grad
    

def computeNumericalGradient(costfun, theta):
    numgrad = np.zeros((theta.shape))
    chng = np.zeros((theta.shape))
    e = 0.0001
    for i in range(len(theta)):
        chng[i] = e
        loss1,w1 = costfun(theta - chng)
        loss2,w2 = costfun(theta + chng)
        numgrad[i]= (loss2-loss1)/(2*e)
        chng[i] = 0
    return numgrad

def callback(xk):
    print ".",    

'''
def train(X, y, lamda, maxiter):
    m,n = X.shape
    X = np.append(np.ones((int(m),1),dtype = np.int),X,axis=1)
    initial_theta = np.zeros((n+1,1))
    print "Xshape:::", X.shape
    print "Theta Shape: ", initial_theta.shape
    options={'maxiter' : maxiter,'disp' : True}
    res = minimize(fun = costFunction , x0=initial_theta, args = (X,y,lamda), method = 'CG', jac = True, options = options, callback=callback(maxiter))
    th = res.x 
    return th
 '''  
def predictBP (Theta1, Theta2, X):
    if len(X.shape) == 1:
        X = X.reshape((1,X.shape[0]))
    m= float(X.shape[0])
    print "XshapeXX::: " , X.shape
    X = np.append(np.ones((int(m),1)),X,axis=1)
    z2 = np.dot(X,Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((a2.shape[0],1)),a2,axis=1)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    return a3.argmax(axis = 1)+1


def splitDataSets(X, y, shuffle = 0,seperation = (60,20,20) ):
    train_percentage, cv_percentage, test_percentage = seperation
    #input data (X) should be in format "training examples x example". 
    # For example: if there is a dataset in form of images, having 1000 image examples where each image size is 10 px x 10 px, then shape of input matrix X should be 1000 x 100
    train_size = (X.shape[0]*train_percentage)/100
    cv_size = (X.shape[0]*cv_percentage)/100
    test_size = (X.shape[0]*test_percentage)/100
    if shuffle == 1:
        print "Shuffling"
        data = np.append(X,y.reshape((y.shape[0],1)), axis = 1 )
        np.random.shuffle(data)
        X = data[:,0:-1]
        y = data[:,-1]
        y = np.asarray(y, dtype = int)

    Xtrain = X[0:train_size,:]
    Xcv = X[train_size:train_size+cv_size,:]
    Xtest = X[train_size+cv_size : train_size+cv_size+test_size , :]

    ytrain = y[0:train_size]
    ycv = y[train_size:train_size+cv_size]
    ytest = y[train_size+cv_size : train_size+cv_size+test_size]

    return [Xtrain, ytrain, Xcv, ycv, Xtest, ytest]


def fit(lamda , Xtrain, Ytrain, input_layer_size, hidden_layer_size, num_labels, maxiter = 50):
    initial_Theta1 = randomInitWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randomInitWeights(hidden_layer_size, num_labels)

    initial_nn_params = np.array([initial_Theta1[:].ravel(order= 'F'),initial_Theta2[:].ravel(order = 'F')])
    initial_nn_params = np.hstack(initial_nn_params)
    print "Training",
    options={'maxiter' : maxiter,'disp' : False}
    costfunc = lambda x : nncostFunction(x,input_layer_size,hidden_layer_size,num_labels,Xtrain,Ytrain,lamda)
    nn_params_res = minimize(fun = costfunc , x0=initial_nn_params, method = 'CG', jac = True, options = options, callback=callback)

    Theta1 = nn_params_res.x[0:int(hidden_layer_size*(input_layer_size+1))]
    Theta1 = Theta1.reshape((hidden_layer_size,(input_layer_size+1)),order = 'F')
    Theta2 = nn_params_res.x[int(hidden_layer_size*(input_layer_size+1)):]
    Theta2 = Theta2.reshape((num_labels,(hidden_layer_size+1)),order = 'F')

    J,grad = costfunc(nn_params_res.x)
    return [J, Theta1, Theta2]

def test(Xtest, Ytest, Theta1, Theta2):
    pred = predictBP(Theta1, Theta2, Xtest)
    if len(Ytest)>1:
        print "Accuracy:::", np.mean(map(int, pred==Ytest))*100
    return pred
