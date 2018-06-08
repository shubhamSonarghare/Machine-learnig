import numpy as np
import matplotlib.pyplot as plt
from svmutil import *

class SVM():
	def __init__(self,svm_type = 0 ,kernel = 'rbf', degree = 3,
				 gamma = 'auto', coef0 = 0.0, C = 1,
				 epsilon = 'auto', shrinking = True, probability=False, weight = 1.0):
		self.param = svm_parameter('-q')
		self.param.svm_type = svm_type
		kernel_type = {'linear':0, 'poly':1, 'rbf':2, 'sigmoid':3, 'precomputed':4} 
		self.param.kernel_type = kernel_type[kernel]
		self.param.degree = degree
		self.param.coef0 = coef0
		self.param.C = C
		self.param.shrinking =shrinking
		self.param.probability_estimates = probability
		#self.param.weight = weight
 		if epsilon != 'auto':
			self.param.epsilon = epsilon
		if gamma != 'auto':
			self.param.gamma = gamma

	def fit(self, X, y):
		self.X = X
		self.y = y
		if type(X)== type(np.array([])):
			self.X = X.tolist()
		if type(y)== type(np.array([])):
			self.y = y.tolist()	
		# create a problem instance
		self.prob = svm_problem(self.y, self.X)
		self.model = svm_train(self.prob, self.param)

	def get_nr_class(self):
		return self.model.get_nr_class()

	def get_svr_probability(self):
		return self.model.get_svr_probability()

	def get_labels(self):
		return self.model.get_labels()

	def get_sv_indices(self):
		return self.model.get_sv_indices()

	def get_nr_sv(self):
		return self.model.get_nr_sv()

	def  is_probability_model(self):
		return self.model.is_probability_model()

	def get_sv_coef(self):
		return self.model.get_sv_coef()

	def get_SV(self):
		return self.model.get_SV()

	def predict(self, X, y = None, print_accuracy = False):
		if type(X) == type(np.array([])):
			X = X.tolist()
		if print_accuracy == True:
			option = ''
		else:
			option = 'q'
		if y==None:
			y = [0]*len(X)
		elif type(y) == type(np.array([])):
			y = y.tolist()
		p_label, p_acc, p_vals= svm_predict(y, X, self.model, '-q')
		return p_label 
		
	def decision_functions (self, X):
		if type(X) == type(np.array([])):
			X = X.tolist()
		y = [0]*len(X)
		p_label, p_acc, p_vals = svm_predict(y, X, self.model, '-q')
		return p_vals

	def visualize_linear_hyperplane(self, X, y):
		X = np.array(X)
		y = np.array(y)
		if self.model.get_svm_type() == 0:
			pos = np.where(y==1)
			neg = np.where(y==0)
			val = [self.model.get_SV()[z].values() for z in range(len(self.model.get_SV()))]
			sv = np.array(val)[:,0:2]
			coeff = np.array(map(list,self.model.get_sv_coef())).T
			W = np.dot(sv.T,coeff.T)
			slope = -float(W[0])/float(W[1])
			b = -self.model.rho[0]

			xx = np.linspace(0,X.max())
			yy = slope * xx - (b / float(W[1]))

			h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

			plt.scatter(X[pos,0],X[pos,1],c= "Black", marker = "+")
			plt.scatter(X[neg,0],X[neg,1],c= "Yellow", edgecolors = "Black")
			plt.show(block=False)
			raw_input("Press any key to continue..")
			plt.close("all")

		else:
			print "Not a linear model!!!!"

	def visualize_non_linear_hyperplane(self,X, y, ax = None):
		if ax is None:
			ax = plt.gca()
		pos = np.where(y==1)
		neg = np.where(y==0)
		plt.scatter(X[pos,0],X[pos,1],c= "Black", marker = "+")
		plt.scatter(X[neg,0],X[neg,1],c= "Yellow", edgecolors = "Black")
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		x_test = np.linspace(xlim[0], xlim[1], 100)
		y_test = np.linspace(ylim[0], ylim[1], 100)
		Y_test, X_test = np.meshgrid(y_test, x_test)
		x_test_sim = np.c_[X_test.ravel(), Y_test.ravel()]
		y_test_sim= [0]*len(x_test_sim)
		P = self.decision_functions(x_test_sim)
		P = np.array(P)
		P = P.reshape(X_test.shape)
		ax.contour(X_test, Y_test, P, colors='Blue', levels=[0], alpha=0.5, linestyles=['-'])
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		plt.show(block=False)
		raw_input("Press any key to continue..")
		plt.close("all")