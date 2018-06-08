import numpy as np
import scipy.io as spio
from SVM import *
mat = spio.loadmat('ex_data_frm_coursera_non_linear.mat',squeeze_me=True)

X=mat['X']
y=mat['y']

sigma = 0.1
gamma = 1/(2*(sigma**2))

clf = SVM(kernel= 'rbf',gamma = gamma, C = 1.0)
clf.fit(X ,y)
clf.visualize_non_linear_hyperplane(X,y)
