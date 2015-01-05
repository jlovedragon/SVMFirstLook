'''
===============================
A Test model about Perceptron
===============================

Perceptron model with different order

We can see the Perceptron will get different decision boundary

'''

print(__doc__)

#########################################################
# author   : Quentin_Hsu                                #
# email    : jlove.dragon@gmail.com                     #
# date     : 2014.09.17                                 #
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm

def createDataSet():
	# we create 40 separable points
	np.random.seed(0)
	X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
	Y = [0] * 50 + [1] * 50

	# X[9] = [-1, 10]

	return X, Y

def fitModel(clff, X, Y):
	# clff.fit(X, Y, coef_init=[coefParam,coefParam],intercept_init=coefParam)
	clff.fit(X, Y)
	w = clff.coef_[0]

	print w

	a = -w[0] / w[1]
	xx = np.linspace(-5, 5)
	yy = a * xx - (clff.intercept_[0]) / w[1]
	return xx, yy, a


def test():
	X, Y = createDataSet()

	h = .02  # step size in the mesh

	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# shuffle the data to get the different order
	clf1 = linear_model.Perceptron(warm_start=False,random_state=0, shuffle=True)

	clf2 = linear_model.Perceptron(warm_start=False,random_state=1, shuffle=True)
	clf3 = linear_model.Perceptron(warm_start=False,random_state=2, shuffle=True)

	# fit the model with different classifier
	xx1, yy1, a1 = fitModel(clf1, X, Y)
	xx2, yy2, a2 = fitModel(clf2, X, Y)
	xx3, yy3, a3 = fitModel(clf3, X, Y)

	# plot the data node
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, label='data')

	# plot the decision boundary
	plt.plot(xx1, yy1, 'k-',  label='perceptron1')
	plt.plot(xx2, yy2, 'b-',  label='perceptron2')
	plt.plot(xx3, yy3, 'r-',  label='perceptron3')

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())

	# a very useful tools: show label
	plt.legend()

	plt.title("The same data with different Order")
	plt.show()


# python plot_perceptron_test2.py
if __name__ == '__main__':
    test()

# import plot_perceptron_test2 or reload(plot_perceptron_test2)
if __name__ == 'plot_perceptron_test2':
    print 'Testing......'
    test()



