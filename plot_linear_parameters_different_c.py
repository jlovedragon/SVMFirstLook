'''
===============================================
linear kernel of svm with different parameter c
===============================================

choose different c will have different decision boundary?
really?

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

def getClf(cParam):
    clf = svm.SVC(kernel='linear', C=cParam)
    return clf


def fitModel(clff, X, Y):
	clff.fit(X, Y)
	w = clff.coef_[0]
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

    svm_clf1 = getClf(0.1)
    svm_clf2 = getClf(1)
    svm_clf3 = getClf(10)
    svm_clf4 = getClf(1000)

    # title for the plots
    titles = ['SVM (C=0.1, no noise)',
              'SVM (C=1, no noise)',
              'SVM (C=10, no noise)',
              'SVM (C=1000, no noise)']

    for i, clf in enumerate((svm_clf1, svm_clf2, svm_clf3, svm_clf4)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        svm_xx, svm_yy, svm_a = fitModel(clf, X, Y)

        b =clf.support_vectors_[0]
        yy_down = svm_a * svm_xx + (b[1] - svm_a * b[0])
        b =clf.support_vectors_[-1]
        yy_up = svm_a * svm_xx + (b[1] - svm_a * b[0])

        # Plot also the training points
        plt.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1],
                s=80, facecolors='none')
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, label='data')

        plt.hold('on')

        plt.plot(svm_xx, svm_yy, 'r-', label='SVM')
        plt.plot(svm_xx, yy_down, 'r--', label='SVM Margin')
        plt.plot(svm_xx, yy_up, 'r--', label='SVM Margin')

        plt.xlabel(i + 1)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # plt.legend()
        plt.title(titles[i])

    plt.show()

# python plot_linear_parameters_different_c.py
if __name__ == '__main__':
    test()

# import plot_linear_parameters_different_c or reload(plot_linear_parameters_different_c)
if __name__ == 'plot_linear_parameters_different_c':
    print 'Testing......'
    test()

