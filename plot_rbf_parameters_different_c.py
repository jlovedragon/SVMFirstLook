'''
====================
RBF SVM parameters C
====================

This example illustrates the effect of the parameters `gamma`
and `C` of the rbf kernel SVM.

Intuitively, the `gamma` parameter defines how far the influence
of a single training example reaches, with low values meaning 'far'
and high values meaning 'close'.
The `C` parameter trades off misclassification of training examples
against simplicity of the decision surface. A low C makes
the decision surface smooth, while a high C aims at classifying
all training examples correctly.

'''
print(__doc__)

#########################################################
# author   : Quentin_Hsu                                #
# email    : jlove.dragon@gmail.com                     #
# date     : 2014.09.17                                 #
#########################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

##############################################################################
def test():
    # Load and prepare data set
    #
    # dataset for grid search
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # dataset for decision function visualization
    X_2d = X[:, 2:4]
    X_2d = X_2d[Y > 0]
    Y_2d = Y[Y > 0]
    Y_2d -= 1

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)

    # Now we need to fit a classifier for all parameters in the 2d version
    # (we use a smaller set of parameters here because it takes a while to train)
    C_2d_range = [1, 100, 10000]
    gamma_2d_range = [0.001, 0.1, 10]
    classifiers = []
    # for C in C_2d_range:
    #     for gamma in gamma_2d_range:
    #         clf = SVC(C=C, gamma=gamma)
    #         clf.fit(X_2d, Y_2d)
    #         classifiers.append((C, gamma, clf))

    for C in C_2d_range:
        clf = SVC(C=C, gamma=0.1)
        clf.fit(X_2d, Y_2d)
        classifiers.append((C, 0.1, clf))

    ##############################################################################
    # visualization
    #
    # draw visualization of parameter effects
    plt.figure(figsize=(16, 6))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(1, 3, k + 1)

        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.title("C 10^%d" % (np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        # plt.pcolormesh(xx, yy, Z, cmap=plt.cm.jet)

        # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        plt.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.Paired)
        plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=-Y_2d, cmap=plt.cm.Paired)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')


    plt.show()

# python plot_rbf_parameters_different_c.py
if __name__ == '__main__':
    test()

# import plot_rbf_parameters_different_c or reload(plot_rbf_parameters_different_c)
if __name__ == 'plot_rbf_parameters_different_c':
    print 'Testing......'
    test()

