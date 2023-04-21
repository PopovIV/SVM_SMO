import pandas as pd
import numpy as np
import pylab as pl

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from smo import SVM, OneVsAllClassifier
from sklearn import svm

from timeit import default_timer as timer

# Generate linearly separable data
def generate_linearly_separable_data(size):
    allData = -2.5 + np.random.uniform(0, 1, size * 2) * 5
    allData = allData.reshape((size, 2)).tolist()
    classes = []
    for d in allData:
        if d[1] >= 2.5 * d[0] + 1.5:
            classes.append(0)
        else:
            classes.append(1)
    return allData, classes

def test_svmlib_linear(X, Y):
    # fit the model
    clf = svm.SVC(kernel='linear', C=3)
    print("LibSvm")
    start = timer()
    clf.fit(X, Y)
    end = timer()
    print("Time")
    print(end - start)
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-2.5, 2.5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    pl.xlim([-2.5, 2.5])
    pl.ylim([-2.5, 2.5])
    pl.set_cmap(pl.cm.Paired)
    pl.plot(xx, yy, 'k-')
    #pl.plot(xx, yy_down, 'k--')
    #pl.plot(xx, yy_up, 'k--')
    
    # plot points
    for i in range(len(X)):
        if (Y[i] == 1):
            color = 'r'
        else:
            color = 'b'
        pl.plot(X[i][0], X[i][1], c = color, marker="o")

    # plot support vectors
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, marker='x', color='black')
    pl.show()

def test_platt_linear(X, Y):
    solver = OneVsAllClassifier(
        solver=SVM,
        num_classes=2,
        c=3.0,
        kkt_thr=1e-3,
        max_iter=1e3,
        version='platt',
    )
    # plot points
    for i in range(len(X)):
        if (Y[i] == 1):
            color = 'r'
        else:
            color = 'b'
        pl.plot(X[i][0], X[i][1], c = color, marker="o")

    start = timer()
    solver.fit(X, Y)
    end = timer()
    print("Time")
    print(end - start)
    solver.plot()
    pl.show()

def test_kerthi1_linear(X, Y):
    solver = OneVsAllClassifier(
        solver=SVM,
        num_classes=2,
        c=3.0,
        kkt_thr=1e-3,
        max_iter=1e3,
        version='keerthi1',
    )
    # plot points
    for i in range(len(X)):
        if (Y[i] == 1):
            color = 'r'
        else:
            color = 'b'
        pl.plot(X[i][0], X[i][1], c = color, marker="o")

    start = timer()
    solver.fit(X, Y)
    end = timer()
    print("Time")
    print(end - start)
    solver.plot()
    pl.show()

X, Y = generate_linearly_separable_data(100)

test_svmlib_linear(X, Y)
test_platt_linear(np.array(X), np.array(Y))
test_kerthi1_linear(np.array(X), np.array(Y))
