import numpy as np
import matplotlib.pyplot as plt
import time

#from simplified_smo import SVM#, OneVsAllClassifier
from smo import SVM#, OneVsAllClassifier
from sklearn.svm import SVC
import math
import random
from numpy import linalg as LA
from my_smo import MY_SVM

def sdf(x, y):
    return (x-3.5)*(x-3.5) + (y-3.5)*(y-3.5) - 0.25 - random.uniform(-0.1, 0.1)

error_C = 1000
dataset_size = 200
dataset_x = [[random.uniform(2, 5), random.uniform(2, 5)] for i in range(dataset_size)]
dataset_y = []

for i in range(len(dataset_x)):
    if sdf(dataset_x[i][0], dataset_x[i][1]) > 0:
        dataset_y.append(-1)
    else:
        dataset_y.append(1)

blue = [dataset_x[i] for i in range(len(dataset_x)) if dataset_y[i] == -1]
red = [dataset_x[i] for i in range(len(dataset_x)) if dataset_y[i] == 1]

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
print("Dataset size", len(dataset_y))
blue_x, blue_y = zip(*blue)
red_x, red_y = zip(*red)
gamma = 1.0
print("gamma", gamma)

for error_C in [10, 100, 1000, 2000]:
    print("C", error_C)
    #solver = SVM(kernel_type="rbf", c=error_C, gamma_rbf=gamma)
    solver2 = SVC(gamma=gamma, C=error_C)

    model = MY_SVM(C=error_C, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=1e-5, max_iter=1e4)

    #start_time = time.time()
    #solver.fit(dataset_x, dataset_y)
    #end_time = time.time()

    libsvm_start_time = time.time()
    solver2.fit(dataset_x, dataset_y)
    libsvm_end_time = time.time()
    #print("Platt implementation", "time %s" % (end_time - start_time))
    print("LibSVM", "time %s" % (libsvm_end_time - libsvm_start_time))
    
    start_time = time.time()
    model.fit_modification1(dataset_x, dataset_y)
    end_time = time.time()
    print("Keerthi 1 implementation", "time %s" % (end_time - start_time), "w:", model.get_w(), "b:", -model.get_b())

    count = 0
    for i in range(len(solver2.dual_coef_)):
        if abs(solver2.dual_coef_[0][i]) == error_C:
            count += 1

    print("LibSVM support_vectors", len(solver2.support_vectors_), "bound", count)

    #count = 0
    #for i in range(len(solver.alpha)):
     #   if abs(solver.alpha[i]) == error_C:
     #       count += 1
    
   # print("Platt support_vectors", len(solver.support_vectors), "bound", count)


    print("Keerthi 1 support_vectors", len(model.support_vectors), "bound", len(model.bounded))

    #print("Keerthi 2 support_vectors", len(model2.support_vectors), "bound", len(model2.bounded))

    plt.figure()
    xx, yy = np.meshgrid(np.arange(2, 5, 0.02),
                    np.arange(2, 5, 0.02))

    bg1 = model.predict(np.c_[xx.ravel(), yy.ravel()])
    bg1 = bg1.reshape(xx.shape)
    dist = model.decision_function(np.c_[xx.ravel(), yy.ravel()]) 
    dist = dist.reshape(xx.shape)

    #        elif bg1[i][j] > 0:
    #            bg1[i][j] = 1000
    #        else:
    #            bg1[i][j] = -1000

    plt.contourf(xx, yy, bg1, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.contour(xx, yy, np.sign(dist + 1), colors="#222222", alpha=0.3, linewidths=1, label='supporting hyperplane')
    plt.contour(xx, yy, np.sign(dist), colors="#222222", alpha=0.3, linewidths=1, label='dividing hyperplane')
    plt.contour(xx, yy, np.sign(dist - 1), colors="#222222", alpha=0.3, linewidths=1, label='supporting hyperplane')


    blue_x, blue_y = zip(*blue)
    red_x, red_y = zip(*red)
    plt.scatter(blue_x, blue_y, c="blue")
    plt.scatter(red_x, red_y, c="red")

    green_x, green_y = zip(*model.support_vectors)
    plt.scatter(green_x, green_y, c="black", marker="x", label='support vector')

    plt.xlim(left=2, right=5)
    plt.ylim(bottom=2, top=5)

plt.show()

