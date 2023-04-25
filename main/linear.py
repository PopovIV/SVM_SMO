import numpy as np
import matplotlib.pyplot as plt
import time

#from simplified_smo import SVM#, OneVsAllClassifier
from smo import SVM#, OneVsAllClassifier
from sklearn.svm import SVC
import math
import random
from my_smo import MY_SVM


def sdf(x, y):
    a = 6
    b = -5
    c = -3
    return (a*x + b*y + c) / math.sqrt(a*a + b*b)

error_C = 1000
dataset_size = 100
dataset_x = [[random.uniform(4, 5), random.uniform(4, 5)] for i in range(dataset_size)]
dataset_y = []

for i in range(len(dataset_x)):
    if sdf(dataset_x[i][0], dataset_x[i][1]) > 0:
        dataset_y.append(-1)
    else:
        dataset_y.append(1)

blue = [dataset_x[i] for i in range(len(dataset_x)) if dataset_y[i] == 1]
red = [dataset_x[i] for i in range(len(dataset_x)) if dataset_y[i] == -1]

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
print("Dataset size", len(dataset_y))
blue_x, blue_y = zip(*blue)
red_x, red_y = zip(*red)


for error_C in [10, 100, 1000]:
    #dataset_x =[[-0.14800483, 0.48021052], [0.76849682, -0.24297545], [0.00734492, -0.86015146], [-0.26328409, -0.71402462]]
    #dataset_x = [[-1.0,2.0], [3.0, -1.0]]
    #dataset_x = [[-0.15, 0.50], [0.75, -0.25], [0.1, -0.8],
    #             [-0.2, -0.7]]
    #solver = SVM(c = error_C, kernel_type="linear")
    solver2 = SVC(kernel="linear", C=error_C)
    
    model = MY_SVM(C=error_C, kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=1e-5, max_iter=1e4)
    #model2 = MY_SVM(C=error_C, kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=1e-5, max_iter=1e4)

    print("C =", error_C)

    libsvm_start_time = time.time()
    solver2.fit(dataset_x, dataset_y)
    libsvm_end_time = time.time()
    print("LibSVM", "time %s" % (libsvm_end_time - libsvm_start_time), "w:", solver2.coef_, "b:", solver2.intercept_)

    #start_time = time.time()
    #solver.fit(dataset_x, dataset_y)
    #end_time = time.time()
    #print("Platt implementation", "time %s" % (end_time - start_time), "w:", solver.get_w(), "b:", -solver.get_b())

    start_time = time.time()
    model.fit_modification1(dataset_x, dataset_y)
    end_time = time.time()
    print("Keerthi 2 implementation", "time %s" % (end_time - start_time), "w:", model.get_w(), "b:", -model.get_b())

    #start_time = time.time()
    #model2.fit_modification2(dataset_x, dataset_y)
    #end_time = time.time()
    #print("Keerthi 2 implementation", "time %s" % (end_time - start_time), "w:", model2.get_w(), "b:", -model.get_b())

    
    green_x, green_y = zip(*solver2.support_vectors_)
    #plt.scatter(green_x, green_y, c="black", marker="x")

    count = 0
    for i in range(len(solver2.dual_coef_)):
        if abs(solver2.dual_coef_[0][i]) == error_C:
            count += 1
    print("LibSVM support_vectors", len(solver2.support_vectors_), "bound", count)

    #count = 0
    #for i in range(len(solver.alpha)):
    #    if abs(solver.alpha[i]) == error_C:
    #        count += 1
    #print("Platt support_vectors", len(solver.support_vectors), "bound", count)

    #print("Keerthi 1 support_vectors", len(model.support_vectors), "bound", len(model.bounded))

    print("Keerthi 2 support_vectors", len(model.support_vectors), "bound", len(model.bounded))

    w = model.get_w()[0]
    k = -w[0] / w[1]
    b = model.get_b() / w[1]
    line_x = [0, 10]
    line_y = [k * x + b for x in line_x]
    plt.figure()
    plt.scatter(blue_x, blue_y, c="blue")
    plt.scatter(red_x, red_y, c="red")
    plt.plot(line_x, line_y, label="Keerthi dividing hyperlpane", linewidth=3, color="black")
    plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=80, marker='x', color='black', label="Support vectors")

    w = model.get_w()[0]
    k = -w[0] / w[1]
    b = (model.get_b() - 1) / w[1]
    line1_y = [k * x + b for x in line_x]
    plt.plot(line_x, line1_y, linestyle="dashdot", color="black", label="Supporting hyperplane")
    
    w = model.get_w()[0]
    k = -w[0] / w[1]
    b = (model.get_b() + 1) / w[1]
    line1_y = [k * x + b for x in line_x]
    plt.plot(line_x, line1_y, linestyle="dashdot", color="black", label="Supporting hyperplane")

    plt.legend()

    plt.xlim(left=4, right=5)
    plt.ylim(bottom=4, top=5)


plt.show()

