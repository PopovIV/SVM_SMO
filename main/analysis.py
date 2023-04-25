import numpy as np
import matplotlib.pyplot as plt
import time

#from simplified_smo import SVM#, OneVsAllClassifier
from my_smo import MY_SVM
from sklearn.svm import SVC
import math
import random

def sdf(x, y):
    return (x-3)*(x-3) + (y-4)*(y-4) - 0.25 - random.uniform(-0.1, 0.1)


print("Size, Support vec, errors")
for dataset_size in range(10, 1000, 10):
    dataset_x = [[random.uniform(2, 5), random.uniform(2, 5)] for i in range(dataset_size)]
    dataset_y = []
    for i in range(len(dataset_x)):
        if sdf(dataset_x[i][0], dataset_x[i][1]) > 0:
            dataset_y.append(-1)
        else:
            dataset_y.append(1)

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    #solver = SVM(kernel_type="rbf")
    model = MY_SVM(C=1000, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=1e-3, max_iter=1e4)

    model.fit_modification2(dataset_x, dataset_y)

    errors = 0
    for i in range(len(dataset_x)):
        #if solver.predict(dataset_x[i])[0] != dataset_y[i]:
        if model.predict([dataset_x[i]]) != dataset_y[i]:
            errors += 1


    print(len(dataset_y), ",", len(model.support_vectors), ",", errors)

