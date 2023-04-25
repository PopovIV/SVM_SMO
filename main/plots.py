import matplotlib.pyplot as plt
import scipy.signal

data = []

with open("data.csv", "r") as f:
    f.readline()
    for line in f:
        data.append([float(t) for t in line.split(" , ")])



size, sup, err = zip(*data)


plt.plot(size, sup)
plt.plot(size, scipy.signal.medfilt(sup, 5))
plt.figure()
plt.plot(size, err)
plt.plot(size, scipy.signal.medfilt(err, 5))
plt.figure()

sup_x = list(set(sup))
sup_x.sort()
err1 = [0] * len(sup_x)
err2 = [0] * len(sup_x)

sup_x_counts = [0] * len(sup_x)

for i in range(len(sup)):
    for j in range(len(sup_x)):
        if sup_x[j] == sup[i]:
            err1[j] += err[i]
            err2[j] += err[i] / size[i]
            sup_x_counts[j] += 1
            break

for j in range(len(sup_x)):
    err1[j] /= max(sup_x_counts[j], 1)
    err2[j] /= max(sup_x_counts[j], 1)

for j in range(len(sup_x)):
    sup_x[j] /= 5

plt.plot(sup_x, err1)
plt.plot(sup_x, scipy.signal.medfilt(err1, 5))
plt.figure()

plt.plot(sup_x, err2)
plt.plot(sup_x, scipy.signal.medfilt(err2, 5))
plt.figure()

plt.show()