from typing import Tuple

import numpy as np
import random
import time

class Solver:
    """
    Abstract class for solver. Override to implement a Solver.
    """

    def __init__(
        self
    ) -> None:

        pass

    def predict(
        self
    ) -> None:

        pass

    def fit(
        self
    ) -> None:

        pass


class SVM(Solver):
    
    """
    Support Vector Machine model.
    The model is trained using the Sequential Minimal Optimization as described in:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    """
    
    def __init__(
        self,
        c: float = 10.,
        kkt_thr: float = 1e-3,
        eps: float = 10e-5,
        max_iter: int = 1e4,
        kernel_type: str = 'rbf',
        gamma_rbf: float = 0.01
    ) -> None:

        """
        Arguments:
            c: Penalty parameter, trade-offs wide margin (lower c) and small number of margin failures

            kkt_thr: threshold for satisfying the KKT conditions

            max_iter: maximal iteration for training

            kernel_type: can be either 'linear' or 'rbf'

            gamma: gamma factor for RBF kernel
        """

        if not kernel_type in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        # Initialize
        self.c = float(c)
        self.eps = eps
        self.max_iter = max_iter
        self.kkt_thr = kkt_thr
        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

        self.b = np.array([])  # SVM's threshold
        self.alpha = np.array([])  # Alpha parameters of the support vectors
        self.support_vectors = np.array([])  # Matrix in which each row is a support vector
        self.support_labels = np.array([])  # Vector with the ground truth labels of the support vectors

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        SVM predict method.

        Arguments:
            x : (N,D) data matrix, each row is a sample.

        Return:
            pred : SVM predictions, the i-th element corresponds to the i-th sample, y={-1,+1}

            scores: raw scores per sample
        """

        w = self.support_labels * self.alpha
        x = self.kernel(self.support_vectors, x)

        scores = np.matmul(w, x) - self.b
        pred = np.sign(scores)

        return pred, scores

    def takeStepPlatt(self, i_1, i_2) -> int:
        if i_1 == i_2:
            # Same sample, skip
            return 0

        # Extract samples and labels
        x_1, y_1, alpha_1 = self.support_vectors[i_1, :], self.support_labels[i_1], self.alpha[i_1]
        x_2, y_2, alpha_2 = self.support_vectors[i_2, :], self.support_labels[i_2], self.alpha[i_2]

        # Calculate predictions for x_1 and x_2
        _, score_1 = self.predict(x_1)
        _, score_2 = self.predict(x_2)

        # Calculate errors for x_1 and x_2
        E_1 = score_1 - y_1
        E_2 = score_2 - y_2

        # Update boundaries
        L = 0
        H = 0
        if y_1 == y_2:
            L = max(0, alpha_1 + alpha_2 - self.c)
            H = min(self.c, alpha_1 + alpha_2)
        else:
            L = max(0, alpha_2 - alpha_1)
            H = min(self.c, self.c + alpha_2 - alpha_1)

        if L == H:
            return 0

        # Compute eta
        k11 = self.kernel(x_1, x_1)
        k22 = self.kernel(x_2, x_2)
        k12 = self.kernel(x_1, x_2)
        eta = k11 + k22 - 2 * k12
        
        a2 = 0
        if (eta > 0):
            a2 = alpha_2 + y_2 * (E_1 - E_2) / eta
            a2 = np.minimum(a2, H)
            a2 = np.maximum(a2, L)
        else:
            # objective function at a2=L
            c1 = -eta/2
            c2 = y_2 * (E_1 - E_2) + eta * alpha_2
            
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if (Lobj > Hobj):
                a2 = L
            elif (Lobj < Hobj):
                a2 = H
            else:
                a2 = alpha_2

        if (abs(a2 - alpha_2) < self.eps):
            return 0
            
        # Compute alpha_1
        a1 = alpha_1 + y_1*y_2*(alpha_2 - a2)
        
        # Check if alphaNew1 is a legal value for a LM, if it is not
        # between 0 and C, update alphaNew2 accordingly
        if a1 < 0:
            a2 = a2 + y_1*y_2*a1
            a1 = 0
        elif a1 > self.c:
            t = a1 - self.c
            a2 = a2 + y_1*y_2 * t
            a1 = self.c

        # Update threshold b (must be before updating alpha's)
        b1 = self.b + (E_1 + y_1 * (a1 - alpha_1) * k11 + y_2 * (a2 - alpha_2) * k12)
        b2 = self.b + (E_2 + y_1 * (a1 - alpha_1) * k12 + y_2 * (a2 - alpha_2) * k22)

        if a1 > 0 and a1 < self.c:
            self.b = b1
        elif a2 > 0 and a2 < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # Update alpha vector
        self.alpha[i_1] = a1
        self.alpha[i_2] = a2

        return 1

    def examineExamplePlatt(self, i2) -> int:
        y2 = self.support_labels[i2]
        a2 = self.alpha[i2]
        E2 = self.predict(self.support_vectors[i2, :])[1] - y2
        r2 = E2 * y2
        
        if (r2 < -self.kkt_thr and a2 < self.c) or (r2 > self.kkt_thr and a2 > 0):
            i1 = -1

            # 1) From non-bound examples, so that E1-E2 is maximized.
            mx = 0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 0 and self.alpha[i] < self.c:
                    E1 = self.predict(self.support_vectors[i, :])[1] - self.support_labels[i]
                    diff = abs(E1 - E2)
                    if (diff > mx):
                        mx = diff
                        i1 = i

            if i1 > 0:
                if (self.takeStepPlatt(i1, i2)):
                    return 1
            
            # 2) If we cannot make progress with the best non-bound example, then try any non-bound examples
            #  (start iterating at random position in order not to bias smo towards example at the beginnig of the dataset)
            startIndex = 0#random.randrange(0, len(self.alpha))
            for i in range(startIndex, len(self.alpha)):
                if self.alpha[i] > 0 and self.alpha[i] < self.c:
                    if (self.takeStepPlatt(i, i2)):
                        return 1
                        
            # Repeat the same iteration for the index before start index, if takeStep have not succeeded yet 
            for i in range(0, startIndex):
                if self.alpha[i] > 0 and self.alpha[i] < self.c:
                    if (self.takeStepPlatt(i, i2)):
                        return 1
            
            # 3) If we cannot make progress with the non-bound examples, then try any example.
            #(start iterating at random position in order not to bias smo towards example at the beginnig of the dataset)
            
            startIndex = 0#random.randrange(0, len(self.alpha))
            for i in range(startIndex, len(self.alpha)):
                if (self.takeStepPlatt(i, i2)):
                    return 1
                        
            # Repeat the same iteration for the index before start index, if takeStep have not succeeded yet 
            for i in range(0, startIndex):
                if (self.takeStepPlatt(i, i2)):
                    return 1
        return 0

    # Platt original algorithm
    def fitPlatt(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train

        iter_idx = 0
        numChanged = 0
        examineAll = 1

        # calculate start bias
        biases = 0
        self.b = 0
        for i in range(len(self.support_vectors)):
            biases += self.support_labels[i] - self.predict(self.support_vectors[i])[1]

        self.b = biases / max(len(self.support_vectors), 1)

        #print("SVM training using origianl SMO algorithm - START")
        while iter_idx < self.max_iter and (numChanged > 0 or examineAll):
            numChanged = 0
            
            if examineAll:
                for i in range(N):
                    numChanged += self.examineExamplePlatt(i)
            else:
                for i in range(N):
                    if self.alpha[i] > 0 and self.alpha[i] < self.c:
                        numChanged += self.examineExamplePlatt(i)
            
            if examineAll:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

            iter_idx = iter_idx + 1

        # calculate final bias
        #biases = 0
        #self.b = 0
        #for i in range(len(self.support_vectors)):
        #    biases += self.predict(self.support_vectors[i])[1] - self.support_labels[i]

        #self.b = biases / max(len(self.support_vectors), 1)

        # Store only support vectors
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        #print(f"Training summary: {iter_idx} iterations, {self.alpha.shape[0]} supprts vectors")
        #print("SVM training using SMO algorithm - DONE!")

    def takeStepKeerthi(self, i_1, i_2) -> int:
        if i_1 == i_2:
            # Same sample, skip
            return 0

        F_1 = 0
        F_2 = 0

        # Extract samples and labels
        x_1, y_1, alpha_1 = self.support_vectors[i_1, :], self.support_labels[i_1], self.alpha[i_1]
        x_2, y_2, alpha_2 = self.support_vectors[i_2, :], self.support_labels[i_2], self.alpha[i_2]

        # Calculate errors for x_1 and x_2
        if (self.alpha[i_1] > 0 and self.alpha[i_1] < self.c):
            F_1 = self.Fcache[i_1]
        else:
            F_1 = self.predict(self.support_vectors[i_1, :])[1] - y_1 # no need to add +self.b, because it is zero
            self.Fcache[i_1] = F_1

        if (self.alpha[i_2] > 0 and self.alpha[i_2] < self.c):
            F_2 = self.Fcache[i_2]
        else:
            F_2 = self.predict(self.support_vectors[i_2, :])[1] - y_2 # no need to add +self.b, because it is zero
            self.Fcache[i_2] = F_2

        # Calculate predictions for x_1 and x_2
        _, score_1 = self.predict(x_1)
        _, score_2 = self.predict(x_2)

        # Update boundaries
        L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)
        if L == H:
            return 0

        # Compute eta
        eta = self.compute_eta(x_1, x_2)
        
        a2 = 0
        if (eta > 0):
            a2 = alpha_2 + y_2 * (F_1 - F_2) / eta
            a2 = np.minimum(a2, H)
            a2 = np.maximum(a2, L)
        else:
            # objective function at a2=L
            c1 = -eta/2
            c2 = y_2 * (F_1 - F_2) + eta * alpha_2
            
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if (Lobj < Hobj - self.eps):
                a2 = L
            elif (Lobj > Hobj + self.eps):
                a2 = H
            else:
                a2 = alpha_2

        if (abs(a2 - alpha_2) < self.eps * (a2 + alpha_2 + self.eps)):
            return 0
            
        # Compute alpha_1
        a1 = alpha_1 + y_1*y_2*(alpha_2 - a2)
        
        # Check if alphaNew1 is a legal value for a LM, if it is not
        # between 0 and C, update alphaNew2 accordingly
        if a1 < 0:
            a2 = a2 + y_1*y_2*a1
            a1 = 0
        elif a1 > self.c:
            t = a1 - self.c;
            a2 = a2 + y_1*y_2 * t;
            a1 = self.c;

        # Round the new alphas that are too close to the boundaries.
        if a2 < 1e-7:
            a2 = 0
        elif a2 > (self.c - 1e-7):
            a2 = self.c

        if a1 < 1e-7:
           a1 = 0
        elif a1 > (self.c - 1e-7):
           a1 = self.c

        # Update alpha vector
        self.alpha[i_1] = a1
        self.alpha[i_2] = a2

        deltaAlpha1 = a1 - alpha_1
        deltaAlpha2 = a2 - alpha_2
        
        #Update Fcache for non boundary LMs
        for i in range(len(self.alpha)):
            if (self.alpha[i] > 0 and self.alpha[i] < self.c) or i == i_1 or i == i_2:
                self.Fcache[i] += y_1 * deltaAlpha1 * self.kernel(self.support_vectors[i_1], self.support_vectors[i])
                self.Fcache[i] += y_2 * deltaAlpha2 * self.kernel(self.support_vectors[i_2], self.support_vectors[i])

        # update b_up and b_down
        for i in range(len(self.alpha)):
            if (self.alpha[i] > 0 and self.alpha[i] < self.c) or i == i_1 or i == i_2:
                if self.Fcache[i] < self.b_up:
                    self.b_up = self.Fcache[i]
                    self.i_up = i
                if self.Fcache[i] > self.b_down:
                    self.b_down = self.Fcache[i]
                    self.i_down = i

        return 1

    def examineExampleKeerthi(self, i2) -> int:
        y2 = self.support_labels[i2]
        F2 = 0
        i1 = -1
        
        if self.alpha[i2] > 0 and self.alpha[i2] < self.c:
            F2 = self.Fcache[i2]
        else:
            F2 = self.predict(self.support_vectors[i2, :])[1] - y2 # no need to add +self.b, because it is zero
            self.Fcache[i2] = F2
            
            # Update (b_down, i_down) or (b_up, i_up) using (F2, i2)
            #if in I_1 or I_2
            if ((self.alpha[i2] == 0 and self.support_labels[i2] == 1) or (self.alpha[i2] == self.c and self.support_labels[i2] == -1)) and (F2 < self.b_up):
                self.b_up = F2
                self.i_up = i2
            #if in I_3 or I_4
            elif ((self.alpha[i2] == self.c and self.support_labels[i2] == 1) or (self.alpha[i2] == 0 and self.support_labels[i2] == -1)) and (F2 > self.b_down):
                self.b_down = F2
                self.i_down = i2
        
        optimality = 1
        
        # if in I_0 or I_1 or I_2
        if (self.support_labels[i2] == 1 and self.alpha[i2] < self.c) or (self.support_labels[i2] == -1 and self.alpha[i2] > 0):
            if self.b_down - F2 > 2 * self.kkt_thr:
                optimality = 0
                i1 = self.i_down
        
        # if in I_0 or I_3 or I_4
        if (self.support_labels[i2] == -1 and self.alpha[i2] < self.c) or (self.support_labels[i2] == 1 and self.alpha[i2] > 0):
            if F2 - self.b_up > 2 * self.kkt_thr:
                optimality = 0
                i1 = self.i_up
        
        if optimality == 1:
            return 0
            
        # If i1 is a non boundary example and b_up <= Fi1 <= b_down
        # both i_down and i_up are valid choice. Then we need to choose
        # the best among the 2 possible values.
        if (self.alpha[i2] > 0 and self.alpha[i2] < self.c):
            if self.b_down - F2 > F2 - self.b_up:
                i1 = self.i_down
            else:
                i1 = self.i_up

        return self.takeStepKeerthi(i1,i2)

    # Keerthi optimization
    def fitKeerthi(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train
        self.Fcache = np.zeros(N)
        self.i_up = 0
        self.i_down = 0
        self.b_up = -1
        self.b_down = 1
        
        while self.support_labels[self.i_up] != 1:
            self.i_up += 1
            
        while self.support_labels[self.i_down] != -1:
            self.i_down += 1
        
        self.Fcache[self.i_up] = -1
        self.Fcache[self.i_down] = 1

        iter_idx = 0
        numChanged = 0
        examineAll = 1
        
        print("SVM training using Keerthi SMO algorithm - START")
        while iter_idx < self.max_iter and (numChanged > 0 or examineAll):
            numChanged = 0
            
            if examineAll:
                for i in range(N):
                    numChanged += self.examineExampleKeerthi(i)
            else:
                for i in range(N):
                    if self.alpha[i] > 0 and self.alpha[i] < self.c:
                        numChanged += self.examineExampleKeerthi(i)
                    
                    if self.b_up > self.b_down - 2 * self.kkt_thr:
                        break
            
            if examineAll:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

            iter_idx = iter_idx + 1
        
        # round alpha too near to 0
        for i in range(N):
            if self.alpha[i] < 1e-10:
                self.alpha[i] = 0

        # Store only support vectors
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        # compute bias
        allBias = 0
        for i in range(N):
            allBias += y_train[i] - self.predict(x_train[i, :])[1]
        
        print("Iterations ", iter_idx)
        self.b = allBias / N
        #print(f"Training summary: {iter_idx} iterations, {self.alpha.shape[0]} supprts vectors")
        #print("SVM training using SMO algorithm - DONE!")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        """
        Train SVM classifier.

        Arguments:
            x_train : (N,D) data matrix, each row is a sample.

            y_train : Labels vector, y must be {-1,1}
        """
        self.fitPlatt(x_train, y_train)
        #self.fitKeerthi(x_train, y_train)
        return

    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2 ) -> Tuple[float, float]:

        """"
        Computes the lower and upper bounds for alpha_2.

        Arguments:
            alpha_1, alpha_2: Values before optimization

            y_1, y_1: labels of x[i1,:] and x[i2,:]

        Return:
            lb: Lower bound of alpha_2

            ub: Upper bound of alpha_2
        """

        if y_1 == y_2:
            lb = np.max([0, alpha_1 + alpha_2 - self.c])
            ub = np.min([self.c, alpha_1 + alpha_2])
        else:
            lb = np.max([0, alpha_2 - alpha_1])
            ub = np.min([self.c, self.c + alpha_2 - alpha_1])
        return lb, ub

    
    def compute_eta(self, x_1, x_2) -> float:

        """
        Computes eta = K(x_1,x_1) + K(x_2,x_2) - 2K(x_1,x_2)

        Arguments:
            x_1, x_2: feature vectors of samples i1, i2

        Return:
            eta
        """

        return self.kernel(x_1, x_1) + self.kernel(x_2, x_2) - 2*self.kernel(x_1, x_2)

    def compute_b(self, alpha_1_new, alpha_2_new, E_1, E_2, i_1, i_2) -> None:

        """"
        Computes the updated threshold b.
        Uses the same method as in the original paper.

        Arguments:
            alpha_1_new: New value of alpha_1

            alpha_2_new: New value of alpha_2

            E_1: Difference between SVM's prediction and label

            E_2:  Difference between SVM's prediction and label

            i_1: Index of alpha_1

            i_2: Index of alpha_2
        """

        x_1 = self.support_vectors[i_1]
        x_2 = self.support_vectors[i_2]

        b1 = self.b - E_1 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel(x_1, x_1) - \
            self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel(x_1, x_2)

        b2 = self.b - E_2 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel(x_1, x_2) - \
            self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel(x_2, x_2)

        if 0 < alpha_1_new < self.c:
            self.b = b1
        elif 0 < alpha_2_new < self.c:
            self.b = b2
        else:
            self.b = np.mean([b1, b2])

    def rbf_kernel(self, u, v):

        """
        RBF kernel implementation, i.e. K(u,v) = exp(-gamma_rbf*|u-v|^2).
        gamma_rbf is a hyper parameter of the model.

        Arguments:
            u: an (N,) vector or (N,D) matrix,

            v: if u is a vector, v must have the same dimension
               if u is a matrix, v can be either an (N,) or (N,D) matrix.

        Return:
            K(u,v): kernel matrix as follows:
                    case 1: u, v are both vectors:
                        K(u,v): a scalar K=u.T*v

                    case 2: u is a matrix, v is a vector
                        K(u,v): (N,) vector, the i-th element corresponds to K(u[i,:],v)

                    case 3: u and V are both matrices
                        k(u,v): (N,D) matrix, the i,j entry corresponds to K(u[i,:],v[j,:])
        """

        # In case u, v are vectors, convert to row vector
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]

        if np.ndim(u) == 1:
            u = u[np.newaxis, :]

        # Broadcast to (N,D,M) array
        # Element [i,:,j] is the difference between i-th row in u and j-th row in v
        # Squared norm along second axis, to get the norm^2 of all possible differences, results an (N,M) array
        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)

        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(u, v) -> np.ndarray:

        """
        Linear kernel implementation.

        Arguments:
            u: an (N,) vector or (N,D) matrix,

            v: if u is a vector, v must have the same dimension
               if u is a matrix, v can be either an (N,) or (N,D) matrix.

        Return:
            K(u,v): kernel matrix as follows:
                    case 1: u, v are both vectors:
                        K(u,v): a scalar K=u.T*v

                    case 2: u is a matrix, v is a vector
                        K(u,v): (N,) vector, the i-th element corresponds to K(u[i,:],v)

                    case 3: u and V are both matrices
                        k(u,v): (N,D) matrix, the i,j entry corresponds to K(u[i,:],v[j,:])
        """

        return np.dot(u, v.T)
    
    def get_w(self):
        w = np.zeros_like(self.support_vectors[0])
        for i in range(len(self.support_labels)):
            w += self.support_labels[i] * self.alpha[i] * self.support_vectors[i]
        return w
        
    def get_b(self):
        return self.b


class OneVsAllClassifier:

    """
    Implements a one-vs-all strategy for multi-class classification.
    
    We run n SVM-s where each says "Is it in this class or not".
    """

    def __init__(
        self,
        solver: Solver,
        num_classes: int,
        **kwargs
    ) -> None:

        """
        Arguments:
            solver : solver class, e.g., SVM

            num_classes : numer of classes

            kwargs : keyword arguments passed to each solver instance
            
        """

        self._binary_classifiers = [solver(**kwargs) for i in range(num_classes)]
        self._num_classes = num_classes

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        """
        Arguments:
            x : (N,D) data matrix, each row is a sample.

        Return:
            pred : predictions, the i-th element corresponds to the i-th sample, y={-1,+1}
        """
        
        n = x.shape[0]
        scores = np.zeros((n, self._num_classes))

        for idx in range(self._num_classes):
            scores[:,idx] = self._binary_classifiers[idx].predict(x)[1]

        pred = np.argmax(scores, axis=1)

        return pred

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:

        """
        Train One-vs-All classifier.

        Arguments:
            x_train : (N,D) data matrix, each row is a sample.

            y_train : Labels vector, y must be {-1,1}
        """

        for idx in range(self._num_classes):
            # Convert labels to binary {+1,-1}
            y_tmp = 1.*(y_train == idx) - 1.*(y_train != idx)
            
            print(f"Fitting classifier {idx}/{self._num_classes}")
            
            start_time = time.time()
            # Train a binary classifier
            self._binary_classifiers[idx].fit(x_train, y_tmp)
            end_time = time.time()
            print(f"Finished. Time passed {end_time - start_time}s")