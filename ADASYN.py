import numpy as np
from sklearn import neighbors

class ADASYN:
    def fit_resample(X, y, K=5, beta=1, threshold=1):
        seed = 997
        np.random.seed(seed)

        ms = int(np.sum(y))
        ml = len(y) - ms
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X, y)
        d = np.divide(ms, ml)
        if d > threshold:
            return print("The data set is not imbalanced enough.")
        G = (ml - ms) * beta
        Ri = []
        Minority_per_xi = []
        Minority_index = []
        for i in range(len(y)):
            if y[i] == 1:
                Minority_index.append(i)
        for i in range(len(y)):
            if y[i] == 1:
                xi = X[i, :].reshape(1, -1)
                neighbours = clf.kneighbors(xi, n_neighbors=K + 1, return_distance=False)[0]
                neighbours = neighbours[1:]
                count = 0
                for value in neighbours:
                    if not value in Minority_index:
                        count += 1
                minority = []
                for value in neighbours:
                    if value in Minority_index:
                        minority.append(value)
                Minority_per_xi.append(minority)
                Ri.append(count / K)
        Rhat_i = []
        for ri in Ri:
            rhat_i = ri / sum(Ri)
            Rhat_i.append(rhat_i)
        Gi = []
        for rhat_i in Rhat_i:
            gi = round(rhat_i * G)
            Gi.append(int(gi))
        syn_data = []
        flag = 0
        for i in range(len(y)):
            if y[i] == 1:
                xi = X[i, :].reshape(1, -1)
                for j in range(Gi[flag]):
                    if Minority_per_xi[flag]:
                        index = np.random.choice(Minority_per_xi[flag])
                        xzi = X[index, :].reshape(1, -1)
                        si = xi + (xzi - xi) * np.random.uniform(0, 1)
                        syn_data.append(si)
                    else:
                        syn_data.append(xi)
                flag += 1
        data = []
        labels = []
        for values in syn_data:
            data.append(values[0])
        # print("{} amount of minority class samples generated".format(len(data)))
        labels2 = np.ones([len(data), 1])
        labels = np.concatenate([y.reshape(-1, 1), labels2])
        data = np.concatenate([X, data])
        return data, labels
