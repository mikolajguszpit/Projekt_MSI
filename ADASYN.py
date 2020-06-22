
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import datasets

seed = 997
np.random.seed(seed)

class adasyn():

    """
    Adaptively generating minority data samples according to their distributions.
    More synthetic data is generated for minority class samples that are harder to learn.
    Harder to learn data is defined as positive examples with not many examples for in their respective neighbourhood.
    Inputs
         -----
         X:  Input features, X, sorted by the minority examples on top.  Minority example should also be labeled as 1
         y:  Labels, with minority example labeled as 1
      beta:  Degree of imbalance desired.  Neg:Pos. A 1 means the positive and negative examples are perfectly balanced.
         K:  Amount of neighbours to look at
 threshold:  Amount of imbalance rebalance required for algorithm
    Variables
         -----
         xi:  Minority example
        xzi:  A minority example inside the neighbourhood of xi
         ms:  Amount of data in minority class
         ml:  Amount of data in majority class
        clf:  k-NN classifier model
          d:  Ratio of minority : majority
       beta:  Degree of imbalance desired
          G:  Amount of data to generate
         Ri:  Ratio of majority data / neighbourhood size.  Larger ratio means the neighbourhood is harder to learn,
              thus generating more data.
     Minority_per_xi:  All the minority data's index by neighbourhood
     Rhat_i:  Normalized Ri, where sum = 1
         Gi:  Amount of data to generate per neighbourhood (indexed by neighbourhoods corresponding to xi)
    Returns
         -----
  syn_data:  New synthetic minority data created
    """
    def fit_resample(self, X, y, beta = 1, K = 5, threshold = 1):
        ms = int(sum(y))
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
                xi = X[i, :].reshape(1,-1)
                neighbours = clf.kneighbors(xi, n_neighbors=K+1, return_distance=False)[0]
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
                xi = X[i, :].reshape(1,-1)
                for j in range(Gi[flag]):
                    if Minority_per_xi[flag]:
                        index = np.random.choice(Minority_per_xi[flag])
                        xzi = X[index, :].reshape(1, -1)
                        si = xi + (xzi - xi) * np.random.uniform(0, 1)
                        syn_data.append(si)
                    else:
                        syn_data.append(xi)
                flag+=1;
        data = []
        print(Ri)
        print(Gi)
        print(Minority_index)
        print(Minority_per_xi)
        labels = []
        for values in syn_data:
            data.append(values[0])
        #print("{} amount of minority class samples generated".format(len(data)))
        labels2 = np.ones([len(data), 1])
        labels = np.concatenate([labels2,y.reshape(-1, 1)])
        data = np.concatenate([data, X])
        adasyn_data = np.concatenate([data,labels], axis=1)
        X_resampled = adasyn_data[:, :-1]
        y_resampled = adasyn_data[:, -1].astype(int)
        #plt.figure(figsize=(10,10))
        #plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled, cmap='bwr')
        #plt.tight_layout()
        #plt.savefig('po.png')
        #np.savetxt("adasyn.csv", adasyn_data, delimiter=",",fmt=["%.4f" for i in range(X.shape[1])] + ["%i"],)
        return X_resampled, y_resampled

#dataset = 'yeast4'
#dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
#X = dataset[:, :-1]
#y = dataset[:, -1].astype(int)
#plt.figure(figsize=(10,10))
#plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
#plt.tight_layout()
#plt.savefig('przed.png')
#adasyn(X,y,1,5)
