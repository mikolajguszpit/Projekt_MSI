import numpy as np
from ADASYN import ADASYN
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import geometric_mean_score_1
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ttest_ind

dataset = ['ecoli4', 'glass-0-1-6_vs_2', 'glass5',
           'page-blocks-1-3_vs_4', 'vowel0', 'yeast-1-2-8-9_vs_7', 'yeast4',  'yeast6']

preprocs = {
    'none': None,
    'ADASYN': ADASYN(),
    'SMOTE' : SMOTE(random_state=997),
    'ROS': RandomOverSampler(random_state = 997),
    'RUS': RandomUnderSampler(random_state = 997)
}

labels = {
    'ecoli4': None,
    'glass-0-1-6_vs_2': None,
    'glass5': None,
    'page-blocks-1-3_vs_4': None,
    'vowel0': None,
    'yeast-1-2-8-9_vs_7': None,
    'yeast4': None,
    'yeast6': None
}
n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=997)
clf = DecisionTreeClassifier(random_state=997)

scores = np.zeros((len(preprocs),len(dataset),n_splits * n_repeats))

for data_id, dataset in enumerate(dataset):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for preproc_id, preproc in enumerate(preprocs):
            clf = clone(clf)

            if preprocs[preproc] == None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])
            scores[preproc_id, data_id, fold_id] = geometric_mean_score_1(y[test], y_pred)

mean_scores = np.mean(scores, axis=2).T

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(labels.keys())), axis=1)
scores_M = np.concatenate((names_column, mean_scores), axis=1)
scores_M = tabulate(scores_M, headers, tablefmt="2f", floatfmt='0.3f')

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

print("\nG-mean score")
print("\nUśrednionie wyniki\n", scores_M)

alfa = 0.05
data_name = list(labels.keys())

for label in range(len(labels)):
    scores_label = scores[:, label, :]
    for i in range(len(preprocs)):
        for j in range(len(preprocs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores_label[i], scores_label[j])

    advantage = np.zeros((len(preprocs), len(preprocs)))
    advantage[t_statistic > 0] = 1

    significance = np.zeros((len(preprocs), len(preprocs)))
    significance[p_value <= alfa] = 1

    sign_better = significance * advantage

    print("\n\nStatystycznie znacząco lepszy od(" + data_name[label] + "):\n")
    for i in range(len(headers)):
        print(i+1,".",headers[i],": ", end = '')
        for j in range(len(sign_better[i,:])):
            if sign_better[i,j] == 1:
                print(j+1," ", end = '')
        print("\n")
