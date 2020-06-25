import numpy as np
from ADASYN import ADASYN
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import precision
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ttest_ind

dataset = ['yeast4',  'vowel0', 'glass5', 'ecoli4', 'glass-0-1-6_vs_2',
           'page-blocks-1-3_vs_4', 'yeast-1-2-8-9_vs_7', 'yeast6']

preprocs = {
    'none': None,
    'ADASYN': ADASYN(),
    'SMOTE' : SMOTE(random_state=997),
    'ROS': RandomOverSampler(random_state = 997),
    'RUS': RandomUnderSampler(random_state = 997)
}

labels = {
    'yeast4': None,
    'vowel0': None,
    'glass5': None,
    'ecoli4': None,
    'glass-0-1-6_vs_2': None,
    'page-blocks-1-3_vs_4': None,
    'yeast-1-2-8-9_vs_7': None,
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
            scores[preproc_id, data_id, fold_id] = precision(y[test], y_pred)

mean_scores = np.mean(scores, axis=2).T

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(labels.keys())), axis=1)
scores_M = np.concatenate((names_column, mean_scores), axis=1)
scores_M = tabulate(scores_M, headers, tablefmt="2f")

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))
names_column_1 = np.expand_dims(np.array(list(preprocs.keys())), axis=1)

print("\nPrecision score")
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
    advantage_table = tabulate(np.concatenate(
        (names_column_1, advantage), axis=1), headers)

    significance = np.zeros((len(preprocs), len(preprocs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column_1, significance), axis=1), headers)

    sign_better = significance * advantage
    sign_better_table = tabulate(np.concatenate(
        (names_column_1, sign_better), axis=1), headers)

    print("\n Statystycznie znacząco lepszy (" + data_name[label] + "):\n",
          sign_better_table, "\n\n")