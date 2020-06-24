import numpy as np
from ADASYN import ADASYN
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import precision
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ranksums
from scipy.stats import rankdata
from scipy.stats import ttest_ind
from sklearn import datasets

clf = DecisionTreeClassifier(random_state=997)

preprocs = {
    'none': None,
    'ADASYN': ADASYN(),
    'SMOTE' : SMOTE(random_state=997),
    'ROS': RandomOverSampler(random_state = 997),
    'RUS': RandomUnderSampler(random_state = 997)
}

X,y= datasets.make_classification(n_samples=1000, random_state=997, weights=[0.90, 0.10])

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=997)

scores = np.zeros((len(preprocs),n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X,y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])
        scores[preproc_id, fold_id] = precision(y[test], y_pred)

mean_scores = np.mean(scores, axis=1).T
print("\nPrecision score")
print("\nUśrednionie wyniki\n", mean_scores)

alfa = .05
t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")

p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nTabela przewagi:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nMacierz istotności:\n", significance_table)

sign_better = significance * advantage
sign_better_table = tabulate(np.concatenate(
    (names_column, sign_better), axis=1), headers)

print('\nStatystycznie znacząco lepszy:\n', sign_better_table)

