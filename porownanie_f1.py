import numpy as np
from ADASYN import ADASYN
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ranksums
from scipy.stats import rankdata
from scipy.stats import ttest_ind
dataset = ['yeast4',  'vowel0', 'glass5', 'ecoli4', 'glass-0-1-6_vs_2', 'page-blocks-1-3_vs_4', 'yeast-1-2-8-9_vs_7']

clf = DecisionTreeClassifier(random_state=997)

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
    'yeast-1-2-8-9_vs_7': None
}

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=997)

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
            scores[preproc_id, data_id, fold_id] = f1_score(y[test], y_pred)

mean_scores = np.mean(scores, axis=2).T

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)

mean_ranks = np.mean(ranks, axis=0)

alfa = .05
w_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)

w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")

p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nTabela przewagi:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nMacierz istotności:\n", significance_table)

names_column1 = np.expand_dims(np.array(list(labels.keys())), axis=1)
scores_M = np.concatenate((names_column1, mean_scores), axis=1)
scores_M = tabulate(scores_M, headers, tablefmt="2f")

sign_better = significance * advantage
sign_better_table = tabulate(np.concatenate(
    (names_column, sign_better), axis=1), headers)

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value1 = np.zeros((len(preprocs), len(preprocs)))

mean_ranks_table = tabulate((headers, mean_ranks))

print("\n F1_score")
print("\nUśrednionie wyniki\n", scores_M)
print("\nUśrednione rangi\n", mean_ranks_table)

data_name = list(labels.keys())

for label in range(len(labels)):
    scores_label = scores[:, label, :]
    for i in range(len(preprocs)):
        for j in range(len(preprocs)):
            t_statistic[i, j], p_value1[i, j] = ttest_ind(scores_label[i], scores_label[j])

    advantage1 = np.zeros((len(preprocs), len(preprocs)))
    advantage1[t_statistic > 0] = 1
    advantage1_table = tabulate(np.concatenate(
        (names_column, advantage1), axis=1), headers)

    significance1 = np.zeros((len(preprocs), len(preprocs)))
    significance1[p_value1 <= alfa] = 1
    significance1_table = tabulate(np.concatenate(
        (names_column, significance1), axis=1), headers)

    sign_better1 = significance1 * advantage1
    sign_better_table1 = tabulate(np.concatenate(
        (names_column, sign_better1), axis=1), headers)

    print("\n Statystycznie znacząco lepszy ("+data_name[label]+"):\n", sign_better_table1)
    print("\n")