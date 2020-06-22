import numpy as np
from ADASYN import adasyn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ranksums
from scipy.stats import rankdata
dataset = ['yeast4',  'vowel0', 'glass5', 'ecoli4', 'glass-0-1-6_vs_2', 'page-blocks-1-3_vs_4', 'yeast-1-2-8-9_vs_7']

clf = DecisionTreeClassifier(random_state=997)

preprocs = {
    'none': None,
    'ADASYN': adasyn(),
    'SMOTE' : SMOTE(random_state=997),
    'ROS': RandomOverSampler(random_state = 997),
    'RUS': RandomUnderSampler(random_state = 997)
}

metrics = {
    'accuracy': balanced_accuracy_score,
    'recall': recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
}

yolo ={
    'yeast4':None,
    'vowel0':None,
    'shuttle-c2-vs-c4':None,
    'glass5':None,
    'ecoli4':None,
    'glass-0-1-6_vs_2':None,
    'page-blocks-1-3_vs_4':None,
    'yeast-1-2-8-9_vs_7':None
}

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=997)

scores = np.zeros((len(preprocs),len(dataset),n_splits * n_repeats, len(metrics)))

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

            for metric_id, metric in enumerate(metrics):
                scores[preproc_id, data_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)

np.save('results', scores)

print('\nScores:\n', scores.shape)

mean_scores = np.mean(scores, axis=2)

print('\nMean scores:\n', mean_scores)

print('\nMean scores:\n', mean_scores.shape)

mean_scores2 = np.mean(mean_scores, axis=1).T

print('\nMean scores:\n', mean_scores2)

print('\nMean scores:\n', mean_scores2.shape)

ranks = []
for ms in mean_scores2:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)

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
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)