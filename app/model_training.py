import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pickle

warnings.simplefilter('ignore')

offsets = [50,60,70,80,90,100]
nn_points = [17,19,21,23,25,27,39,31]

offset = 90
nn_point = 17

offset = 10
nn_point = 17

df = pd.read_csv('data/original_files/ETHXBT_60.csv', header=None,
                names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])

df = df.dropna()

df['unix_timestamp'] = df['unix_timestamp'].astype(int)

df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')

df = df.drop(['unix_timestamp', 'other_1', 'other_2'], axis=1)

df = df[['timestamp','open_price','high_price','low_price','close_price']]

df = df.resample('4H', on='timestamp').agg({
    'open_price': 'first',
    'high_price': 'max',
    'low_price': 'min',
    'close_price': 'last'
    })

df = df.dropna()

df = df[['open_price', 'high_price', 'low_price', 'close_price']]

threshold = 6

def set_spike_status(row):
    cng = 100 * (row['close_price'] - row['open_price'])/row['close_price']
    if cng >= threshold:
        status = 'True'
    else:
        status = 'False'
    return status

df['spike'] = df.apply(set_spike_status, axis=1)

df = df[['open_price','close_price','spike']]

#print(df.head(10))

open_prices_list = sliding_window_view(df['open_price'], offset).tolist()[:-1]
open_prices_list = np.array(open_prices_list)

close_prices_list = sliding_window_view(df['close_price'], offset).tolist()[:-1]
close_prices_list = np.array(close_prices_list)

#print(close_prices_list[:5])

spike_list = sliding_window_view(df['spike'], 1).tolist()[offset:]
spike_list = [elem[0] for elem in spike_list]

#print(spike_list[:5])

y = spike_list

names = [
    # "Nearest Neighbors",
    # "Gaussian Process",
    "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
]

classifiers = [
    # KNeighborsClassifier(nn_point),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    # MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    # AdaBoostClassifier(random_state=42),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]

for name, clf in zip(names, classifiers):

    true_ratios = []

    for _ in range(20):

        rus = RandomUnderSampler()

        X_ = []
        for ii in range(len(close_prices_list)):
            #X_.append(np.concatenate((arr, arr/min(arr), arr/max(arr), arr/arr[0], arr/arr[-1], [max(arr)], [min(arr)], [np.mean(arr)]), axis=0))
            X_.append(np.concatenate((close_prices_list[ii], close_prices_list[ii]/min(close_prices_list[ii]), [min(close_prices_list[ii])], open_prices_list[ii], open_prices_list[ii]/min(open_prices_list[ii]), [min(open_prices_list[ii])]), axis=0))
            #X_.append(arr)

        X = np.array(X_).tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        X_train, y_train = rus.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

        # with open('model.pickle', 'wb') as f:
        #     pickle.dump(clf, f)

        yhat = clf.predict(X_test)

        acc = accuracy_score(y_test, yhat)
        #print(name, 'Accuracy: %.3f' % acc)
        #print('Count of true in training set', y_train.count('True'))
        #print('Count of true in test set', y_test.count('True'))

        total_count = 0
        count_wrong = 0
        count_num_true = 0
        count_true_neg = 0
        for i in range(len(y_test)):
            total_count += 1
            if yhat[i] != y_test[i]:
                count_wrong += 1
            if y_test[i] == 'True':
                count_num_true += 1
                if yhat[i] != 'True':
                    count_true_neg += 1

        #print('total_count', total_count)
        #print('count_wrong', count_wrong)
        #print('count_true_neg', count_true_neg)
        #print('Real accuracy', (total_count - count_wrong)/total_count)
        #print('Count num true ratio', (count_num_true - count_true_neg)/count_num_true)
        #print(nn_point, offset, 'Count num true ratio', (count_num_true - count_true_neg)/count_num_true)
        true_ratio = (count_num_true - count_true_neg)/count_num_true
        true_ratios.append(true_ratio)
        print(name, 'Count num true ratio', true_ratio)
        #print('---------------------------------------')
    print('------------Average true ratio stats', np.array(true_ratios).mean(), np.array(true_ratios).min(), np.array(true_ratios).max())