import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import  mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('input.txt', sep=" ", names=['user', 'item', 'rating'])
n_user = max(df['user'])
n_item = max(df['item'])

train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_user, n_item))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_user, n_item))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')


def predict(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    p = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return p


prediction = predict(train_data_matrix, user_similarity)


# evaluation
def rmse(pred, ground_truth):
    pred = pred[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(pred,ground_truth))


# test evaluation
print("Evaluation(RMSE): " + str(rmse(prediction, test_data_matrix)))

# fill out matrix
for i in range(n_user):
    high = max(prediction[i])
    low = min(prediction[i])
    val = (high - low)/5
    for j in range(n_item):
        t = prediction[i, j]
        if train_data_matrix[i, j] == 0:
            if t < low + val:
                train_data_matrix[i, j] = 1
            elif low+val <= t < low + 2*val:
                train_data_matrix[i, j] = 2
            elif low+2*val <= t < low + 3*val:
                train_data_matrix[i, j] = 3
            elif low+3*val <= t < low + 4*val:
                train_data_matrix[i, j] = 4
            elif low+4*val <= t <= high:
                train_data_matrix[i, j] = 5
# write file
f = open("output.txt", 'w')
for i in range(n_user):
    for j in range(n_item):
        f.write("% d % d % d\n" % (i + 1, j + 1, train_data_matrix[i, j]))
f.close()

print('done.\n')
