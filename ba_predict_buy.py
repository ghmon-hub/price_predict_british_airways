from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

model = MLPClassifier(hidden_layer_sizes=(13,2), max_iter=110, alpha=1e-4, solver='sgd', random_state=1, verbose=True, learning_rate_init=0.1)

data_set = pd.read_csv("/root/codes/ds_train/british_airways/customer_booking.csv", encoding="ISO-8859-1")

names_cul = []

for data in data_set:
    if data_set[data].dtypes == 'object':
        # print(data)
        edt = pd.get_dummies(data_set[data])
        data_set = data_set.join(edt)
        data_set.drop(data, axis=1, inplace=True)

for col in data_set.columns:
    if col == 'booking_complete':
        continue
    else:
        names_cul.append(col)

x = data_set.drop(columns='booking_complete')
y = data_set['booking_complete'].values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

model.fit(train_x, train_y)
print(model.score(train_x, train_y))
print(model.score(test_x, test_y))

