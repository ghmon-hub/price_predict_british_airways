
import pandas as pd

def pre_processing():

    threads = []
    feature_name = {}

    ###Input Module
    #from google.colab import drive
    pd.set_option("mode.chained_assignment", None)
    data_set = pd.read_csv("/root/codes/ds_train/british_airways/customer_booking.csv", encoding="ISO-8859-1")
    data_set_1 = pd.DataFrame(data_set)
    #drive.mount('/content/drive')
    #data_set = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Learn_ML/customer_booking.csv", encoding="ISO-8859-1")

    x = data_set.drop(columns='booking_complete')
    y = data_set['booking_complete'].values
    n_t = 0
    n_n = 0

    for i, columns in enumerate(x):
        for n, types in enumerate(x.dtypes):
            if i == n:
                feature_name.setdefault(columns, []).append(str(types))
                feature_name.setdefault(columns, []).append(n)
    for x in data_set['booking_complete'].values:
        if x == 1:
            n_t += 1
        else:
            n_n += 1

    for key, value in feature_name.items() :
        if value[0] == 'object':
            list_tr = set(data_set[key].values)
            d_tr = {k: int(0) for k in list_tr}
            feature_name.setdefault(key, []).append(d_tr)
            for x in range(0, len(data_set['booking_complete'])):
                if data_set['booking_complete'][x] == 1:
                    feature_name[key][2][(data_set[key][x])] += 1
            for key_1, value_1 in feature_name[key][2].items():
                value_1 = (value_1/n_t)*(n_t/(n_t+n_n))
                feature_name[key][2][key_1] = value_1
            for x_1 in range(0, len(data_set['booking_complete'])):
                #data_set_1[key] = data_set_1[key].astype('int64')
                value_test = data_set_1[key][x_1]
                data_set_1[key][x_1] = feature_name[key][2][value_test]
            data_set_1[key] = data_set_1[key].astype('int64')

    data_set_1.to_csv("/root/codes/ds_train/british_airways/customer_booking_edit.csv", encoding="ISO-8859-1")
    x_e = data_set_1.drop(columns='booking_complete')
    y_e = data_set['booking_complete'].values
    return x_e, y_e, feature_name

