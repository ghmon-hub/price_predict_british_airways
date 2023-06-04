import pandas as pd
import csv
import ba_buy_ticket_pre_proccess as tpp
#new_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Learn_ML/customer_real_data.csv", encoding="ISO-8859-1")

def new_data_pre(dir_path = "/root/codes/ds_train/british_airways/customer_real_data.csv"):
    new_row_list_x = []
    new_row_list_y = []
    csv_file = open(dir_path, 'r')
    reader = csv.reader(csv_file)
    next(reader)  # skip first row
    pre_pro_data = tpp.pre_processing()
    dict_fea = pre_pro_data[2]
    for row in reader:
        ele_y = row.pop()
        new_row_list_y.append(ele_y)
        for key, value in dict_fea.items():
            if value[0] == 'object':
                if row[value[1]] in value[2]:
                    row[value[1]] = value[2][row[value[1]]]
                else:
                    row[value[1]] = '0'
        new_row_list_x.append(row)
    return new_row_list_x, new_row_list_y, pre_pro_data[0], pre_pro_data[1]

