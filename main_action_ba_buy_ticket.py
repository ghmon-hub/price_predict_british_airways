import threading
import ba_buy_ticket_model_processing as mp
import new_data_ba_buy_ticket_pre_process as np
from sklearn.model_selection import train_test_split
threads = []

models = ['gnb_cl', 'knn_cl', 'svm_ln_cl', 'svm_rbf_cl', 'svm_poly_cl','lor_cl', 'dtc_cl', 'mlp_cl', 'lda_cl']

np.new_data_pre()

train_x, test_x, train_y, test_y = train_test_split(np.new_data_pre()[2], np.new_data_pre()[3], test_size=0.2, random_state=0)

for model in models:
    model_1 = getattr(mp, model)
    th = threading.Thread(model_1(train_x, train_y, test_x, test_y, np.new_data_pre()[0], np.new_data_pre()[1]))
    th.start()
    threads.append(th)

for th in threads:
    th.join()
    #time.sleep(10)
    #return