import sklearn.metrics as met
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def model_processing(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    # sc = StandardScaler()
    # train_x = sc.fit_transform(train_x)
    # test_x = sc.transform(test_x)
    return train_x, test_x, train_y, test_y

def lda_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'LDA'
    model_fit = LinearDiscriminantAnalysis().fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def gnb_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'GNB'
    model_fit = GaussianNB().fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def knn_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'KNN'
    model_fit = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski').fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def svm_ln_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'SVM_Linear'
    model_fit = SVC(kernel='linear', random_state=0, C=1).fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def svm_rbf_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'SVM_rbf'
    model_fit = SVC(kernel='rbf', gamma='scale', C=0.5).fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def svm_poly_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'SVM_Poly'
    model_fit = SVC(kernel='poly', degree=3, gamma='scale').fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def lor_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'LoR'
    model_fit = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', C=1e5).fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def dtc_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'DTC'
    model_fit = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new

def mlp_cl(trainx, trainy, testx, testy, new_d_x, new_d_y):
    model_name = 'NN'
    model_fit = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=55, alpha=1e-4,solver='lbfgs', random_state=0,
                                  verbose=False, learning_rate_init=0.1).fit(trainx, trainy)
    model_score = model_fit.score(testx, testy)
    predict_x = model_fit.predict(testx)
    score_class = met.classification_report(testy, predict_x, target_names=['CLASS1', 'CLASS2'])
    pre_new = model_fit.predict(new_d_x)
    return model_name, model_score, score_class, pre_new
