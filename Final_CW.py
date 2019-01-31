import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def process_data():
    wdbc = pd.read_csv("wdbc.data", header=None)
    X = wdbc.iloc[:, 2:31].values
    Y = wdbc.iloc[:, 1].values
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    # split data into training and test and shuffle
    le.transform(['M', 'B'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.297, shuffle=True)

    # normalize data
    X_test = preprocessing.normalize(X_test)
    X_train = preprocessing.normalize(X_train)

    return X_test, X_train, Y_test, Y_train


def pca(i):
    X_train, X_test, Y_train, Y_test = process_data()
    pca = PCA(n_components=i)
    pca.fit_transform(X_train)
    #print(pca)
    print("PCA " + str(i) + " Variance Ratio: ")
    print(pca.explained_variance_ratio_)



def decision_tree():
    #K-fold Cross Validation
    X_train, X_test, Y_train, Y_test = process_data()
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=3), LogisticRegression(random_state=1))
    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)
    print("Accuracy : %.3f" % pipeline.score(X_test, Y_test))
    kfold = StratifiedKFold(n_splits=10).split(X_train, Y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train], Y_train[train])
        score = pipeline.score(X_train[test], Y_train[test])
        scores.append(scores)
        print('Fold: %2d, Accuracy: %.3f' % (k+1, score))
    scores = cross_val_score(estimator=pipeline, X=X_train, y=Y_train, cv=10, n_jobs=1)
    print("CV accuracy scores: %s" % scores)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline,
                                                            X=X_train,
                                                            y=Y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=1)
    #Hyperparameters
    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, Y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    # Decision Tree
    print("\nDecision Tree")
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy', cv=10)
    scores = cross_val_score(gs, X_train, Y_train,scoring='accuracy', cv=5)
    print('Original accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    n = [3, 5, 7, 9, 11]
    pipe_svc.fit(X_train, Y_train)
    f1_scores = []
    for num in n:
        X_train, X_test, Y_train, Y_test = process_data()
        pca = PCA(n_components=num, random_state=1)
        train = pca.fit_transform(X_train)
        scores = cross_val_score(gs, train, Y_train, cv=10)
        print('PCA ' + str(num) + ' accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
        y_pred = pipe_svc.predict(X_test)
        f1 = f1_score(Y_test, y_pred)
        f1_scores.append(f1)
        print('F1: %.3f' % f1)
        print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
    scores = cross_val_score(gs, X_test, Y_test, cv=10)
    print('Test Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    f1 = f1_score(Y_test, y_pred)
    f1_scores.append(f1)
    #print(f1)
    n.append(30)
    plt.plot(n, f1_scores, label="F1 Scores")
    plt.xlabel('Dimensions')
    plt.ylabel('F1 Value')
    plt.grid()
    #plt.xticks(np.arange(min(n), max(n) + 1, 1.0))
    #plt.show()
    plt.savefig('f1scores.png')



def svm_rbf():
    X_train, X_test, Y_train, Y_test = process_data()
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=11), LogisticRegression(random_state=1))
    pipeline.fit(X_train, Y_train)
    svm1 = SVC(kernel='rbf')
    svm1.fit(X_train, Y_train)
    scores = cross_val_score(svm1, X_train, Y_train, cv=10, scoring='accuracy')
    print('\nRBF')
    print('Train Accuracy: %.3f' % svm1.score(X_train, Y_train))
    print('Test Accuracy: %.3f\n' % svm1.score(X_test, Y_test))
    kfold = StratifiedKFold(n_splits=10).split(X_train, Y_train)
    scores = []
    f1_scores= []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train], Y_train[train])
        score = pipeline.score(X_train[test], Y_train[test])
        scores.append(scores)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(Y_test, y_pred)
        f1_scores.append(f1)
        print('F1: %.3f' % f1)
        print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
    scores = cross_val_score(estimator=pipeline, X=X_train, y=Y_train, cv=10, n_jobs=1)
    print("CV accuracy scores: %s" % scores)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

def svm_linear():
    X_train, X_test, Y_train, Y_test = process_data()
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=11), LogisticRegression(random_state=1))
    pipeline.fit(X_train, Y_train)
    svm1 = SVC(kernel='linear')
    svm1.fit(X_train, Y_train)
    scores = cross_val_score(svm1, X_train, Y_train, cv=10, scoring='accuracy')
    print('\nLinear')
    print('Train Accuracy: %.3f' % svm1.score(X_train, Y_train))
    print('Test Accuracy: %.3f' % svm1.score(X_test, Y_test))
    kfold = StratifiedKFold(n_splits=10).split(X_train, Y_train)
    scores = []
    f1_scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train], Y_train[train])
        score = pipeline.score(X_train[test], Y_train[test])
        scores.append(scores)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(Y_test, y_pred)
        f1_scores.append(f1)
        print('F1: %.3f' % f1)
        print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
    scores = cross_val_score(estimator=pipeline, X=X_train, y=Y_train, cv=10, n_jobs=1)
    print("CV accuracy scores: %s" % scores)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))


def svm_poly():
    X_train, X_test, Y_train, Y_test = process_data()
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=11), LogisticRegression(random_state=1))
    pipeline.fit(X_train, Y_train)
    svm1 = SVC(kernel='poly')
    svm1.fit(X_train, Y_train)
    scores = cross_val_score(svm1, X_train, Y_train, cv=10, scoring='accuracy')
    print('\nPolynomial')
    print('Train Accuracy: %.3f' % svm1.score(X_train, Y_train))
    print('Test Accuracy: %.3f' % svm1.score(X_test, Y_test))
    kfold = StratifiedKFold(n_splits=10).split(X_train, Y_train)
    scores = []
    f1_scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train], Y_train[train])
        score = pipeline.score(X_train[test], Y_train[test])
        scores.append(scores)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(Y_test, y_pred)
        f1_scores.append(f1)
        print('F1: %.3f' % f1)
        print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
    scores = cross_val_score(estimator=pipeline, X=X_train, y=Y_train, cv=10, n_jobs=1)
    print("CV accuracy scores: %s" % scores)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
# pca(3)
# pca(5)
# pca(7)
# pca(9)
# pca(11)
decision_tree()
svm_rbf()
svm_linear()
svm_poly()
