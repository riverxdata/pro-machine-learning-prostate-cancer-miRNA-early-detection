import seaborn as sns
import matplotlib.pyplot as plt
# Save model
import joblib
import pickle
from pkg_resources import parse_requirements
# Run parralell
import pyblaze.multiprocessing as xmp
# Basic process
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Feature Selection
import itertools
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Can get the functions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
# Can not get the functions
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Parameters optimization
from sklearn.model_selection import GridSearchCV
# Performance
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# Ignore wanring
import warnings
warnings.filterwarnings("ignore")

# Section 1: Feature selections


def select_features(train_test, num_features, test_size, method):
    y = train_test["target"]
    X = train_test.drop(["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    # Transform
    if method == "Standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    # Training
    X_train = scaler.fit_transform(X_train)
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(X_train, y_train)
    cols = selector.get_support(indices=True)
    cols = X.iloc[:, cols].columns
    print(cols)
    return cols


# Section 2: Hyper parameter turing
#  Finding best parameters for the models
def parameter_tuning(value, path):
    X_train, X_test, X_valid = value[0], value[1], value[2]
    y_train, y_test, y_valid = value[3], value[4], value[5]
    num_folds, feature_name = value[6], value[7]
    # Create models
    # Model 1
    random_forest_params = dict(n_estimators=[5, 10, 15, 20, 25], criterion=['gini', 'entropy'], max_features=[
                                2, 3, 4, 'auto', 'log2', 'sqrt', None], bootstrap=[False, True])
    # Model 2
    decision_tree_params = dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], min_samples_split=[
                                2, 3, 4], max_features=[2, 3, 'auto', 'log2', 'sqrt', None], class_weight=['balanced', None], presort=[False, True])
    # Model 3
    # perceptron_params = dict(penalty=[None, 'l2', 'l1', 'elasticnet'], fit_intercept=[False, True], shuffle=[False, True],class_weight=['balanced', None], alpha=[0.0001, 0.00025], max_iter=[30,50,90])
    svm_params = dict(shrinking=[False, True], degree=[
                      3, 4], class_weight=['balanced', None])
    # neural_net_params = dict(activation=['identity', 'logistic', 'tanh', 'relu'], hidden_layer_sizes = [(20,15,10),(30,20,15,10),(16,8,4)],max_iter=[50,80,150], solver=['adam','lbfgs'], learning_rate=['constant', 'invscaling', 'adaptive'], shuffle=[True, False])
    # Model 4
    log_reg_params = dict(class_weight=['balanced', None], solver=[
                          'newton-cg', 'lbfgs', 'liblinear', 'sag'], fit_intercept=[True, False])
    # Model 5
    knn_params = dict(n_neighbors=[2, 3, 5, 10], weights=['uniform', 'distance'],
                      algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], leaf_size=[5, 10, 15, 20])
    # Model 6
    bagging_params = dict(
        n_estimators=[5, 12, 15, 20], bootstrap=[False, True])
    # Model 7
    ada_boost_params = dict(
        n_estimators=[50, 75, 100], algorithm=['SAMME', 'SAMME.R'])
    # Model 8
    guassiannb_params = dict()
    # Model 9
    gradient_boosting_params = dict(n_estimators=[15, 25, 50])
    # Create the model
    params = [
        random_forest_params, decision_tree_params,
        svm_params, log_reg_params, knn_params,
        bagging_params, ada_boost_params, guassiannb_params, gradient_boosting_params
    ]
    # classifiers to test
    classifiers = [
        RandomForestClassifier(), DecisionTreeClassifier(),
        SVC(probability=True), LogisticRegression(), KNeighborsClassifier(),
        BaggingClassifier(), AdaBoostClassifier(), GaussianNB(), GradientBoostingClassifier()
    ]

    names = [
        'RandomForest', 'DecisionTree',
        'SVM', 'LogisticRegression', 'KNearestNeighbors',
        'Bagging', 'AdaBoost', 'Naive-Bayes', 'GradientBoosting'
    ]
    models = dict(zip(names, zip(classifiers, params)))
    print(num_folds, 'fold cross-validation is used')
    results_train_train = []
    results_train_test = []
    results_train_valid = []
    # dataframe to store intermediate results
    for name, clf_and_params in models.items():
        # Handling error when the features is not come up with models
        try:
            print('Computing GridSearch on {} '.format(name))
            clf, clf_params = clf_and_params
            # Parameter and training
            grid_clf = GridSearchCV(
                estimator=clf, param_grid=clf_params, cv=num_folds)
            grid_clf = grid_clf.fit(X_train, y_train)

            # Train
            perform_train = performance(grid_clf, X_train, y_train)
            # Testing
            perform_test = performance(grid_clf, X_test, y_test)
            # Validation
            perform_valid = performance(grid_clf, X_valid, y_valid)

            cv_scores = cross_val_score(clf, X_train, y_train, cv=num_folds)

            # Performance
            perform_train.update({"cv": np.mean(cv_scores), "name": feature_name,
                                  "model_name": name, "best_params": grid_clf.best_params_})
            perform_valid.update({"cv": np.mean(cv_scores), "name": feature_name,
                                  "model_name": name, "best_params": grid_clf.best_params_})
            perform_test.update({"cv": np.mean(cv_scores), "name": feature_name,
                                "model_name": name, "best_params": grid_clf.best_params_})

            # Results
            results_train_train.append(perform_train)
            results_train_test.append(perform_test)
            results_train_valid.append(perform_valid)
        except:
            pass
    pd.DataFrame(results_train_train).to_csv(
        path+feature_name+"_train.csv", index=False)
    pd.DataFrame(results_train_test).to_csv(
        path+feature_name+"_test.csv", index=False)
    pd.DataFrame(results_train_valid).to_csv(
        path+feature_name+"_valid.csv", index=False)


def parameter_tuning_retrain(value, path):
    X_train, X_test, X_valid, X_whole = value[0], value[1], value[2], value[3]
    y_train, y_test, y_valid, y_whole = value[4], value[5], value[6], value[7]
    num_folds, feature_name = value[8], value[9]
    # Create models
    # Model 1
    random_forest_params = dict(n_estimators=[5, 10, 15, 20, 25], criterion=['gini', 'entropy'], max_features=[
                                2, 3, 4, 'auto', 'log2', 'sqrt', None], bootstrap=[False, True])
    # Model 2
    decision_tree_params = dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], min_samples_split=[
                                2, 3, 4], max_features=[2, 3, 'auto', 'log2', 'sqrt', None], class_weight=['balanced', None], presort=[False, True])
    # Model 3
    # perceptron_params = dict(penalty=[None, 'l2', 'l1', 'elasticnet'], fit_intercept=[False, True], shuffle=[False, True],class_weight=['balanced', None], alpha=[0.0001, 0.00025], max_iter=[30,50,90])
    svm_params = dict(shrinking=[False, True], degree=[
                      3, 4], class_weight=['balanced', None])
    # neural_net_params = dict(activation=['identity', 'logistic', 'tanh', 'relu'], hidden_layer_sizes = [(20,15,10),(30,20,15,10),(16,8,4)],max_iter=[50,80,150], solver=['adam','lbfgs'], learning_rate=['constant', 'invscaling', 'adaptive'], shuffle=[True, False])
    # Model 4
    log_reg_params = dict(class_weight=['balanced', None], solver=[
                          'newton-cg', 'lbfgs', 'liblinear', 'sag'], fit_intercept=[True, False])
    # Model 5
    knn_params = dict(n_neighbors=[2, 3, 5, 10], weights=['uniform', 'distance'],
                      algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], leaf_size=[5, 10, 15, 20])
    # Model 6
    bagging_params = dict(
        n_estimators=[5, 12, 15, 20], bootstrap=[False, True])
    # Model 7
    ada_boost_params = dict(
        n_estimators=[50, 75, 100], algorithm=['SAMME', 'SAMME.R'])
    # Model 8
    guassiannb_params = dict()
    # Model 9
    gradient_boosting_params = dict(n_estimators=[15, 25, 50])
    # Create the model
    params = [
        random_forest_params, decision_tree_params,
        svm_params, log_reg_params, knn_params,
        bagging_params, ada_boost_params, guassiannb_params, gradient_boosting_params
    ]
    # classifiers to test
    classifiers = [
        RandomForestClassifier(), DecisionTreeClassifier(),
        SVC(probability=True), LogisticRegression(), KNeighborsClassifier(),
        BaggingClassifier(), AdaBoostClassifier(), GaussianNB(), GradientBoostingClassifier()
    ]

    names = [
        'RandomForest', 'DecisionTree',
        'SVM', 'LogisticRegression', 'KNearestNeighbors',
        'Bagging', 'AdaBoost', 'Naive-Bayes', 'GradientBoosting'
    ]
    models = dict(zip(names, zip(classifiers, params)))
    print(num_folds, 'fold cross-validation is used')
    results_train_train = []
    results_train_test = []
    results_train_valid = []
    # dataframe to store intermediate results
    for name, clf_and_params in models.items():
        # Handling error when the features is not come up with models
        try:
            print('Computing GridSearch on {} '.format(name))
            clf, clf_params = clf_and_params
            # Parameter and training
            grid_clf = GridSearchCV(
                estimator=clf, param_grid=clf_params, cv=num_folds)
            grid_clf = grid_clf.fit(X_train, y_train)

            joblib.dump(grid_clf, path + "best_model.sav")

            whole_result = evaluate_model(grid_clf, X_whole, y_whole)
            # Train
            # perform_train = performance(grid_clf, X_train, y_train)
            # # Testing
            # perform_test = performance(grid_clf, X_test, y_test)
            # # Validation
            # perform_valid = performance(grid_clf, X_valid, y_valid)

            # cv_scores = cross_val_score(clf, X_train, y_train, cv=num_folds)

            # # Performance
            # perform_train.update({"cv": np.mean(cv_scores), "name": feature_name,
            #                       "model_name": name, "best_params": grid_clf.best_params_})
            # perform_valid.update({"cv": np.mean(cv_scores), "name": feature_name,
            #                       "model_name": name, "best_params": grid_clf.best_params_})
            # perform_test.update({"cv": np.mean(cv_scores), "name": feature_name,
            #                     "model_name": name, "best_params": grid_clf.best_params_})

            # # Results
            # results_train_train.append(perform_train)
            # results_train_test.append(perform_test)
            # results_train_valid.append(perform_valid)
        except:
            pass
    sns.set_theme(style="white")
    fig, (ax2) = plt.subplots(1, 1)
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(7)
    fig.set_facecolor('white')

    ax2.plot(whole_result['fpr'], whole_result['tpr'],
             label=feature_name)
    # Configure x and y axis
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')

    # Create legend & title
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc=4)
    sns.despine()
    print(path+name+".png")
    plt.savefig(
        "/home/nguyen/Desktop/Tools/Machine-Learning-BenchMarking-Classification/example_data/result.png")
    pd.DataFrame(results_train_train).to_csv(
        path+feature_name+"_train.csv", index=False)
    pd.DataFrame(results_train_test).to_csv(
        path+feature_name+"_test.csv", index=False)
    pd.DataFrame(results_train_valid).to_csv(
        path+feature_name+"_valid.csv", index=False)
# Get the combination of features all
    return whole_result


# Section 3: Making combination inputs
def combinations(features, combinations):
    features = list(itertools.combinations(features, combinations))
    return features

# Model name


def model_name(selected_features):
    return ' '.join(selected_features)


# Section 4:Performance by matrices
def performance(model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    f1 = metrics.f1_score(y, y_pred)
    kappa = metrics.cohen_kappa_score(y, y_pred)
    try:
        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(X)[::, 1]
        auc = metrics.roc_auc_score(y, y_pred_proba)
    except:
        auc = "NA"
    return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "auc": auc, "PPV": PPV, "NPV": NPV, "f1": f1, "kappa": kappa}

# Section 5: Prepare data


def data_prepare(train_test, validation, num_folds=5, num_features=20, test_size=0.1, method=StandardScaler()):
    features = select_features(
        train_test, num_features, test_size, method)
    combinations_all = []
    for i in range(1, len(features)+1):
        combinations_all.extend(list(itertools.combinations(features, i)))
    features = combinations_all
    # features = columns
    models = []
    for i in features:
        selected_features = []
        for k in i:
            selected_features.append(k)
        feature_name = model_name(selected_features)
        selected_features.append("target")
        print("Processing on:", feature_name)
        data1 = train_test[selected_features]
        data2 = validation[selected_features]
        # Splitting the data
        y = data1["target"]
        X = data1.drop(["target"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        # Transformation
        if method == "Standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Validation
        X_valid = data2.drop(["target"], axis=1)
        X_valid = scaler.transform(X_valid)
        y_valid = data2["target"]
        models.append([X_train, X_test, X_valid, y_train,
                      y_test, y_valid, num_folds, feature_name])
    return models

# Running parralel


def parallel(value, output, num_workers):
    tokenizer = xmp.Vectorizer(parameter_tuning, num_workers=num_workers)
    return tokenizer.process(value, output)


def parallel_retrain(value, output, num_workers):
    tokenizer = xmp.Vectorizer(
        parameter_tuning_retrain, num_workers=num_workers)
    return tokenizer.process(value, output)

# Summary result and collect the models


def concat_files(output, key):
    frame = []
    for i in os.listdir(output):
        if i.endswith(key):
            frame.append(pd.read_csv(
                output+i))
    frame = pd.concat(frame, axis=0)
    frame = frame.drop_duplicates()
    return frame


def save_result(output):
    # train
    train = concat_files(output, "train.csv")
    # test
    test = concat_files(output, "test.csv")
    # valid
    valid = concat_files(output, "validation.csv")
    # Merge all
    train["dataset"] = "01_train"
    test["dataset"] = "02_test"
    valid["dataset"] = "03_valid"

    all = pd.concat([train, test, valid], axis=0)

    all["number_of_features"] = all["name"].apply(lambda x: len(x.split(" ")))

    # Select model with mean in train, test and valid achive best result with not much difference
    df = all.groupby(["model_name", "name"]).mean().reset_index()
    df = df[df.groupby("number_of_features")["f1"].transform(max) == df["f1"]]
    df = df[['number_of_features', 'name', 'model_name', 'accuracy', 'sensitivity', 'specificity',
            'auc',  'PPV', 'NPV', 'f1', 'kappa', 'cv', ]]
    df = df.sort_values("number_of_features")
    all["class"] = all["name"]+":"+all["model_name"]
    df["class"] = df["name"]+":"+df["model_name"]

    final = all.loc[all['class'].isin(df["class"])]
    final = final.sort_values(
        ["number_of_features", "dataset"], ascending=[1, 1])
    final = final[['number_of_features', "dataset", 'name', 'model_name', "best_params", 'accuracy', 'sensitivity', 'specificity',
                   'auc',  'PPV', 'NPV', 'f1', 'kappa', 'cv', ]]
    final.to_csv(output+"result_all.tsv", sep="\t", index=False)

    table1 = final[["number_of_features", "name",
                    "model_name", "best_params"]].drop_duplicates()
    table1.to_csv(
        output+"models_info.tsv", sep="\t", index=False)


def evaluate_model(model, x_test, y_test):
    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)

    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

# Save model using pickle


def save_models(output, train_test, validation, num_folds=5, num_of_selected_features=2, test_size=0.1, method="Standard"):
    result = pd.read_csv(output+"models_info.tsv", sep="\t")
    result = result[result["number_of_features"] == num_of_selected_features]
    print(result["name"].values[0])
    features = result["name"].values[0].split(" ")
    features.append("target")
    data1 = train_test[features]
    data2 = validation[features]
    # Splitting the data
    y = data1["target"]
    X = data1.drop(["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    # Transformation
    if method == "Standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_whole = scaler.transform(X)
    # Validation
    X_valid = data2.drop(["target"], axis=1)
    X_valid = scaler.transform(X_valid)
    y_valid = data2["target"]
    y_whole = y
    models = []
    models.append([X_train, X_test, X_valid, X_whole,
                   y_train, y_test, y_valid, y_whole,  num_folds, result["name"].values[0]])
    return parameter_tuning_retrain(models[0], output)


# joblib.dump(dtc1,"model_2_gene.sav")
# joblib.dump(dtc2,"Cd.sav")
# joblib.dump(dtc3,"csf1r.sav")
# joblib.dump(dtc4,"score.sav")
