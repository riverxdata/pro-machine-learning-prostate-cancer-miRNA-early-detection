"""
In this section, includes:
1. Training models with hyper parameter turing
It is just the optimization base on combination of parameters from single model machine learning
"""
import pandas as pd
import numpy as np
# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Models
from sklearn.neighbors import KNeighborsClassifier
# Evaludation
from sklearn import metrics 
# Parameters optimization
from sklearn.model_selection import GridSearchCV
# Performance
from sklearn.model_selection import cross_val_score

# Define parameter
test_size=0.5
scaler=MinMaxScaler()
select_top=10
folds=10

# Read data
train_test=pd.read_csv("data/train_test.csv",index_col=0)

# Split data
y = train_test["target"]
X = train_test.drop(["target"], axis=1)

# Randome state to make sure all of our replicate result shoudl receive the same dataset.
# If not, the size may be still the same but get different samples (suffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Transform data
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
# For novel data just use: scaler.transform(new_data)

# Feature selection: Should be performed on only trainning data
selector = SelectKBest(score_func=f_classif, k=select_top)
selector.fit(X_train, y_train)
cols = selector.get_support(indices=True)
cols = X.iloc[:, cols].columns

# Print selected columnns
cols
# Index(['MIMAT0005582', 'MIMAT0005880', 'MIMAT0016878', 'MIMAT0019776',
#        'MIMAT0019957', 'MIMAT0022259', 'MIMAT0027392', 'MIMAT0027430',
#        'MIMAT0027654', 'MIMAT0031000'],
#       dtype='object')

# Select input
inputs=list(cols)
inputs.append("target")

# Load data again with selected columns
train_test=pd.read_csv("data/train_test.csv",index_col=0)
validation=pd.read_csv("data/validation.csv",index_col=0)
train_test=train_test[inputs]
validation=validation[inputs]

# Split data train_test
y = train_test["target"]
X = train_test.drop(["target"], axis=1)
# Split data validation
y_valid=validation["target"]
X_valid=validation.drop(["target"],axis=1)

# Randome state to make sure all of our replicate result shoudl receive the same dataset.
# If not, the size may be still the same but get different samples (suffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Transform data
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
X_valid=scaler.transform(X_valid)


# Build function
"""
This is used for evaluation with the performance of the models. Based on the input, it can be used for
train, test as well as validation
"""
def performance(model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
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


value=[X_train,X_test,X_valid,y_train,y_test,y_valid,folds,"All_10_miRNA"]
path="results/"

# Function for turing
def parameter_tuning(value, path):
    X_train, X_test, X_valid = value[0], value[1], value[2]
    y_train, y_test, y_valid = value[3], value[4], value[5]
    num_folds, feature_name = value[6], value[7]
    # Create models
    # Model 5
    knn_params = dict(n_neighbors=[2, 3, 5, 10], weights=['uniform', 'distance'],
                      algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], leaf_size=[5, 10, 15, 20])
    # Create the model
    params = [ knn_params]
    # classifiers to test
    classifiers = [KNeighborsClassifier()]

    names = ['KNearestNeighbors']

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
        path+feature_name+"_validation.csv", index=False)

# Running this one
parameter_tuning(value,path)

"""
After these all steps, we did:
1. Benchmark one methods for parameter turning
Nextstep:
2. Combination of : Normalizaton, Features, Other methods, Test Size and Fold Change
However, test size and fold change should be kept as constant
"""


