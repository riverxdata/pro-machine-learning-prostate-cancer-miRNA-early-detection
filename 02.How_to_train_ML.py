"""
In this section, includes:
1. Split data: train and test with test size: 0.5
2. Normalize data: z score + min_max
3. Feature selection: Choose the features that fit the model
4. Train models with simple methods
5. Return models and use later for validation
"""
import pandas as pd
# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Models
from sklearn.ensemble import GradientBoostingClassifier

# Evaludation
from sklearn import metrics 

# Define parameter
test_size=0.5
scaler=StandardScaler()
select_top=10


# Read data
train_test=pd.read_csv("/home/nguyen/Desktop/Projects/Gastric-Cancer-Early-Detection/data/5genes_train26253_median.txt",sep="\t")



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

# anova

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
train_test=train_test[inputs]
# Split data
y = train_test["target"]
X = train_test.drop(["target"], axis=1)

# Randome state to make sure all of our replicate result shoudl receive the same dataset.
# If not, the size may be still the same but get different samples (suffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Transform data
# X_train = 
scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
X_train=scaler.transform(X_train)

# Train models
clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)

# Predict
clf.predict(X_test)
# array([0, 0, 1, ..., 0, 0, 0])

# Basic evaluation
y_pred = clf.predict(X_train)
tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred).ravel()

print(tn,fp,fn,tp)

accuracy = (tp+tn)/(tn+fp+fn+tp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

print(accuracy,sensitivity,specificity)

# Build function
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

# Test function
# train
performance(clf,X_train,y_train)
# test
performance(clf,X_test,y_test)
# valid
# Absolutely wrong, need to scale before running the prediction
performance(clf,X,y)

# Correct
X_all=scaler.transform(X)
performance(clf,X_all,y)


