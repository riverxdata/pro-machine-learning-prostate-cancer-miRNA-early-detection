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
from sklearn.ensemble import RandomForestClassifier
# Visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt
    


clf=RandomForestClassifier(bootstrap=True,criterion='gini',max_features='auto',n_estimators=15,random_state=42)
# Define parameter
test_size=0.5
scaler=MinMaxScaler()
folds=5

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



# Print selected columnns
cols=["MIMAT0005582","MIMAT0022259"]

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
X_whole=scaler.transform(X)
y_whole=y
clf.fit(X_train,y_train)
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
    cm = metrics.confusion_matrix(y, y_pred)
    try:
        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(X)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)
        auc = metrics.roc_auc_score(y, y_pred_proba)
    except:
        auc = "NA"
    return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,"fpr":fpr,"tpr":tpr, "auc": auc, "PPV": PPV, "NPV": NPV, "f1": f1, "kappa": kappa,"cm":cm}


value=[X_train,X_test,X_valid,y_train,y_test,y_valid,folds,"All_10_miRNA"]
path="results/"


def Visualization(model,x,y,data_type="train"):
    # Evaluation
    result = performance(model,x,y)
    sns.set_theme(style="white")
    fig, (ax2) = plt.subplots(1, 1)
    fig.suptitle(result["auc"], fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(7)
    fig.set_facecolor('white')

    ax2.plot(result['fpr'],result['tpr'])
    # Configure x and y axis
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')

    # Create legend & title
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    sns.despine()
    # Save figure
    plt.savefig("figures/"+data_type+"_ROC.png")
    plt.clf()

    df_cm = pd.DataFrame(result["cm"], range(result["cm"].shape[0]), range(result["cm"].shape[0]))
    df_cm= df_cm.iloc[[1,0],[1,0]]
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={
                "size": 30}, fmt='g')  # font size
    plt.savefig("figures/"+data_type+"_Confusion_Matrix.png")
    plt.clf()

Visualization(clf,X_train,y_train,data_type="train")
Visualization(clf,X_test,y_test,data_type="test")
Visualization(clf,X_valid,y_valid,data_type="valid")
Visualization(clf,X_whole,y_whole,data_type="whole")