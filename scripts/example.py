from numpy import number
import scripts.machine_learning as ml
import pandas as pd
# Process data
# Input data
train_test = pd.read_csv(
    "./example_data/processed_data_GSE112264_series_matrix.txt", sep="\t", index_col=0)
validation = pd.read_csv(
    "./example_data/processed_data_GSE113486_series_matrix.txt", sep="\t", index_col=0)

train_test.head()
#             MIMAT0000062  MIMAT0000063  MIMAT0000064  MIMAT0000065  MIMAT0000066  ...  MIMAT0033692  MIMAT0035542  MIMAT0035703  MIMAT0035704                 target
# GSM3063093        -0.297        -0.297        -0.297         3.977        -0.297  ...         5.389        -0.297        -0.297        -0.297   Biliary Tract Cancer
# GSM3063094         1.030         3.489        -0.838         4.832         3.161  ...         5.964         4.810        -0.838         3.180   Biliary Tract Cancer
# GSM3063095         4.599        -0.292         3.997         2.016         4.105  ...         2.263         2.526        -0.292        -0.292   Biliary Tract Cancer
# GSM3063096         4.669         3.523         6.537         5.673        -0.529  ...         5.845         3.527        -0.529        -0.529   Biliary Tract Cancer
# GSM3063097         3.339         4.485         3.153         3.787         4.690  ...         5.286         3.961         0.314         0.314   Biliary Tract Cancer

validation.head()
#             MIMAT0000062  MIMAT0000063  MIMAT0000064  MIMAT0000065  MIMAT0000066  MIMAT0000067  ...  MIMAT0032116  MIMAT0033692  MIMAT0035542  MIMAT0035703  MIMAT0035704           target
# GSM3106847        -1.061        -1.061         2.303        -1.061         2.972         4.457  ...         6.507         3.906        -1.061        -1.061        -1.061   Bladder Cancer
# GSM3106848         0.765         0.765         4.920         0.765         0.765         0.765  ...         5.946         0.765         0.765         0.765         0.765   Bladder Cancer
# GSM3106849         2.949         3.451         0.420         2.594         1.034        -1.492  ...         6.058         4.482         2.917        -1.492        -1.492   Bladder Cancer
# GSM3106850         3.033         6.224         3.496         4.870         3.977         0.867  ...         6.315         2.759         5.028         0.867         4.042   Bladder Cancer
# GSM3106851         4.832         5.349         5.571         6.055         1.237         1.237  ...         6.788         6.695         5.987         1.237         5.985   Bladder Cancer


def classification_train(df):
    if df["target"] == " Prostate Cancer":
        return 1
    elif (df["target"] == ' Negative prostate biopsy') | (df["target"] == " non-Cancer"):
        return 0
    else:
        return -1


def classification_valid(df):
    if df["target"] == " Prostate Cancer":
        return 1
    elif df["target"] == ' Non-cancer control':
        return 0
    else:
        return -1


train_test["target"] = train_test.apply(classification_train, axis=1)
validation["target"] = validation.apply(classification_valid, axis=1)
validation = validation[validation["target"] != -1]
train_test = train_test[train_test["target"] != -1]


# What you need to known
# train_test and validation data
train_test = train_test
validation = validation
# top features with highest score
number_of_features = 10
# The test size you want to split from train_test rang from (0:1), for large data chooose 0.1, smaller try 0.2-0.4
test_size = 0.5
# Method for scale data, if equal to Standard it will be z-score normalization else will be MinMaxScaler
method = "Standard"
# num_folds= number of fold validation which mean it will validation on the training data by 5 subset training, should be 5 or 10 if data is larger
number_of_folds = 5
# The path where the result will be save, create this folder by hand
path = "/home/nguyen/Desktop/Tools/Machine-Learning-BenchMarking-Classification/example_data/result"
# 01. Feature selection/ actually it immplemented to the second step already just make sure how does it work
features = ml.select_features(
    train_test, number_of_features, test_size, method)
features
# Index(['MIMAT0000071', 'MIMAT0005792', 'MIMAT0005880', 'MIMAT0018978',
#        'MIMAT0022259', 'MIMAT0022713', 'MIMAT0022838', 'MIMAT0022924',
#        'MIMAT0023701', 'MIMAT0027580'],
#       dtype='object')
# 02. Prepare data
all_inputs = ml.data_prepare(
    train_test, validation, test_size, number_of_folds, number_of_features, method)
# 03. Run model
ml.parallel(
    all_inputs, "/home/nguyen/Desktop/Tools/Machine-Learning-BenchMarking-Classification/example_data/results", 4)
