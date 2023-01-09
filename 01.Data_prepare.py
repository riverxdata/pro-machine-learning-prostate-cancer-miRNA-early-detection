"""
Summary project:
- The data are collected from NCBI GEO from 2 indepdent datasets
- The miRNA collected from serum (blood) measured by miRNA microarray technology with 2565 (maybe) probes ( 1 probe n)
- They are all normalized (2 datasets had the same author)
- 
"""

import pandas as pd

# Import data
validation= pd.read_csv(
    "data/processed_data_GSE112264_series_matrix.txt", index_col=0, sep="\t")
train_test= pd.read_csv(
   "data/processed_data_GSE164174_series_matrix.txt", index_col=0, sep="\t")


# Observe data
train_test.head()
#             MIMAT0000062  MIMAT0000063  MIMAT0000064  MIMAT0000065  MIMAT0000066  ...  MIMAT0033692  MIMAT0035542  MIMAT0035703  MIMAT0035704           target
# GSM4998853      6.266276      5.624105      5.809890      5.469135      5.164521  ...      6.094850      5.981251      4.386562     -0.599270   Gastric Cancer
# GSM4998854      6.001691      6.176736      4.835808      4.515228      2.268467  ...      5.779526      5.374496      4.016351     -0.280957   Gastric Cancer
# GSM4998855      6.045082      4.893886      6.912571      6.340629     -0.058757  ...      4.343939      5.942373     -0.058757      2.466506   Gastric Cancer
# GSM4998856      5.676545      6.333787      5.889745      5.713336      4.379869  ...      6.327720      4.280188     -0.298893     -0.298893   Gastric Cancer
# GSM4998857      5.542557      5.961992      4.938783      5.299568      4.740694  ...      4.626803      4.990163      2.824906     -0.480958   Gastric Cancerb
validation.head()
#             MIMAT0000062  MIMAT0000063  MIMAT0000064  MIMAT0000065  MIMAT0000066  ...  MIMAT0033692  MIMAT0035542  MIMAT0035703  MIMAT0035704                 target
# GSM3063093        -0.297        -0.297        -0.297         3.977        -0.297  ...         5.389        -0.297        -0.297        -0.297   Biliary Tract Cancer
# GSM3063094         1.030         3.489        -0.838         4.832         3.161  ...         5.964         4.810        -0.838         3.180   Biliary Tract Cancer
# GSM3063095         4.599        -0.292         3.997         2.016         4.105  ...         2.263         2.526        -0.292        -0.292   Biliary Tract Cancer
# GSM3063096         4.669         3.523         6.537         5.673        -0.529  ...         5.845         3.527        -0.529        -0.529   Biliary Tract Cancer
# GSM3063097         3.339         4.485         3.153         3.787         4.690  ...         5.286         3.961         0.314         0.314   Biliary Tract Cancer

# Check labels
train_test["target"].value_counts() 
#  Gastric Cancer       1417
#  Non-cancer C          505
#  Non-cancer A          487
#  Non-cancer B          425
#  Colorectal Cancer      50
#  Esophageal Cancer      50
# Name: target, dtype: int64
validation["target"].value_counts()
#  Prostate Cancer             809
#  Negative prostate biopsy    241
#  Biliary Tract Cancer         50
#  Bladder Cancer               50
#  Colorectal Cancer            50
#  Esophageal Cancer            50
#  Gastric Cancer               50
#  Glioma                       50
#  Hepatocellular Carcinoma     50
#  Lung Cancer                  50
#  Pancreatic Cancer            50
#  Sarcoma                      50
#  non-Cancer                   41

# Tranform data
# Note: Can use library to convert label to number, however,two datasets have different numbers
def transform_data(df,column,encode_1,encode_0):
    if encode_1.count(df[column])==1:
        return 1
    elif encode_0.count(df[column])==1:
        return 0
    else: return -1
    
# Transform for train_test
label_1_train_test=[" Gastric Cancer"]
label_0_train_test=[" Non-cancer C"," Non-cancer B"," Non-cancer A"]
train_test["target"] = train_test.apply(lambda x: transform_data(x,"target",label_1_train_test,label_0_train_test), axis=1)

# Transform for validation
label_1_validation=[" Gastric Cancer"]
label_0_validaiton=[" non-Cancer"]
validation["target"] = validation.apply(lambda x: transform_data(x,"target",label_1_validation,label_0_validaiton), axis=1)

# Double check
validation.target.value_counts()
# -1    1500
#  1      50    
#  0      41
train_test.target.value_counts()
#  1    1417
#  0    1417
# -1     100

# Ignore -1 value
validation = validation[validation["target"] != -1]
train_test = train_test[train_test["target"] != -1]

# Export data
train_test.to_csv("data/train_test.csv")
validation.to_csv("data/validation.csv")