import os
import pandas as pd

path = "results/result_50_minmax/"

# validation
frame = []
for i in os.listdir(path):
    if i.endswith("validation.csv"):
        frame.append(pd.read_csv(path+i))
valid = pd.concat(frame, axis=0)
valid = valid.drop_duplicates()

# testing
frame = []
for i in os.listdir(path):
    if i.endswith("test.csv"):
        frame.append(pd.read_csv(
            path+i))
test = pd.concat(frame, axis=0)
test = test.drop_duplicates()

# train
frame = []
for i in os.listdir(path):
    if i.endswith("train.csv"):
        frame.append(pd.read_csv(
            path+i))
train = pd.concat(frame, axis=0)
train = train.drop_duplicates()

# Merge all
train["dataset"] = "atrain"
test["dataset"] = "btest"
valid["dataset"] = "cvalidation"

all = pd.concat([train, test, valid], axis=0)

all["number_of_features"] = all["name"].apply(lambda x: len(x.split(" ")))


df = all.groupby(["model_name", "name"]).mean().reset_index()
df = df[df.groupby("number_of_features")["f1"].transform(max) == df["f1"]]

df = df[['number_of_features', 'name', 'model_name', 'accuracy', 'sensitivity', 'specificity',
         'auc',  'PPV', 'NPV', 'f1', 'kappa', 'cv', ]]

df = df.sort_values("number_of_features")

all["class"] = all["name"]+":"+all["model_name"]

df["class"] = df["name"]+":"+df["model_name"]

final = all.loc[all['class'].isin(df["class"])]

final = final.sort_values(["number_of_features", "dataset"], ascending=[1, 1])

final = final[['number_of_features', "dataset", 'name', 'model_name', "best_params", 'accuracy', 'sensitivity', 'specificity',
               'auc',  'PPV', 'NPV', 'f1', 'kappa', 'cv', ]]

final.to_csv(path+"minmax50.tsv", sep="\t", index=False)

table1 = final[["number_of_features", "name","model_name", "best_params"]].drop_duplicates()
table1.to_csv(path+"Table1.tsv", sep="\t", index=False)

table2 = final[["number_of_features", "cv"]].drop_duplicates()
table2.T.to_csv(path+"cross_validation.tsv", sep="\t", index=False)
