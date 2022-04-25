import pandas as pd
import sys
import os
# Import the path where the data
path = sys.argv[1]
target = sys.argv[2]
for file_name in os.listdir(path):
    if file_name.startswith("GSE"):
        print(file_name)
        # Clinical
        clin = pd.read_csv(
            path+"/clin_"+file_name, sep="\t")

        clin = clin.T.reset_index()
        clinical = {}
        for i in range(1, clin.shape[0]):
            infor = {}
            for k in clin.iloc[i][1:]:
                try:
                    data = k.split(":")
                    infor[data[0]] = data[1]
                except:
                    pass
            clinical[clin["index"].iloc[i]] = infor
        clin = pd.DataFrame.from_dict(clinical).T
        # Data
        data = pd.read_csv(
            path+"/data_"+file_name, sep="\t", index_col=0)
        data = data.iloc[:-1, :]
        data = data.T
        data = data.join(clin.filter(regex=(target)))
        # Rename target columns
        data = data.rename(columns={data.columns[-1]: "target"})
        # Save
        clin.to_csv(path+"/processed_clin_"+file_name, sep="\t")
        data.to_csv(path+"/processed_data_"+file_name, sep="\t")
