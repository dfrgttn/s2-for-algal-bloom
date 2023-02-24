## nessisary dependents
import numpy as np
import pandas as pd
import argparse

from IPython import embed

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description="""This script runs the preparation
                                 of the data of bloom data""")
    ap.add_argument("-data", "--data", required=False,
            default = "./Data_algal_bloom_v4.txt", help="path to data file")

    args = vars(ap.parse_args())

    ARG_DATA_PATH = args["data"]

    print("---------------------------------------------------")
    print("The data prep configs:")
    print("The data file is set to {}".format(ARG_DATA_PATH))
    print("---------------------------------------------------")

    # read the data
    df = pd.read_csv(ARG_DATA_PATH, delimiter='\t', encoding = "ISO-8859-1")
    #df = df[df["bloom"] == 1]
    df = df.reset_index()
    # transform km2 to log10 scale
    #df["Area_km2"] = np.log10(df["Area_km2"])
    #df["chla_ug_L"] = df["chla_ug_L"] / 200

    # identify the proportions of training and testing subsets of data
    lakes = df["monSiteCode"]
    unique_lakes = np.unique(lakes)
    n_unique_lakes = len(unique_lakes)
    n_train_lakes = int(n_unique_lakes * 0.7)
    n_test_lakes = n_unique_lakes - n_train_lakes


    # set the lakes split
    np.random.seed(seed=123)

    train_lake_idx = np.random.choice(n_unique_lakes, n_train_lakes, replace=False)
    test_lake_idx  = np.setdiff1d(list(range(n_unique_lakes)), train_lake_idx)

    train_lake_idx.sort()
    test_lake_idx.sort()

    # set specific lakes measuments to train/test

    train_idx = [i for i, x in enumerate(lakes) if x in unique_lakes[train_lake_idx]]
    test_idx  = np.setdiff1d(list(range(len(lakes))), train_idx)
    print("lakes splited to train/test", len(train_idx), len(test_idx))

    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    COLS_TO_USE = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B5_B4',
                   'B3_B2', 'B3_B4', 'B4_B5_B6', 'avw', 'B3_B5', 'diff_alg',
                   #'Area_km2', 'Shoreline_development', 'Type',
                   'chla_ug_L']

    df_test[COLS_TO_USE].to_csv("test_reg_data.csv", index=False)
    df_train[COLS_TO_USE].to_csv("train_reg_data.csv", index=False)

    df_test.to_csv("test_reg_data_orig.csv", index=False)
    df_train.to_csv("train_reg_data_orig.csv", index=False)
