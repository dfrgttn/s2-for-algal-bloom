## nessisary dependents
from glob2 import glob
import numpy as np
import math
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

from utils import get_dataset_from_csv
from IPython import embed

if __name__ == '__main__':


    ap = argparse.ArgumentParser(description="""This script runs the testing 
                                 of all machine learning methods of bloom 
                                 prediction""")
    ap.add_argument("-lake", "--lake", required=False,
            default = 80, help="lake selection ")
    ap.add_argument("-idx1", "--col1", required=False,
            default = 0, help="first variable ")
    ap.add_argument("-idx2", "--col2", required=False,
            default = 1, help="second  variable ")
    ap.add_argument("-data", "--data", required=False,
                    default = "./train_data.csv", 
                    help="modelling data")
    ap.add_argument("-path", "--models_path", required=False,
            default = "./models", help="path to models")
    ap.add_argument("-row", "--row_id", required=False,
            default = 80, help="row of dataset")

    args = vars(ap.parse_args())

    ARG_DATA_FILE = args["data"]
    ARG_MODELS_PATH = args["models_path"]
    ARG_ROW_ID = args["row_id"]
    ARG_IDX_1 = int(args["col1"])
    ARG_IDX_2 = int(args["col2"])
    ARG_LAKE = int(args["lake"])

    print("---------------------------------------------------")
    print("The experiment configs:")
    print("The data is set to {}".format(ARG_DATA_FILE))
    print("The model path is set to {}".format(ARG_MODELS_PATH))
    print("The row is set to {}".format(ARG_ROW_ID))
    print("---------------------------------------------------")

    models = glob(ARG_MODELS_PATH + "/*")
    print("Founded models:")
    for i, model in enumerate(models):
        print(i, model)


    data = pd.read_csv(ARG_DATA_FILE)#, delimiter="\t", encoding = "ISO-8859-1")

    CSV_HEADER = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B5_B4', 
                   'B3_B2', 'B3_B4', 'B4_B5_B6', 'avw', 'B3_B5', 'diff_alg', 
                   #'Area_km2', 'Shoreline_development', 'Type', 
                   'bloom']
    COLUMN_DEFAULTS = [[x] for x in list(np.zeros(len(CSV_HEADER)))]

    print(CSV_HEADER)

    print(COLUMN_DEFAULTS)
 
    sample = data[CSV_HEADER].iloc[ARG_LAKE:ARG_LAKE+1]
    sample = data[CSV_HEADER].agg(['mean'])


    COL_1 = [CSV_HEADER[ARG_IDX_1]]
    COL_2 = [CSV_HEADER[ARG_IDX_2]]

    col_1_range = np.linspace(data[COL_1].min(),data[COL_2].max(), 50)
    col_2_range = np.linspace(data[COL_2].min(),data[COL_2].max(), 50)

    #newdf = pd.DataFrame(np.repeat(sample.values, 
    #                               len(col_1_range)*len(col_2_range), axis=0))
    newdf = pd.concat([sample]*len(col_1_range)*len(col_2_range))
    newdf.reset_index(level=0, inplace=True)
    newdf.drop("index", inplace= True, axis = 1)
    print(newdf.shape)
    k = 0 

    #embed()
    col_1_list = []
    col_2_list = []
    for a in col_1_range:
        for b in col_2_range:
            newdf.loc[k,COL_1] = a
            newdf.loc[k,COL_2] = b
            col_1_list.append(a)
            col_2_list.append(b)
            k = k + 1

    OUT_FILE = "tmp.file"
    newdf.to_csv(OUT_FILE, index=False)
    #print(newdf.head())
    #print(newdf.tail())

    batch_size = 1000

    eval_dataset = get_dataset_from_csv(
        OUT_FILE, 
        CSV_HEADER,
        "bloom",
        shuffle=True, batch_size=batch_size
    ) 

    DIM_X = len(col_1_range)
    DIM_Y = len(col_2_range)

    X1 = np.array(col_1_list)
    Y1 = np.array(col_2_list)
    #Z1 = yhat

    x1 = X1.reshape((DIM_X, DIM_Y))
    y1 = Y1.reshape((DIM_X, DIM_Y))
    #z1 = Z1.reshape((DIM_X, DIM_Y))

    xmin = np.min(x1)
    xmax = np.max(x1)
    ymin = np.min(y1)
    ymax = np.max(y1)
    zmin = 0.0
    zmax = 1.0

    failed_list = []
    for model in models:
        try:
            m0 = keras.models.load_model(model)
            yhat = m0.predict(eval_dataset)

            Z1 = yhat
            z1 = Z1.reshape((DIM_X, DIM_Y))
            #zmin = np.min(z1)
            #zmax = np.max(z1)


            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(x1, y1, z1, zorder = 0.25)#, alpha = 1, rstride=1, cstride=1, cmap=cm.autumn,linewidth=0.5, antialiased=True, zorder = 0.3)

            zticks = np.round(np.linspace(zmin, int(zmax), 5), 2)
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticks, fontsize=13)


            ax.set_xlabel(COL_1, rotation=90, fontsize=15)
            ax.set_ylabel(COL_2, rotation=90, fontsize=15)
            ax.set_zlabel("prediction", rotation=90, fontsize=15)

            ax.view_init(elev=45, azim=90+25+90)

            #plt.tight_layout()

            plt.savefig('./out/{}-{}-{}.png'.format(COL_1, COL_2, model.split("/")[2]), format='png', dpi=300)


            #plt.show()
            plt.cla()


        except:
            print("failed", model)
            failed_list.append(model)
    for i, failed in enumerate(failed_list):
        print(i, failed)
    #embed()
