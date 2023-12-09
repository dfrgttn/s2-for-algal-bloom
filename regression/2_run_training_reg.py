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

from utils import get_dataset_from_csv, calculate_target_class_weighs
from IPython import embed
from models import *
from utils import *

# Reproducability
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

if __name__ == '__main__':


    ap = argparse.ArgumentParser(description="""This script runs the testing
                                 of all machine learning methods of bloom
                                 prediction""")
    ap.add_argument("-model", "--model", required=False,
            default = "baseline_dnn", help="selected  model")
    ap.add_argument("-train", "--train_data", required=False,
            default = "./train_reg_data.csv", help="path to train data")
    ap.add_argument("-test", "--test_data", required=False,
            default = "./test_reg_data.csv", help="path to test data")
    ap.add_argument("-epochs", "--epochs", required=False,
            default = 200, help="number of epochs to train")
    ap.add_argument("-batch_size", "--batch_size", required=False,
            default = 16, help="number of batch size")
    ap.add_argument("-fc_size", "--fc_size", required=False,
            default = 32, help="number of fc size")
    ap.add_argument("-name", "--name", required=False,
            default = "", help="(optional) name of experiment")
    ap.add_argument("-path", "--models_path", required=False,
            default = "./models", help="path to models")

    args = vars(ap.parse_args())

    ARG_MODELS_PATH = args["models_path"]
    ARG_TRAIN_DATA_PATH = args["train_data"]
    ARG_TEST_DATA_PATH = args["test_data"]
    ARG_MODEL_NAME_FULL = args["model"]
    ARG_EPOCHS = int(args["epochs"])
    ARG_BATCH_SIZE = int(args["batch_size"])
    ARG_FC_SIZE = int(args["fc_size"])
    ARG_NAME = args["name"]

    print("---------------------------------------------------")
    print("The experiment configs:")
    print("The model path is set to {}".format(ARG_MODELS_PATH))
    print("The train data path is set to {}".format(ARG_TRAIN_DATA_PATH))
    print("The test data path is set to {}".format(ARG_TEST_DATA_PATH))
    print("---------------------------------------------------")

    train_data  = pd.read_csv(ARG_TRAIN_DATA_PATH)
    test_data = pd.read_csv(ARG_TEST_DATA_PATH)

    INPUTS_COLS = train_data.columns[:-1]
    TARGET_NAME = train_data.columns[-1]

    print("Train and test data shapes", train_data.shape, test_data.shape)


    model_map = {
        "baseline_dnn" : model_1_baseline(INPUTS_COLS),
        "residual_dnn" : model_2_residual(INPUTS_COLS),
        "residual_mha_dnn" : model_3_mha_residual(INPUTS_COLS),
        "residual_mha_dnn_v2" : model_3_mha_residual_v2(INPUTS_COLS),
        "grn_vsn_dnn" : model_4_grn_vsn(INPUTS_COLS, 16),
        "grn_vsn_dnn_v2" : model_4_grn_vsn(INPUTS_COLS, 8, fc_layer = True),
        "grn_vsn_dnn_v3" : model_4_grn_vsn(INPUTS_COLS, 32),
        "grn_vsn_dnn_v4" : model_4_grn_vsn(INPUTS_COLS,  4, fc_layer = True),
        "grn_vsn_dnn_mha" : model_4_grn_vsn(INPUTS_COLS, 8, mha_layer = True),
        "mha_dnn" : model_5_mha(INPUTS_COLS, 32),
        "wide_deep_dnn" : model_6_wide_deep(INPUTS_COLS),
        "wide_deep_dnn_mha" : model_6_wide_deep(INPUTS_COLS, mha_layer = True),
        "cross_deep_dnn" : model_7_deep_cross(INPUTS_COLS),
        "cross_deep_mha_dnn" : model_7_deep_cross(INPUTS_COLS, mha_layer = True),
        "cnn_dnn" : model_8_cnn(INPUTS_COLS),
    }
    model = model_map.get(ARG_MODEL_NAME_FULL)

    #embed()
    train_dataset = get_dataset_from_csv(
        ARG_TRAIN_DATA_PATH,
        train_data.columns,
        TARGET_NAME,
        shuffle=True, batch_size=ARG_BATCH_SIZE
    )
    test_dataset = get_dataset_from_csv(
        ARG_TEST_DATA_PATH,
        train_data.columns,
        TARGET_NAME,
        shuffle=True, batch_size=ARG_BATCH_SIZE
    )

    history = run_experiment(model,
                             train_dataset,
                             test_dataset,
                             num_epochs = ARG_EPOCHS)

    print("Evaluating model performance...")

    loss = model.evaluate(test_dataset)
    print(f"Test loss: {round(loss, 2)}")

    model_name = "{}/{}-{}-{}".format(ARG_MODELS_PATH,
                                   ARG_MODEL_NAME_FULL + ARG_NAME,
                                   ARG_FC_SIZE,
                                   round(loss, 2))
    model.save(model_name)

    tf.keras.utils.plot_model(model,
                              to_file=model_name + "/model.png",
                              show_shapes=True)


    if True:

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.savefig(model_name + "/training.png")
