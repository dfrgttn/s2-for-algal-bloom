import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Conv1D
from tensorflow.keras.models import Model
from keras_multi_head import MultiHead

## Data pre-processing was done using Keras Examples
## https://keras.io/examples/structured_data/classification_with_grn_and_vsn/
def create_model_inputs(input_feature_names):
    inputs = {}
    for feature_name in input_feature_names:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )
    return inputs


def encode_inputs(inputs, use_embedding=False):
    encoded_features = []
    for feature_name in inputs:
        # Use the numerical features as-is.
        encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        encoded_features.append(encoded_feature)

    return encoded_features


def model_1_baseline(input_feature_names, 
                     hidden_units = [128, 32],
                     dropout_rate_0 = 0.2,
                     dropout_rate = 0.2):

    inputs = create_model_inputs(input_feature_names)
    features = encode_inputs(inputs)
    features = layers.concatenate(features)
    features = layers.Dropout(dropout_rate_0)(features)

    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


## The residual modules used from
## https://gist.github.com/lazuxd/d7aaba284123bf3340e723701e381e6e
def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x, filters, dropout_rate = 0.2):
    y = Dense(filters)(x)
    y = layers.Dropout(dropout_rate)(y)
    y = relu_bn(y)
    y = Dense(filters)(y)

    out = Add()([x, y])
    out = relu_bn(out)
    return out 

def model_2_residual(input_feature_names,
                     hidden_units = 64):

    inputs = create_model_inputs(input_feature_names)
    features = encode_inputs(inputs)
    features = layers.concatenate(features)


    t = BatchNormalization()(features)
    t = Dense(hidden_units)(t)
    t = relu_bn(t)

    num_blocks_list = [1, 2, 1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, filters=hidden_units)

    t = Flatten()(t)
    outputs = Dense(1, activation='sigmoid')(t)

    model = Model(inputs, outputs)
    return model 

def residual_meha_block(x, filters, dropout_rate = 0.2):
    y = MultiHead(keras.layers.Dense(units=filters), layer_num=4)(x)
    y = keras.layers.Flatten()(y)
    y = layers.Dropout(dropout_rate)(y)
    y = relu_bn(y)
    y = MultiHead(keras.layers.Dense(units=filters), layer_num=4)(y)
    y = keras.layers.Flatten()(y)

    out = Add()([x, y])
    out = relu_bn(out)
    return out

def model_3_mha_residual(input_feature_names,
                         hidden_units = 32):

    inputs = create_model_inputs(input_feature_names)
    features = encode_inputs(inputs)
    features = layers.concatenate(features)

    t = BatchNormalization()(features)
    t = MultiHead(keras.layers.Dense(units=hidden_units, 
                                     kernel_regularizer=keras.regularizers.l2(0.001)), 
                  layer_num=4)(t)
    t = keras.layers.Flatten()(t)
    t = relu_bn(t)

    num_blocks_list = [2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, filters=hidden_units*4)

    t = Flatten()(t)
    outputs = Dense(1, activation='sigmoid')(t)

    model = Model(inputs, outputs)

    return model 

def model_3_mha_residual_v2(input_feature_names,
                         hidden_units = 32):

    inputs = create_model_inputs(input_feature_names)
    features = encode_inputs(inputs)
    features = layers.concatenate(features)

    t = BatchNormalization()(features)

    t = MultiHead(keras.layers.Dense(units=hidden_units), layer_num=4)(t)
    t = keras.layers.Flatten()(t)
    t = relu_bn(t)

    num_blocks_list = [1,2,1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, filters=hidden_units * 4)


    t = Flatten()(t)
    t = MultiHead(keras.layers.Dense(units=hidden_units // 4), layer_num=2)(t)
    t = keras.layers.Flatten()(t)

    outputs = Dense(1, activation='sigmoid')(t)

    model = Model(inputs, outputs)

    return model 



class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)



class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate = 0.2):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x



class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate = 0.2):
        super(VariableSelection, self).__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


def model_4_grn_vsn(input_feature_names, 
                 encoding_size, fc_layer = False, fc_size = 32,
                    mha_layer = False,
                    dropout_rate = 0.2):
    inputs = create_model_inputs(input_feature_names)
    feature_list = encode_inputs(inputs, encoding_size)
    num_features = len(feature_list)
    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )
    if fc_layer:
        bn = layers.BatchNormalization()(features)
        features = layers.Dense(units=fc_size)(bn)

    if mha_layer:
        features = MultiHead(keras.layers.Dense(units=fc_size), layer_num=8)(features)
        features = keras.layers.Flatten(name='Flatten')(features)

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def model_5_mha(input_feature_names, 
                 encoding_size, dropout_rate = 0.2):
    inputs = create_model_inputs(input_feature_names)
    feature_list = encode_inputs(inputs)
    features = layers.concatenate(feature_list)

    features = MultiHead(keras.layers.Dense(units=64), layer_num=8)(features)
    features = keras.layers.Flatten()(features)

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



def model_6_wide_deep(input_feature_names,
                      hidden_units = [128, 16],
                      mha_layer = False,
                      dropout_rate = 0.2,
                      dropout_rate_0 = 0.2):

    inputs = create_model_inputs(input_feature_names)
    wide = encode_inputs(inputs)
    wide = layers.concatenate(wide)
    wide = layers.BatchNormalization()(wide)

    if mha_layer:
        wide = MultiHead(keras.layers.Dense(units=32), 
                         layer_num=8)(wide)
        wide = keras.layers.Flatten()(wide)
    
    deep = encode_inputs(inputs)
    deep = layers.concatenate(deep)
    deep = layers.Dropout(dropout_rate_0)(deep)
    
    if mha_layer:
        deep = MultiHead(keras.layers.Dense(units=16), layer_num=8)(deep)
        deep = keras.layers.Flatten()(deep)
    
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([wide, deep])


    if mha_layer:
        merged = MultiHead(keras.layers.Dense(units=16), layer_num=8)(merged)
        merged = keras.layers.Flatten()(merged)

    outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def model_7_deep_cross(input_feature_names, 
                      hidden_units = [128, 16], mha_layer = False,
                       dropout_rate = 0.2, dropout_rate_0 = 0.2):

    inputs = create_model_inputs(input_feature_names)
    x0 = encode_inputs(inputs)
    x0 = layers.concatenate(x0)

    if mha_layer:
        x0 = MultiHead(keras.layers.Dense(units=32), layer_num=8)(x0)
        x0 = keras.layers.Flatten()(x0)

    cross = x0
    for _ in hidden_units:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)

    deep = x0
    deep = layers.Dropout(dropout_rate_0)(deep)
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])


    if mha_layer:
        merged = MultiHead(keras.layers.Dense(units=16), layer_num=8)(merged)
        merged = keras.layers.Flatten()(merged)

    outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
