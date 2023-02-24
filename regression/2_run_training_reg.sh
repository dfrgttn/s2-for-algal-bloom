#!/bin/bash
for model in "cross_deep_mha_dnn" "cross_deep_dnn" "wide_deep_dnn" "wide_deep_dnn_mha" "mha_dnn"  "grn_vsn_dnn_mha" "grn_vsn_dnn_v3" "grn_vsn_dnn_v4" "grn_vsn_dnn" "grn_vsn_dnn_v2"  "residual_mha_dnn_v2" "residual_mha_dnn" "residual_dnn" "baseline_dnn"
do
	echo "Training model $model"
	python3 ./2_run_training_reg.py -model $model

done
