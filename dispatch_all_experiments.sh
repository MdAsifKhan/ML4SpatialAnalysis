#!/bin/bash

# Define the parameter values to try
model_names=("logistic" "gnn" "randomforest" "xgboost")
normalize_features=("True" "False")
fnorm_values=("minmax" "log1p" "raw" "znorm")
lr_values=("0.01" "0.001" "0.1")
gnn_types=("gcn" "ssgcn")
ssgcn_alphas=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
ssgcn_ks=("1" "2" "4" "6" "8")

# Path to the config file
config_file="confs/config.yaml"

# Path to the script that updates the config file and runs the experiment
response_prediction="response_prediction.sh"

# Function to call the response_prediction.sh script with the given parameters
run_experiment() {
    model_name="$1"
    normalize_features="$2"
    fnorm="$3"
    lr="$4"
    gnn_type="$5"
    ssgcn_alpha="$6"
    ssgcn_k="$7"

    echo "Running experiment with parameters:"
    echo "Model: $model_name"
    echo "Normalize Features: $normalize_features"
    echo "Feature Normalization: $fnorm"
    echo "Learning Rate: $lr"
    echo "GNN Type: $gnn_type"
    echo "SSGCN Alpha: $ssgcn_alpha"
    echo "SSGCN K: $ssgcn_k"

    # Call the response_prediction.sh script with the updated parameters
    "$response_prediction" "$config_file" \
        "model.name=$model_name" \
        "model.normalise_features=$normalize_features" \
        "model.fnorm=$fnorm" \
        "model.gnn.gconv=$gnn_type" \
        "model.gnn.lr=$lr" \
        "model.gnn.ssgcn.K=$ssgcn_k" \
        "model.gnn.ssgcn.alpha=$ssgcn_alpha"
}

# Run experiments for all parameter combinations
for model_name in "${model_names[@]}"; do
    for normalize_features in "${normalize_features[@]}"; do
        for fnorm in "${fnorm_values[@]}"; do
            for lr in "${lr_values[@]}"; do
                if [ "$model_name" == "gnn" ]; then
                    for gnn_type in "${gnn_types[@]}"; do
                        if [ "$gnn_type" == "ssgcn" ]; then
                            for ssgcn_alpha in "${ssgcn_alphas[@]}"; do
                                for ssgcn_k in "${ssgcn_ks[@]}"; do
                                    run_experiment "$model_name" "$normalize_features" "$fnorm" "$lr" "$gnn_type" "$ssgcn_alpha" "$ssgcn_k"
                                done
                            done
                        else
                            run_experiment "$model_name" "$normalize_features" "$fnorm" "$lr" "$gnn_type" "" ""
                        fi
                    done
                else
                    run_experiment "$model_name" "$normalize_features" "$fnorm" "$lr" "" "" ""
                fi
            done
        done
    done
done