import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--rand_seed", type=int, default=0, help="an integer for the RAND_SEED"
)

args = parser.parse_args()

# Create a logger
logging.basicConfig(
    filename=f"application1-rand{args.rand_seed}.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
# Use the command-line arguments in your code
RAND_SEED = args.rand_seed
print(f"random seed {RAND_SEED}")

# Step 0: Setup constants
INPUT_TYPE = ["dc_5", "nc_3", "nc_5", "nc_10", "nc_25", "nc_30", "nc_50", "nc_75"]
OUTPUT_TYPE = [
    "umato_srho_0",
    "pca_tnc_25",
    "tsne_pr_0",
    "umato_tnc_25",
    "isomap_tnc_25",
    "lle_pr_0",
    "isomap_pr_0",
    "tsne_tnc_25",
    "umap_pr_0",
    "umap_tnc_25",
    "pca_pr_0",
    "lle_tnc_25",
    "umato_pr_0",
]
SCORE_TYPE = [
    "explained_variance_score",
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_squared_log_error",
    "root_mean_squared_log_error",
    "median_absolute_error",
    "r2_score",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_absolute_percentage_error",
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
]
input = pd.read_csv("data/input.csv", index_col=0)
label = pd.read_csv("data/output.csv", index_col=0)

RESULT_DIR = f"result/application/{RAND_SEED}"

t = "tsne_tnc_25"
assert t in OUTPUT_TYPE
MODEL_DIR = "pretrained_model/application/"

np.random.seed(RAND_SEED)  # Set random seed
idx = np.random.choice(input.index, 86, replace=False)  # Randomly select 86 samples
print(idx)
logging.info(idx)

# Step 1: Select 86 samples out of whole dataset and Train the model
X = input.loc[idx, :]
y = label.loc[idx, t]


for rand in range(5):
    RAND_SEED = rand
    reg = joblib.load(f"{MODEL_DIR}/rand-{RAND_SEED}.pkl")

    # Step 2: Predict the Optimal Score with 10 samples
    idx_not_trained = input.index.difference(idx)
    print(f"Predicting {t} with {len(idx_not_trained)} samples")
    logging.info(f"Predicting {t} with {len(idx_not_trained)} samples")

    X_not_trained = input.loc[idx_not_trained, :]
    y_not_trained = label.loc[idx_not_trained, t]

    pred = reg.predict(X_not_trained)
    r2 = sklearn.metrics.r2_score(y_not_trained, pred)
    print(f"{t} - R2:", r2)
    print(f"{t} - prediction:", pred.tolist())

    opt_scores = {}
    opt_scores["R2"] = r2
    indiv_scores = {}
    for i, idx in enumerate(idx_not_trained):
        indiv_scores[idx] = pred[i]

    opt_scores["individual_scores"] = indiv_scores

    with open(f"{RESULT_DIR}/{RAND_SEED}/opt_scores.json", "w") as f:
        json.dump(opt_scores, f, indent=4)
