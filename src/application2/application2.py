import json
import logging
import os
import time

import joblib
import multiprocess
import numpy as np
import pandas as pd

# Create a logger
logging.basicConfig(
    filename="application2.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

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

RESULT_DIR = "result/application2/"

MODEL_DIR = "pretrained_model/"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# Step 2: Select 10 samples from the input data and calculate Mean of Optimal TNC Score
def calculate_score(seed):
    seed = seed + 40
    np.random.seed(seed)
    idx = np.random.choice(input.index, 10, replace=False)  # Randomly select 86 samples
    print(f"Seed {seed} - idx: ", idx.tolist())
    logging.info(f"Seed {seed} - idx: ", idx.tolist())

    result_dict = {}
    # score_dict = {}
    result_dict["idx"] = idx.tolist()

    for t in OUTPUT_TYPE:  # For each Model
        start = time.time()
        score = 0

        model = joblib.load(f"{MODEL_DIR}/{t}.pkl")
        y_pred = model.predict(input.loc[idx])

        result_dict[f"{t}-array"] = y_pred.tolist()
        score += y_pred.mean()

        # score_dict[t] = score  # store the score in a dictionary
        print(f"Seed {seed} - Score of model {t}: {score}, time: {time.time()-start}")
        logging.info(
            f"Seed {seed} - Score of model {t}: {score}, time: {time.time()-start}"
        )
    # # Rank scores
    # score_df = pd.DataFrame(score_dict, index=["score"]).T
    # score_df = score_df.sort_values("score", ascending=False)

    # # Save df as JSON
    # score_df.to_json(f"{RESULT_DIR}/rand-{seed}.json")
    # Dump the result_dict to a JSON file
    with open(f"{RESULT_DIR}/full-rand-{seed}.json", "w") as f:
        json.dump(result_dict, f)


with multiprocess.Pool(5) as pool:
    pool.map(calculate_score, range(10))
