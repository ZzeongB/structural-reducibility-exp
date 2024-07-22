import argparse
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import reader as rd
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.manifold import TSNE
from zadu import zadu
import autosklearn.regression

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

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

np.random.seed(RAND_SEED)  # Set random seed
idx = np.random.choice(input.index, 86, replace=False)  # Randomly select 86 samples
print(idx)
logging.info(idx)

# Step 1: Select 86 samples out of whole dataset and Train the model
X = input.loc[idx, :]
y = label.loc[idx, t]

reg = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=600,
    per_run_time_limit=30,
    memory_limit=10000,
    resampling_strategy="cv",
    resampling_strategy_arguments={"folds": 5},
)
reg.fit(X, y)

joblib.dump(reg, f"{MODEL_DIR}/rand-{RAND_SEED}.pkl")
print(reg.leaderboard())

pred = reg.predict(X)
print(f"{t} - R2:", sklearn.metrics.r2_score(y, pred))

reg = joblib.load(f"{MODEL_DIR}/rand-{RAND_SEED}.pkl")

# Step 2: Predict the Optimal Score with 10 samples
idx_not_trained = input.index.difference(idx)
print(f"Predicting {t} with {len(idx_not_trained)} samples")
logging.info(f"Predicting {t} with {len(idx_not_trained)} samples")

X_not_trained = input.loc[idx_not_trained, :]
y_not_trained = label.loc[idx_not_trained, t]

pred = reg.predict(X_not_trained)
print(f"{t} - R2:", sklearn.metrics.r2_score(y_not_trained, pred))
# logging.info(f"{t} - R2:", sklearn.metrics.r2_score(y_not_trained, pred))
print(f"{t} - prediction:", pred)
# logging.info(f"{t} - prediction:", pred)
print(f"{t} - actual:", y_not_trained)
# logging.info(f"{t} - actual:", y_not_trained)

# Step 3-A: Bayesian Optimization

# Load the dataset
for data_idx in range(10):
    data_name = idx_not_trained[data_idx]
    # If data_name file exists, pass
    if os.path.exists(f"{RESULT_DIR}/{RAND_SEED}/{data_name}_A.json"):
        print(f"{data_name} already exists")
        logging.info(f"{data_name} already exists")
        continue

    data, label_ = rd.read_dataset(data_name, "labeled-datasets")
    print(f"============RAND-{RAND_SEED} {data_name}============")
    logging.info(f"============RAND-{RAND_SEED} {data_name}============")

    # Define the function to optimize
    def optimize_tsne(perplexity):
        # Create the t-SNE model
        model = TSNE(perplexity=perplexity)

        # Fit and transform the data
        X_transformed = model.fit_transform(data)
        spec = [
            {
                "id": "tnc",
                "params": {"k": 25},
            }
        ]
        # Calculate the score
        score_module = zadu.ZADU(spec, data, return_local=True)
        score, local_list = score_module.measure(X_transformed)
        tr = score[0]["trustworthiness"]
        cn = score[0]["continuity"]
        ret = 2 * tr * cn / (tr + cn)

        return ret

    pbounds = {"perplexity": (2, 500)}

    # Create the optimizer
    optimizer = BayesianOptimization(
        f=optimize_tsne,
        pbounds=pbounds,
        random_state=1,
    )
    print("Initialized")
    logging.info("Initialized")

    start_time = time.time()
    # Optimize
    optimizer.maximize(
        init_points=10,
        n_iter=40,
    )
    exec_time = time.time() - start_time

    # Print the best result
    print("Best Result", optimizer.max)
    logging.info("Best Result", optimizer.max)

    # Step 4: Save the result
    scores = {}
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        logging.info("Iteration {}: \n\t{}".format(i, res))

        scores[i] = res
    scores["total_time"] = exec_time

    if not os.path.exists(f"{RESULT_DIR}/{RAND_SEED}"):
        os.makedirs(f"{RESULT_DIR}/{RAND_SEED}")

    with open(f"{RESULT_DIR}/{RAND_SEED}/{data_name}_A.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Save to " + f"{RESULT_DIR}/{RAND_SEED}/{data_name}_A.json\n")
    logging.info("Save to " + f"{RESULT_DIR}/{RAND_SEED}/{data_name}_A.json\n")

    # Step 3-B: Do the same thing with 3-A, but stop iteration when it achieves the optimal score
    optimal_score = pred[data_idx]

    # Create the optimizer
    optimizer_dr = BayesianOptimization(
        f=optimize_tsne,
        pbounds=pbounds,
        random_state=1,
    )

    start_time = time.time()

    # Manual optimization loop with stopping criterion
    for i in range(50):  # Total of 50 iterations (10 initial points + 40 iterations)
        optimizer_dr.maximize(
            init_points=1 if i < 10 else 0,  # 10 initial points
            n_iter=1,  # 1 iteration at a time
        )
        if optimizer_dr.max["target"] >= optimal_score:
            break

    exec_time = time.time() - start_time

    # Print the best result
    print(optimizer_dr.max)
    logging.info(optimizer_dr.max)

    # Step 4: Save the result
    scores = {}
    for i, res in enumerate(optimizer_dr.res):
        print("Iteration {}: \n\t{}".format(i, res))
        logging.info("Iteration {}: \n\t{}".format(i, res))

        scores[i] = res
    scores["total_time"] = exec_time

    with open(f"{RESULT_DIR}/{RAND_SEED}/{data_name}_B.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Save to " + f"{RESULT_DIR}/{RAND_SEED}/{data_name}_B.json\n")
    logging.info("Save to " + f"{RESULT_DIR}/{RAND_SEED}/{data_name}_B.json\n")
