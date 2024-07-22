import json

import numpy as np
import pandas as pd
from scipy import stats

input = pd.read_csv("data/input.csv", index_col=0)
label = pd.read_csv("data/output.csv", index_col=0)
# Step 0: Setup constants

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
RESULT_DIR = "result/application2"

aggr_df = pd.DataFrame()

# Load ranks, normalize and aggregate
for seed in range(50):
    rank_path = f"{RESULT_DIR}/full-rand-{seed}.json"
    with open(rank_path, "r") as f:
        rank_json = json.load(f)
    idx = rank_json["idx"]

    print(f"Seed {seed} indices are {idx}")

    mean_df = pd.DataFrame()

    # Normalize with optimal score
    for t in OUTPUT_TYPE:
        opt_score = label.loc[idx, t]
        rank_values = np.array(rank_json[f"{t}-array"])

        opt_score_values = np.array(opt_score)

        normalized_values = rank_values / opt_score_values
        mean_df[t] = [normalized_values.mean()]

    # Sort mean_df
    # Rank scores
    mean_df.index = ["score"]
    mean_df = mean_df.T
    mean_df = mean_df.sort_values("score", ascending=False)

    aggr_df[seed] = mean_df.index

aggr_df.to_csv(f"{RESULT_DIR}/app2-step5_rank.csv")

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(
    columns=["Spearman correlation", "Kendall tau", "rank1", "rank2"]
)

for i in range(50):
    for j in range(i):
        rankings1 = aggr_df[i]
        rankings2 = aggr_df[j]

        # Calculate Spearman rank-order correlation coefficient
        spearman_corr, _ = stats.spearmanr(rankings1, rankings2)

        # Calculate Kendall's tau
        kendall_tau, _ = stats.kendalltau(rankings1, rankings2)

        # Create a DataFrame with the results
        temp_df = pd.DataFrame(
            {
                "Spearman correlation": [spearman_corr],
                "Kendall tau": [kendall_tau],
                "rank1": [i],
                "rank2": [j],
            }
        )

        # Append the results to the DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

# Print the results DataFrame
print(results_df.head())

results_df.to_csv("result/app2-step5_rank-corr.csv")
