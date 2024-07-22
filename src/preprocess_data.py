import json

import pandas as pd

DATA_PATH = "./data"
INPUT_PATH = DATA_PATH + "/complexity_input"
OUTPUT_PATH = DATA_PATH + "/ground_truth_output"
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

# Load data
# input = pd.DataFrame()

# for i in INPUT_TYPE:
#     json = pd.read_json(INPUT_PATH + "/" + i + ".json")
#     input[i] = json.T["score"]
# input = input.fillna(0)
# input.to_csv("input_output/input.csv", index_label="dataset")

# Load label
label = pd.DataFrame()

for i in OUTPUT_TYPE:
    json = pd.read_json(OUTPUT_PATH + "/" + i + ".json")
    # flatten nested json
    json_flat = pd.json_normalize(json.T["params"])
    json_flat.index = json.T["score"].index
    label["score"] = json.T["score"]

    # concat json_flat and label
    label = pd.concat([label, json_flat], axis=1)

    label = label.fillna(0)
    label.to_csv(f"{DATA_PATH}/output_{i}.csv", index_label="accuracy")
