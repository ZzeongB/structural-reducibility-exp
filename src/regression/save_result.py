import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

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

result_df = pd.DataFrame()
input = pd.read_csv("data/input.csv", index_col=0)

for t in OUTPUT_TYPE:
    label = pd.read_csv(f"data/output_{t}.csv", index_col=0)

    print(f"===================={t}====================")
    X = input
    y = label

    reg = joblib.load(f"pretrained_model/multi/{t}.pkl")
    pred = reg.predict(X)
    print("R2:", sklearn.metrics.r2_score(y, pred))
    result_df[t] = [sklearn.metrics.r2_score(y, pred)]

    plt.figure(figsize=(6, 6))
    plt.scatter(pred, y, label="Test samples", c="#7570b3")
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"result/multi/{t}.png")
    plt.close()

result_df.to_csv("result/multi/result_multi.csv")
result_df.to_json("result/multi/result_multi.json")
print(result_df)
