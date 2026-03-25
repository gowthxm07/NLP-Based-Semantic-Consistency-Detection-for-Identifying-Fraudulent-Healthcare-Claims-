import pandas as pd
import matplotlib.pyplot as plt

print("Loading candidate pairs...")

df = pd.read_csv("data/all_candidate_pairs_top5.csv")

print("Total candidate pairs:", len(df))

scores = df["final_score"]

print("Score statistics:")
print(scores.describe())

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    scores,
    bins=100,
    color="steelblue",
    edgecolor="black"
)

plt.title("Distribution of Claim–Note Similarity Scores")
plt.xlabel("Final Similarity Score")
plt.ylabel("Number of Pairs")

plt.axvline(scores.mean(), color="red", linestyle="--", label="Mean")

plt.legend()

plt.tight_layout()

plt.savefig("data/similarity_score_distribution.png")

plt.show()

print("Plot saved to data/similarity1_score_distribution.png")