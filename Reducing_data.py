import pandas as pd

# Load full dataset
df = pd.read_csv("/Users/aditi/Desktop/265 project/Raw_data/postings_half.csv")

# Take a random 50% sample
df_half = df.sample(frac=0.5, random_state=42).reset_index(drop=True)

# Save to new file
df_half.to_csv("postings_0.25.csv", index=False)

print("✅ Saved 50% of the dataset as: postings_0.25.csv")
