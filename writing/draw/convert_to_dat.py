import pandas as pd

# Convert second_improve.csv to webshop-rir.dat
df_improve = pd.read_csv("second_improve.csv")
df_rir = df_improve[["Step", "Value"]].copy()
df_rir.columns = ["step", "rir"]
# Convert to percentage (multiply by 100)
df_rir["rir"] = df_rir["rir"] * 100
df_rir.to_csv("webshop-rir.dat", sep=" ", index=False)
print("Created webshop-rir.dat (values converted to percentage)")

# Convert second_reward_diff.csv to webshop-gain.dat
df_reward = pd.read_csv("second_reward_diff.csv")
df_gain = df_reward[["Step", "Value"]].copy()
df_gain.columns = ["step", "gain"]
df_gain.to_csv("webshop-gain.dat", sep=" ", index=False)
print("Created webshop-gain.dat")

# Convert alfworld-7b-gain.csv to alfworld-7b-gain.dat and alfworld-7b-rir.dat
df_alfworld_7b = pd.read_csv("alfworld-7b-gain.csv")
df_alfworld_7b_processed = df_alfworld_7b[["Step", "Value"]].copy()
df_alfworld_7b_processed.columns = ["step", "gain"]

# Save as gain.dat
df_alfworld_7b_processed.to_csv("alfworld-7b-gain.dat", sep="\t", index=False)
print("Created alfworld-7b-gain.dat")

# Save as rir.dat (same content as gain for alfworld 7b, but multiply by 100 for percentage)
df_alfworld_7b_rir = df_alfworld_7b[["Step", "Value"]].copy()
df_alfworld_7b_rir.columns = ["step", "rir"]
df_alfworld_7b_rir["rir"] = df_alfworld_7b_rir["rir"] * 100
df_alfworld_7b_rir.to_csv("alfworld-7b-rir.dat", sep="\t", index=False)
print("Created alfworld-7b-rir.dat (values converted to percentage)")

# Convert dapo-rir.csv to dapo-gain.dat and dapo-rir.dat
df_dapo = pd.read_csv("dapo-rir.csv")

# Save as gain.dat (no percentage conversion)
df_dapo_gain = df_dapo[["Step", "Value"]].copy()
df_dapo_gain.columns = ["step", "gain"]
df_dapo_gain.to_csv("dapo-gain.dat", sep="\t", index=False)
print("Created dapo-gain.dat")

# Save as rir.dat (multiply by 100 for percentage)
df_dapo_rir = df_dapo[["Step", "Value"]].copy()
df_dapo_rir.columns = ["step", "rir"]
df_dapo_rir["rir"] = df_dapo_rir["rir"] * 100
df_dapo_rir.to_csv("dapo-rir.dat", sep="\t", index=False)
print("Created dapo-rir.dat (values converted to percentage)")
