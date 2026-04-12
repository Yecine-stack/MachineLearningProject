#splitting into test/train
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/cleaned_data.csv")
'''
# Split into 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits
train_df.to_csv("data/train_test/train.csv", index=False)
test_df.to_csv("data/train_test/test.csv", index=False)

print("Train and test sets saved in data/train_test/")
'''
import pandas as pd

train_df = pd.read_csv("data/train_test/train.csv")
test_df = pd.read_csv("data/train_test/test.csv")

print(f"Training set: {len(train_df)} rows")
print(f"Testing set: {len(test_df)} rows")

print("\nMissing values in train:")
print(train_df.isnull().sum())

print("\nMissing values in test:")
print(test_df.isnull().sum())


print("Distribution of MonetaryTotal in train:")
print(train_df['MonetaryTotal'].describe())

print("\nDistribution of MonetaryTotal in test:")
print(test_df['MonetaryTotal'].describe())