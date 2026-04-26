#df est le dataset de training (80%)
#df_test est le dataset de test




import pandas as pd
import os
import ipaddress
import geoip2.database
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


df_raw = pd.read_csv('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
print(df_raw.shape)

df, df_test = train_test_split(
    df_raw, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_raw['Churn']  # Preserve Churn class distribution
)

if 'Newsletter' in df.columns:
    df = df.drop(columns=['Newsletter'])
    print("Removed 'Newsletter' column (constant value - useless feature)")

median_age = df['Age'].median()
#df['Age'].fillna(median_age, inplace=True)
df['Age'] = df['Age'].fillna(median_age)

print("Missing values in Age after imputation:", df['Age'].isnull().sum())

# Step 2: Fill missing AvgDaysBetweenPurchases with median
median_avg_days = df['AvgDaysBetweenPurchases'].median()
df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(median_avg_days)

print("Missing values in AvgDaysBetweenPurchases after imputation:",
      df['AvgDaysBetweenPurchases'].isnull().sum())


print("Columns with missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

#aberrant values

# Define numeric columns and their valid ran
valid_ranges = {
    'CustomerID': (10000, 99999),
    'Recency': (0, 400),
    'Frequency': (1, 50),
    'MonetaryTotal': (-5000, 15000),
    'MonetaryAvg': (5, 500),
    'MonetaryStd': (0, 500),
    'MonetaryMin': (-5000, 5000),
    'MonetaryMax': (0, 10000),
    'TotalQuantity': (-10000, 100000),
    'AvgQuantityPerTransaction': (1, 1000),
    'MinQuantity': (-8000, 0),
    'MaxQuantity': (1, 8000),
    'CustomerTenureDays': (0, 730),
    'FirstPurchaseDaysAgo': (0, 730),
    'PreferredDayOfWeek': (0, 6),
    'PreferredHour': (0, 23),
    'PreferredMonth': (1, 12),
    'WeekendPurchaseRatio': (0.0, 1.0),
    'AvgDaysBetweenPurchases': (0.0, 365.0),
    'UniqueProducts': (1, 1000),
    'UniqueDescriptions': (1, 1000),
    'AvgProductsPerTransaction': (1, 100),
    'UniqueCountries': (1, 5),
    'NegativeQuantityCount': (0, 100),
    'ZeroPriceCount': (0, 50),
    'CancelledTransactions': (0, 50),
    'ReturnRatio': (0.0, 1.0),
    'TotalTransactions': (1, 10000),
    'UniqueInvoices': (1, 500),
    'AvgLinesPerInvoice': (1, 100),
    'Age': (18, 81),
}


special_values = {
    'SupportTicketsCount': [-1, *range(0, 16), 999],
    'SatisfactionScore': [-1, 0, 1, 2, 3, 4, 5, 99]
}

# Summary dictionary for aberrant counts
aberrant_summary = {}

# Check numeric ranges
for col, (low, high) in valid_ranges.items():
    count = df[(df[col] < low) | (df[col] > high)].shape[0]
    if count > 0:
        aberrant_summary[col] = count

# Check special value columns
for col, allowed in special_values.items():
    count = df[~df[col].isin(allowed)].shape[0]
    if count > 0:
        aberrant_summary[col] = count

# Print a concise summary
if aberrant_summary:
    print("Columns with aberrant values and number of rows affected:")
    for col, count in aberrant_summary.items():
        print(f"{col}: {count} rows")
else:
    print("No aberrant values found in numeric or special-code columns.")

import numpy as np

# Replace numeric aberrant values with NaN
for col, (low, high) in valid_ranges.items():
    df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

# Replace special-code aberrant values with NaN
for col, allowed in special_values.items():
    df.loc[~df[col].isin(allowed), col] = np.nan

# Verify
print("After cleaning, columns with missing values (including former aberrant values):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Example: Fill numeric NaNs with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    # Check that there are no missing values left
print("Columns with missing values after filling all NaNs:")
print(df.isnull().sum()[df.isnull().sum() > 0])



# Ordinal encoding for features with natural order
ordinal_features = {
    'RFMSegment': ['Dormants', 'Potentiels', 'Fidèles', 'Champions'],
    'AgeCategory': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
    'SpendingCat': ['Low', 'Medium', 'High', 'VIP'],
    'LoyaltyLevel': ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
    'ChurnRisk': ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSize': ['Petit', 'Moyen', 'Grand', 'Inconnu']
}

for col, categories in ordinal_features.items():
    if col in df.columns:
        # Create mapping
        mapping = {cat: i for i, cat in enumerate(categories)}
        df[f'{col}_encoded'] = df[col].map(mapping)
        # Drop original column
        df = df.drop(columns=[col])
        print(f"Encoded {col} (ordinal) → {col}_encoded")

nominal_features = ['CustomerType', 'FavoriteSeason', 'PreferredTime', 'Region', 
                    'WeekendPref', 'ProdDiversity', 'Gender', 'AccountStatus']

for col in nominal_features:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        print(f"Encoded {col} (one-hot) → {dummies.shape[1]} new columns")


df['RegistrationDate'] = pd.to_datetime(
    df['RegistrationDate'],
    dayfirst=True,   
    errors='coerce' 
)

df['RegYear'] = df['RegistrationDate'].dt.year
df['RegMonth'] = df['RegistrationDate'].dt.month
df['RegDay'] = df['RegistrationDate'].dt.day
df['RegWeekday'] = df['RegistrationDate'].dt.weekday


# --- FEATURE ENGINEERING ---
df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)


def is_private_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:  # malformed IP
        return None

df['IsPrivateIP'] = df['LastLoginIP'].apply(is_private_ip)


reader = geoip2.database.Reader(r'C:\ProjetML\MachineLearningProject\data\GeoLite2-Country.mmdb')

def get_country(ip):
    try:
        response = reader.country(ip)
        return response.country.iso_code  
    except:
        return None

df['IPCountry'] = df['LastLoginIP'].apply(get_country)

reader.close()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr_matrix = numeric_df.corr()

strong_corr = corr_matrix[(corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1)]
print(strong_corr)


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

print(df[['LastLoginIP', 'IsPrivateIP', 'IPCountry']].head())

df = df.drop(columns=['MinQuantity'])


# Threshold for high correlation
threshold = 0.8


mask = (abs(corr_matrix) > threshold) & (abs(corr_matrix) < 1)


correlated_pairs = []
for col in mask.columns:
    for row in mask.index:
        if mask.loc[row, col]:
            # Avoid duplicate pairs
            if (col, row) not in correlated_pairs and (row, col) not in correlated_pairs:
                correlated_pairs.append((row, col, corr_matrix.loc[row, col]))

# Convert to DataFrame for better visualization
correlated_df = pd.DataFrame(correlated_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
correlated_df = correlated_df.sort_values(by='Correlation', ascending=False)
print(correlated_df)

target_col = 'Churn'  # Your target variable

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
if 'CustomerID' in numeric_cols:
    numeric_cols.remove('CustomerID')  

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"\n✅ Applied StandardScaler to {len(numeric_cols)} numeric features")
print(f"   - Mean is now 0, Standard deviation is now 1 for these features")


pca_numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if target_col in pca_numeric_cols:
    pca_numeric_cols.remove(target_col)
if 'CustomerID' in pca_numeric_cols:
    pca_numeric_cols.remove('CustomerID')

# Fill all NaN before PCA
df[pca_numeric_cols] = df[pca_numeric_cols].fillna(df[pca_numeric_cols].median())

# Determine number of components (keep 95% of variance or max 10 components)
n_components = min(10, len(pca_numeric_cols))

# Apply PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(df[pca_numeric_cols])

# Create new PCA column names
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(pca_result, columns=pca_columns)

# Drop original numeric columns and add PCA components
df = df.drop(columns=pca_numeric_cols)
df = pd.concat([df, pca_df], axis=1)

print(f"\n✅ Applied PCA: {len(pca_numeric_cols)} features → {n_components} components")
print(f"   - Explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
print(f"   - First 3 components explain: {pca.explained_variance_ratio_[:3].sum():.2%}")


# Check current class distribution
if target_col in df.columns:
    print(f"\n📊 Class distribution BEFORE balancing:")
    print(df[target_col].value_counts())
    print(f"Ratio: {df[target_col].value_counts(normalize=True)[1]:.2%} churn")

    # Separate majority and minority classes
    majority = df[df[target_col] == 0]
    minority = df[df[target_col] == 1]
    
    minority_upsampled = resample(minority,
                                   replace=True,      # Sample with replacement
                                   n_samples=len(majority),  # Match majority count
                                   random_state=42)
    
    # Combine back
    df_balanced = pd.concat([majority, minority_upsampled])
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Class distribution AFTER balancing:")
    print(df_balanced[target_col].value_counts())
    print(f"Ratio: 50% churn, 50% non-churn")
    
    df = df_balanced  # Use balanced dataset

# 4. Save the final processed data
df.to_csv("data/processed/cleaned_data.csv", index=False)
print(f"\n🎉 Final dataset shape: {df.shape}")
print("Saved to data/processed/cleaned_data.csv")


df.to_csv("data/train_test/train_data.csv", index=False)

df_test.to_csv("data/train_test/test_data.csv", index=False)
