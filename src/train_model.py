import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

LEAKY_KEYWORDS = [
    'churn', 'risk', 'accountstatus', 'customertype', 'perdu', 'closed',
    'loyaltylevel', 'rfmsegment', 'spendingcat', 'satisfaction',
    'supportticket', 'customerid', 'recency', 'favoriteseason',
    'customertenuredays', 'preferredmonth'
]

LEAKY_EXACT = [
    'ChurnRisk', 'AccountStatus', 'CustomerType', 'RFMSegment',
    'Satisfaction', 'SupportTickets', 'LoyaltyLevel', 'SpendingCat',
    'CustomerID', 'Churn', 'Recency', 'FavoriteSeason',
    'CustomerTenureDays', 'PreferredMonth'
]

categorical_cols = ['SpendingCategory', 'PreferredTimeOfDay', 'WeekendPreference', 
                    'BasketSizeCategory', 'ProductDiversity', 'Country']

def load_data():
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    print(f"✅ Data loaded: {df.shape}")
    return df

def remove_leaky_features(df):
    to_drop = []
    for col in df.columns:
        if col in LEAKY_EXACT or any(kw in col.lower() for kw in LEAKY_KEYWORDS):
            to_drop.append(col)
    
    if to_drop:
        print(f"🛡️ Removing {len(to_drop)} leaky features: {to_drop}")
        df = df.drop(columns=to_drop)
    
    if 'Newsletter' in df.columns:
        df = df.drop(columns=['Newsletter'])
        print("🛡️ Removed Newsletter column")
    
    return df

def encode_features(df):
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"✅ Encoded {col}")
    
    return df, encoders

def clustering(X_train):
    print("🔬 Clustering KMeans (4 segments)...")
    numeric_data = X_train.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(numeric_data)
    
    if numeric_data.shape[1] >= 2:
        score = silhouette_score(numeric_data, clusters)
        print(f"✅ Silhouette Score: {score:.3f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    print("💾 KMeans saved → models/kmeans_model.pkl\n")
    return kmeans

def train_xgboost(X_train, X_test, y_train, y_test):
    print("🔥 Training XGBoost...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    y_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    print(f"📊 Best parameters: {grid_search.best_params_}\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))
    print(f"→ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"→ F1-Score (Churn): {f1_score(y_test, y_pred):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loyal', 'Churn'], yticklabels=['Loyal', 'Churn'])
    plt.title("Confusion Matrix - XGBoost")
    plt.ylabel("True Label")
    plt.xlabel("Prediction")
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    feat_imp = pd.Series(best_xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind='bar')
    plt.title("Top 20 Features - XGBoost")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Reports saved → reports/\n")
    
    return best_xgb

def run_training():
    df = load_data()
    
    df = remove_leaky_features(df)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    clustering(X_train)
    
    X_train, encoders = encode_features(X_train)
    
    X_test_enc = X_test.copy()
    for col in categorical_cols:
        if col in encoders:
            X_test_enc[col] = encoders[col].transform(X_test_enc[col].astype(str))
    
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])
    
    best_xgb = train_xgboost(X_train, X_test_enc, y_train, y_test)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_xgb, "models/xgboost_churn_model.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Models saved → models/")
    print("   - xgboost_churn_model.pkl")
    print("   - encoders.pkl")
    print("   - scaler.pkl")
    print("   - kmeans_model.pkl\n")
    
    print("🎉 TRAINING COMPLETE!")

if __name__ == "__main__":
    run_training()