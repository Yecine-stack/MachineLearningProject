import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def load_prepared_data():
    X_train = pd.read_csv("data/train_test/train.csv")
    X_test = pd.read_csv("data/train_test/test.csv")
    y_train = pd.read_csv("data/train_test/train.csv").squeeze()
    y_test = pd.read_csv("data/train_test/test.csv").squeeze()
    print(f"✅ Data loaded: {X_train.shape[1]} features | {X_train.shape[0]} train | {X_test.shape[0]} test\n")
    return X_train, X_test, y_train, y_test

def remove_leaky_features(X_train, X_test, leaky_keywords, leaky_exact):
    to_drop = set()
    for col in X_train.columns:
        if col in leaky_exact or any(kw in col.lower() for kw in leaky_keywords):
            to_drop.add(col)
    if to_drop:
        print(f"🛡️ Removed {len(to_drop)} leaky features: {sorted(to_drop)}")
        X_train = X_train.drop(columns=[c for c in to_drop if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    else:
        print("✅ No leaky features detected.")
    print(f"   → Remaining features: {X_train.shape[1]}\n")
    return X_train, X_test

def build_encoder(X_train, ordinal_candidates=None):
    if ordinal_candidates is None:
        ordinal_candidates = ['AgeCategory', 'BasketSizeCategory', 'PreferredTimeOfDay',
                              'ProductDiversity', 'WeekendPreference']
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    ordinal_cols = [c for c in cat_cols if c in ordinal_candidates]
    onehot_cols = [c for c in cat_cols if c not in ordinal_cols]
    
    transformers = []
    if onehot_cols:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols))
    if ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols))
    
    return ColumnTransformer(transformers=transformers, remainder='passthrough')

def encode_features(X_train, X_test, leaky_keywords=None):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        print("✅ No categorical features to encode.")
        return X_train, X_test, None
    
    print(f"🔄 Encoding {len(cat_cols)} categorical variables: {cat_cols}")
    
    encoder = build_encoder(X_train)
    encoder.fit(X_train)
    
    X_train_enc = pd.DataFrame(encoder.transform(X_train), columns=encoder.get_feature_names_out(), index=X_train.index)
    X_test_enc = pd.DataFrame(encoder.transform(X_test), columns=encoder.get_feature_names_out(), index=X_test.index)
    
    if leaky_keywords:
        to_drop = [c for c in X_train_enc.columns if any(kw in c.lower() for kw in leaky_keywords)]
        if to_drop:
            print(f"🛡️ Removed encoded leaky columns: {to_drop}")
            X_train_enc = X_train_enc.drop(columns=to_drop)
            X_test_enc = X_test_enc.drop(columns=[c for c in to_drop if c in X_test_enc.columns])
    
    print(f"✅ Encoding complete → {X_train_enc.shape[1]} columns\n")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/encoder.pkl")
    print("💾 Encoder saved → models/encoder.pkl\n")
    
    return X_train_enc, X_test_enc, encoder

def diagnostic_correlation(X_train, y_train, top_n=15):
    print(f"🔍 DIAGNOSTIC – Top {top_n} correlations with target:")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    corr = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)
    print(corr.head(top_n).to_string())
    max_corr = corr.iloc[0] if len(corr) > 0 else 0
    if max_corr > 0.7:
        print(f"\n⚠️ LEAKAGE DETECTED: {corr.index[0]} (corr={max_corr:.3f})")
    else:
        print(f"\n✅ Max correlation = {max_corr:.3f} — no obvious leakage")
    print("-" * 60 + "\n")

def run_clustering(X_train, n_clusters=4):
    print(f"🔬 Clustering KMeans ({n_clusters} segments)...")
    numeric_data = X_train.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(numeric_data)
    score = silhouette_score(numeric_data, clusters) if numeric_data.shape[1] >= 2 else 0.0
    print(f"✅ Silhouette Score: {score:.3f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    print("💾 KMeans saved → models/kmeans_model.pkl\n")
    return kmeans

def train_random_forest(X_train_enc, X_test_enc, y_train, y_test):
    print("🔥 Training RandomForest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_enc, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_enc)
    
    print(f"📊 Best hyperparameters: {grid_search.best_params_}\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))
    print(f"→ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"→ F1-Score (Churn): {f1_score(y_test, y_pred):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loyal', 'Churn'], yticklabels=['Loyal', 'Churn'])
    plt.title("Confusion Matrix - Churn Prediction")
    plt.ylabel("True Label")
    plt.xlabel("Prediction")
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    feat_imp = pd.Series(best_rf.feature_importances_, index=X_train_enc.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind='bar')
    plt.title("Top 20 Features - RandomForest")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Reports saved → reports/\n")
    
    joblib.dump(best_rf, "models/randomforest_churn.pkl")
    print("💾 RandomForest saved → models/randomforest_churn.pkl\n")
    return best_rf