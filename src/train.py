# =============================================
# src/train_model.py  (Version XGBoost)
# =============================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    silhouette_score
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

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


def load_prepared_data():
    train_df = pd.read_csv("data/train_test/train_data.csv")
    test_df = pd.read_csv("data/train_test/test_data.csv")

    y_train = train_df["Churn"]
    X_train = train_df.drop(columns=["Churn"])

    y_test = test_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    print(f"✅ Données chargées : {X_train.shape[1]} features | "
          f"{X_train.shape[0]} train | {X_test.shape[0]} test\n")
    return X_train, X_test, y_train, y_test


def remove_leaky_features(X_train, X_test):
    to_drop = set()
    for col in X_train.columns:
        if col in LEAKY_EXACT or any(kw in col.lower() for kw in LEAKY_KEYWORDS):
            to_drop.add(col)

    if to_drop:
        print(f"🛡️ {len(to_drop)} features leakantes supprimées : {sorted(to_drop)}")
        X_train = X_train.drop(columns=[c for c in to_drop if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    else:
        print("✅ Aucune feature leakante détectée.")

    print(f"   → Features restantes : {X_train.shape[1]}\n")
    return X_train, X_test


def build_encoder(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    ordinal_candidates = [
        'AgeCategory', 'BasketSizeCategory', 'PreferredTimeOfDay',
        'ProductDiversity', 'WeekendPreference'
    ]

    ordinal_cols = [c for c in cat_cols if c in ordinal_candidates]
    onehot_cols = [c for c in cat_cols if c not in ordinal_cols]

    transformers = []

    if onehot_cols:
        transformers.append(
            ('onehot',
             OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             onehot_cols)
        )

    if ordinal_cols:
        transformers.append(
            ('ordinal',
             OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             ordinal_cols)
        )

    encoder = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return encoder


def encode_features(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        print("✅ Pas de features catégorielles à encoder.")
        return X_train, X_test, None

    print(f"🔄 Encodage de {len(cat_cols)} variables catégorielles : {cat_cols}")
    # convert categorical columns to string
    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)
    encoder = build_encoder(X_train)
    encoder.fit(X_train)

    X_train_enc = pd.DataFrame(
        encoder.transform(X_train),
        columns=encoder.get_feature_names_out(),
        index=X_train.index
    )

    X_test_enc = pd.DataFrame(
        encoder.transform(X_test),
        columns=encoder.get_feature_names_out(),
        index=X_test.index
    )

    to_drop = [
        c for c in X_train_enc.columns
        if any(kw in c.lower() for kw in LEAKY_KEYWORDS)
    ]

    if to_drop:
        print(f"🛡️ Colonnes encodées leakantes supprimées : {to_drop}")
        X_train_enc = X_train_enc.drop(columns=to_drop)
        X_test_enc = X_test_enc.drop(columns=[c for c in to_drop if c in X_test_enc.columns])

    print(f"✅ Encodage terminé → {X_train_enc.shape[1]} colonnes\n")

    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/encoder.pkl")
    print("💾 Encodeur sauvegardé → models/encoder.pkl\n")

    return X_train_enc, X_test_enc, encoder


def diagnostic(X_train, y_train):
    print("🔍 DIAGNOSTIC – Top 15 corrélations |feature, Churn| :")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    corr = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)

    print(corr.head(15).to_string())

    max_corr = corr.iloc[0] if len(corr) > 0 else 0

    if max_corr > 0.7:
        print(f"\n⚠️ LEAKAGE DÉTECTÉ : {corr.index[0]} (corr={max_corr:.3f})")
    else:
        print(f"\n✅ Corrélation max = {max_corr:.3f} — pas de leakage évident")

    print("-" * 60 + "\n")


def clustering(X_train):
    print("🔬 Clustering KMeans (4 segments)...")
    numeric_data = X_train.select_dtypes(include=[np.number])

    # remplir les NaN avec la médiane
    numeric_data = numeric_data.fillna(numeric_data.median())

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(numeric_data)

    score = silhouette_score(numeric_data, clusters) if numeric_data.shape[1] >= 2 else 0.0
    print(f"✅ Silhouette Score : {score:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    print("💾 KMeans sauvegardé → models/kmeans_model.pkl\n")

    return kmeans


def train_xgboost(X_train_enc, X_test_enc, y_train, y_test):
    print("🔥 Entraînement XGBoost...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1]
    }

    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_enc, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test_enc)

    print(f"📊 Meilleurs hyperparamètres : {grid_search.best_params_}\n")
    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))
    print(f"→ Accuracy        : {accuracy_score(y_test, y_pred):.4f}")
    print(f"→ F1-Score (Churn): {f1_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fidèle', 'Churn'],
        yticklabels=['Fidèle', 'Churn']
    )
    plt.title("Matrice de Confusion — Prédiction Churn")
    plt.ylabel("Vraie Valeur")
    plt.xlabel("Prédiction")

    os.makedirs("reports", exist_ok=True)
    plt.savefig(r"C:\ProjetML\MachineLearningProject\reports\Correlation_heatmap.png",
            dpi=300,
            bbox_inches='tight')
    plt.close()

    feat_imp = pd.Series(
        best_xgb.feature_importances_,
        index=X_train_enc.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind='bar')
    plt.title("Top 20 Features — XGBoost")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Rapports sauvegardés → reports/\n")

    joblib.dump(best_xgb, "models/xgboost_churn.pkl")
    print("💾 XGBoost sauvegardé → models/xgboost_churn.pkl\n")

    return best_xgb


def run_training():
    X_train, X_test, y_train, y_test = load_prepared_data()

    X_train, X_test = remove_leaky_features(X_train, X_test)

    clustering(X_train)

    X_train_enc, X_test_enc, encoder = encode_features(X_train, X_test)

    diagnostic(X_train_enc, y_train)

    train_xgboost(X_train_enc, X_test_enc, y_train, y_test)

    print("🎉 MODÉLISATION TERMINÉE !")
    print("   Modèles dans models/ : xgboost_churn.pkl, kmeans_model.pkl, encoder.pkl")
    print("   Rapports dans reports/ : confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    run_training()