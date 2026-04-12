import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

# ============================================
# STEP 1: LOAD YOUR PREPARED DATA
# ============================================

def load_churn_data():
    """Load the train/test split data for churn prediction"""
    
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()  # Churn column
    y_test = pd.read_csv("data/train_test/y_test.csv").squeeze()
    
    print(f"📊 Données chargées:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape} (Churn distribution: {y_train.value_counts().to_dict()})")
    print(f"   y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


# ============================================
# STEP 2: ENCODE CATEGORICAL FEATURES
# ============================================

def encode_features_for_xgboost(X_train, X_test):
    """
    XGBoost needs numerical inputs.
    Convert categorical features to numbers.
    """
    
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n🔄 Encodage des features:")
    print(f"   Features numériques: {len(numerical_cols)}")
    print(f"   Features catégorielles: {len(categorical_cols)}")
    
    if categorical_cols:
        print(f"   Colonnes catégorielles: {categorical_cols[:5]}...")  # Show first 5
    
    # Create a copy to avoid modifying original
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # Encode each categorical column
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Combine train and test to handle all categories
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    print(f"✅ Encodage terminé")
    
    return X_train_encoded, X_test_encoded, encoders


# ============================================
# STEP 3: TRAIN XGBOOST WITH DEFAULT PARAMETERS (BASELINE)
# ============================================

def train_xgboost_baseline(X_train, y_train):
    """Train XGBoost with default parameters to get baseline"""
    
    print("\n" + "="*50)
    print("🚀 TRAINING XGBOOST BASELINE MODEL")
    print("="*50)
    
    # Handle class imbalance automatically
    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts[0] / counts[1]  # ratio of negative to positive
    
    print(f"   Class distribution: {dict(zip(unique, counts))}")
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Handle imbalanced churn data
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Baseline model trained")
    
    return model


# ============================================
# STEP 4: HYPERPARAMETER TUNING WITH GRIDSEARCH
# ============================================

def tune_xgboost_hyperparameters(X_train, y_train):
    """Find the best hyperparameters using GridSearchCV"""
    
    print("\n" + "="*50)
    print("🔧 HYPERPARAMETER TUNING WITH GRIDSEARCH")
    print("="*50)
    
    # Handle class imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts[0] / counts[1]
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create base model
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    print(f"📋 Grid à tester: {sum(len(v) for v in param_grid.values())} combinaisons")
    print("   Cela peut prendre quelques minutes...")
    
    # GridSearch with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='f1',  # Optimize for F1 score (good for imbalanced data)
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Meilleurs hyperparamètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


# ============================================
# STEP 5: EVALUATE THE MODEL
# ============================================

def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation of the model"""
    
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn (class 1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Métriques de performance:")
    print(f"   Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}  ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}    ({f1*100:.2f}%)")
    print(f"   ROC-AUC:   {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    return y_pred, y_pred_proba




import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    print("="*60)
    print("🏃‍♂️ STARTING XGBOOST CHURN MODEL TRAINING")
    print("="*60)
    
    # Load data
    train_data = pd.read_csv("data/train_test/train_data.csv")
    test_data = pd.read_csv("data/train_test/test_data.csv")
    
    # Separate features and target
    X_train = train_data.drop(columns=['Churn'])
    y_train = train_data['Churn']
    
    X_test = test_data.drop(columns=['Churn'])
    y_test = test_data['Churn']
    
    # First alignment
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"📊 Après première alignment: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Remove problematic columns
    columns_to_remove = ['CustomerID', 'ChurnRiskCategory', 'NewsletterSubscribed', 'LastLoginIP', 'RegistrationDate']
    
    for col in columns_to_remove:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])
    
    # SECOND ALIGNMENT - THIS IS CRITICAL
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"📊 Après suppression: X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Colonnes restantes: {X_train.columns.tolist()}")
    
    # Encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train_enc[col] = le.transform(X_train[col].astype(str))
        X_test_enc[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    print(f"✅ Encodage terminé: {X_train_enc.shape[1]} features")
    
    from sklearn.model_selection import GridSearchCV

    # Define parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5]
    }

    # Create XGBoost model
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross-validation
        scoring='roc_auc',       # optimize for ROC-AUC
        n_jobs=-1,               # use all CPU cores
        verbose=1
    )

    # Run the search
    grid_search.fit(X_train_enc, y_train)

    # Get best parameters
    print(f"\n🏆 Best hyperparameters: {grid_search.best_params_}")
    print(f"📈 Best ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # Use the best model
    best_model = grid_search.best_estimator_  # ← THIS IS YOUR TUNED MODEL

    print(f"\n🏆 Best hyperparameters: {grid_search.best_params_}")
    print(f"📈 Best ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # Evaluate the TUNED model (NOT baseline)
    print("\n" + "="*50)
    print("📊 TUNED MODEL PERFORMANCE ON TEST SET")
    print("="*50)

    y_pred = best_model.predict(X_test_enc)
    y_pred_proba = best_model.predict_proba(X_test_enc)[:, 1]

    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Save the TUNED model
    joblib.dump(best_model, "models/xgboost_churn_model.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    
    print("\n💾 Model saved to: models/xgboost_churn_model.pkl")
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)