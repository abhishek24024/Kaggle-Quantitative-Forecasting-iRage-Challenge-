
"""
V6.1 Pipeline - GPU-Accelerated XGBRFRegressor
===============================================
Key innovations (inherited from v6.0):
1. Orthogonal windows: W₁=LagT1, W₂=LagT2-LagT1, W₃=LagT3-LagT2
2. Drop ALL Lag0 (base features) - non-stationary
3. Re-cluster on W features, select by VARIANCE (zero look-ahead)
4. SO3_T one-hot encoding for regime isolation
5. Target volatility scaling replaces manual shrinkage
6. Ensemble: 40% Ridge + 40% XGB-RF + 20% Huber-LightGBM
7. GroupKFold by CV_GROUP for chronological validation
v6.1 CHANGES:
- Replace sklearn RandomForestRegressor with xgb.XGBRFRegressor
- GPU acceleration via tree_method='hist', device='cuda'
- True bagging (not boosting) - independent parallel trees
- colsample_bynode=0.15 for per-split feature randomization (true RF behavior)
- reg_lambda=2.0 for L2 regularization on leaf weights
- Expected runtime: ~15-30 min (vs ~4 hours in v6.0)
"""
import numpy as np
import pandas as pd
import gc
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')
# =============================================================================
# CONFIGURATION
# =============================================================================
IS_KAGGLE = True  # Set to True when running on Kaggle
if IS_KAGGLE:
    TRAIN_PATH = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/train.parquet'
    TEST_PATH = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/test.parquet'
else:
    TRAIN_PATH = 'DATASET/short-horizon-return-prediction-challenge-by-i-rage/train-001.parquet'
    TEST_PATH = 'DATASET/short-horizon-return-prediction-challenge-by-i-rage/test.parquet'
# Clustering config
CLUSTER_DISTANCE_THRESHOLD = 0.30  # Cut dendrogram at 70% correlation
# Ensemble weights
RIDGE_WEIGHT = 0.40
RF_WEIGHT = 0.40
LGBM_WEIGHT = 0.20
# Model hyperparameters - Ridge
RIDGE_ALPHA = 2000
# Model hyperparameters - XGBRFRegressor (GPU-accelerated Random Forest)
RF_N_ESTIMATORS = 1000          # Back to 1000 - GPU can handle it
RF_MAX_DEPTH = 6                # Same as v6.0
RF_SUBSAMPLE = 0.8              # Row bagging (like bootstrap)
RF_COLSAMPLE_BYNODE = 0.15      # Feature sampling PER NODE (true RF behavior)
RF_REG_LAMBDA = 2.0             # L2 regularization on leaf weights
# Model hyperparameters - LightGBM (CPU, already fast)
LGBM_NUM_LEAVES = 8
LGBM_MAX_DEPTH = 3
# CV config
N_SPLITS = 5
# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
def load_data():
    """Load train and test data with float32 optimization."""
    print("=" * 60)
    print("STEP 1: Loading data...")
    
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    
    return train_df, test_df
# =============================================================================
# STEP 2: IDENTIFY FEATURES WITH LAGS
# =============================================================================
def identify_lag_features(columns):
    """
    Find all features that have LagT1, LagT2, LagT3 versions.
    Returns dict: {base_name: [LagT1_col, LagT2_col, LagT3_col]}
    """
    print("=" * 60)
    print("STEP 2: Identifying features with lag versions...")
    
    lag1_cols = [c for c in columns if '_LagT1' in c]
    
    features_with_lags = {}
    for lag1_col in lag1_cols:
        base_name = lag1_col.replace('_LagT1', '')
        lag2_col = f"{base_name}_LagT2"
        lag3_col = f"{base_name}_LagT3"
        
        # Verify all 3 lags exist
        if lag2_col in columns and lag3_col in columns:
            features_with_lags[base_name] = {
                'lag0': base_name,  # Original feature (to be dropped)
                'lag1': lag1_col,
                'lag2': lag2_col,
                'lag3': lag3_col
            }
    
    print(f"  Features with all 3 lags: {len(features_with_lags)}")
    print(f"  Sample: {list(features_with_lags.keys())[:5]}")
    
    # Check SO3_T (should NOT be in this list)
    if 'SO3_T' in features_with_lags:
        print("  WARNING: SO3_T found in lag features - removing")
        del features_with_lags['SO3_T']
    
    return features_with_lags
# =============================================================================
# STEP 3: COMPUTE ORTHOGONAL WINDOWS (W₁, W₂, W₃)
# =============================================================================
def compute_orthogonal_windows(df, features_with_lags):
    """
    Compute orthogonal momentum windows:
    - W₁ = LagT1 (recent momentum: t-T1 → t)
    - W₂ = LagT2 - LagT1 (mid momentum: t-T2 → t-T1)
    - W₃ = LagT3 - LagT2 (distant momentum: t-T3 → t-T2)
    
    Returns new DataFrame with W features only.
    """
    print("=" * 60)
    print("STEP 3: Computing orthogonal windows (W₁, W₂, W₃)...")
    
    w_features = {}
    
    for base_name, lags in features_with_lags.items():
        # W₁ = LagT1 (already a difference: feature[t] - feature[t-T1])
        w1_name = f"{base_name}_W1"
        w_features[w1_name] = df[lags['lag1']].values
        
        # W₂ = LagT2 - LagT1
        # LagT2 = feature[t] - feature[t-T2]
        # LagT1 = feature[t] - feature[t-T1]
        # W₂ = (feature[t] - feature[t-T2]) - (feature[t] - feature[t-T1])
        #    = feature[t-T1] - feature[t-T2] (momentum from t-T2 to t-T1)
        w2_name = f"{base_name}_W2"
        w_features[w2_name] = df[lags['lag2']].values - df[lags['lag1']].values
        
        # W₃ = LagT3 - LagT2 (momentum from t-T3 to t-T2)
        w3_name = f"{base_name}_W3"
        w_features[w3_name] = df[lags['lag3']].values - df[lags['lag2']].values
    
    w_df = pd.DataFrame(w_features)
    
    print(f"  Created {len(w_df.columns)} W features")
    print(f"  W1 features: {len([c for c in w_df.columns if '_W1' in c])}")
    print(f"  W2 features: {len([c for c in w_df.columns if '_W2' in c])}")
    print(f"  W3 features: {len([c for c in w_df.columns if '_W3' in c])}")
    
    # Convert to float32 to save memory
    w_df = w_df.astype(np.float32)
    
    return w_df
# =============================================================================
# STEP 4: DROP LAG0 (Already done - we only use W features)
# =============================================================================
# No explicit step needed - we build W features directly without Lag0
# =============================================================================
# STEP 5: HIERARCHICAL CLUSTERING ON W FEATURES
# =============================================================================
def cluster_w_features(w_df, distance_threshold=0.30):
    """
    Perform hierarchical clustering on W features.
    Cut dendrogram at distance_threshold (0.30 = 70% correlation).
    Returns cluster assignments.
    """
    print("=" * 60)
    print("STEP 5: Hierarchical clustering on W features...")
    
    # Compute Spearman correlation matrix
    print("  Computing Spearman correlation matrix...")
    corr_matrix = w_df.corr(method='spearman').values
    
    # Convert correlation to distance (1 - |corr|)
    # Use absolute correlation to group both positive and negative correlations
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Ensure symmetry and no negative values
    distance_matrix = np.clip(distance_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form for linkage
    condensed_dist = squareform(distance_matrix)
    
    # Hierarchical clustering
    print("  Performing hierarchical clustering...")
    Z = linkage(condensed_dist, method='average')
    
    # Cut dendrogram at threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
    
    n_clusters = len(np.unique(cluster_labels))
    print(f"  Number of clusters at distance {distance_threshold}: {n_clusters}")
    
    # Create cluster assignment DataFrame
    cluster_df = pd.DataFrame({
        'feature': w_df.columns,
        'cluster': cluster_labels
    })
    
    # Show cluster sizes
    cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
    print(f"  Cluster sizes (min/max/mean): {cluster_sizes.min()}/{cluster_sizes.max()}/{cluster_sizes.mean():.1f}")
    
    return cluster_df, Z
# =============================================================================
# STEP 6: SELECT REPRESENTATIVES BY VARIANCE
# =============================================================================
def select_representatives_by_variance(w_df, cluster_df):
    """
    Select one representative per cluster based on highest VARIANCE.
    This guarantees zero look-ahead bias (variance never touches TARGET).
    """
    print("=" * 60)
    print("STEP 6: Selecting cluster representatives by variance...")
    
    # Compute variance for each feature
    variances = w_df.var()
    
    representatives = []
    
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        cluster_features = cluster_df[cluster_df['cluster'] == cluster_id]['feature'].tolist()
        
        # Find feature with highest variance in this cluster
        cluster_variances = variances[cluster_features]
        best_feature = cluster_variances.idxmax()
        best_variance = cluster_variances.max()
        
        representatives.append({
            'cluster': cluster_id,
            'feature': best_feature,
            'variance': best_variance,
            'cluster_size': len(cluster_features)
        })
    
    rep_df = pd.DataFrame(representatives)
    
    print(f"  Selected {len(rep_df)} representative features")
    print(f"  Top 5 by variance:")
    print(rep_df.nlargest(5, 'variance')[['feature', 'variance', 'cluster_size']].to_string(index=False))
    
    return rep_df['feature'].tolist()
# =============================================================================
# STEP 7: AUTO-DETECT SO3_T BANDS
# =============================================================================
def detect_so3t_bands(so3t_values):
    """
    Auto-detect discrete bands in SO3_T using histogram analysis.
    Returns band edges for one-hot encoding.
    """
    print("=" * 60)
    print("STEP 7: Auto-detecting SO3_T bands...")
    
    # Use deciles as bands (10 bands)
    n_bands = 10
    band_edges = np.percentile(so3t_values, np.linspace(0, 100, n_bands + 1))
    
    # Ensure unique edges
    band_edges = np.unique(band_edges)
    
    print(f"  Detected {len(band_edges) - 1} bands")
    print(f"  Band edges: {band_edges[:5]}... (showing first 5)")
    
    return band_edges
def one_hot_encode_so3t(so3t_values, band_edges):
    """
    One-hot encode SO3_T into discrete bands.
    Returns DataFrame with one column per band.
    """
    # Assign each value to a band
    band_indices = np.digitize(so3t_values, band_edges[1:-1])  # Exclude first and last edges
    
    # Create one-hot encoding
    n_bands = len(band_edges) - 1
    one_hot = np.zeros((len(so3t_values), n_bands), dtype=np.float32)
    
    for i, band_idx in enumerate(band_indices):
        one_hot[i, band_idx] = 1.0
    
    # Create DataFrame with band names
    band_names = [f'SO3_T_band_{i}' for i in range(n_bands)]
    one_hot_df = pd.DataFrame(one_hot, columns=band_names)
    
    return one_hot_df, band_indices
# =============================================================================
# STEP 8-9: COMPUTE REGIME STD AND SCALE TARGET
# =============================================================================
def compute_regime_stats(target, band_indices):
    """
    Compute TARGET statistics per SO3_T band.
    Returns dict: {band_idx: {'mean': m, 'std': s}}
    """
    print("=" * 60)
    print("STEP 8: Computing regime statistics...")
    
    regime_stats = {}
    
    for band_idx in np.unique(band_indices):
        mask = band_indices == band_idx
        band_target = target[mask]
        
        regime_stats[band_idx] = {
            'mean': float(np.mean(band_target)),
            'std': float(np.std(band_target)),
            'count': int(np.sum(mask))
        }
    
    # Print stats
    print("  Band  |  Count  |  Mean    |  Std")
    print("  " + "-" * 40)
    for band_idx in sorted(regime_stats.keys()):
        stats = regime_stats[band_idx]
        print(f"  {band_idx:5d} | {stats['count']:7d} | {stats['mean']:8.5f} | {stats['std']:.5f}")
    
    return regime_stats
def scale_target_by_regime(target, band_indices, regime_stats):
    """
    Divide TARGET by regime std during training.
    This normalizes learning across high/low volatility regimes.
    """
    print("=" * 60)
    print("STEP 9: Scaling TARGET by regime volatility...")
    
    scaled_target = np.zeros_like(target)
    
    for band_idx, stats in regime_stats.items():
        mask = band_indices == band_idx
        # Avoid division by zero
        regime_std = max(stats['std'], 1e-8)
        scaled_target[mask] = target[mask] / regime_std
    
    print(f"  Original target std: {np.std(target):.6f}")
    print(f"  Scaled target std: {np.std(scaled_target):.6f}")
    
    return scaled_target
def rescale_predictions(predictions, band_indices, regime_stats):
    """
    Multiply predictions back by regime std during inference.
    """
    rescaled = np.zeros_like(predictions)
    
    for band_idx, stats in regime_stats.items():
        mask = band_indices == band_idx
        regime_std = max(stats['std'], 1e-8)
        rescaled[mask] = predictions[mask] * regime_std
    
    return rescaled
# =============================================================================
# STEP 10-14: TRAINING WITH GROUPKFOLD AND ENSEMBLE
# =============================================================================
def train_ensemble(X_train, y_train, cv_groups, representative_features):
    """
    Train 3-model ensemble with GroupKFold cross-validation.
    Returns trained models and CV scores.
    
    v6.1: Uses XGBRFRegressor (GPU) instead of sklearn RandomForestRegressor (CPU)
    """
    print("=" * 60)
    print("STEP 10-14: Training ensemble models...")
    print("  [v6.1] Using XGBRFRegressor with GPU acceleration")
    
    # Memory optimization
    X_train = X_train.astype(np.float32)
    gc.collect()
    
    # GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    # Storage for CV predictions
    oof_ridge = np.zeros(len(y_train))
    oof_rf = np.zeros(len(y_train))
    oof_lgbm = np.zeros(len(y_train))
    
    # Final models (trained on full data)
    ridge_models = []
    rf_models = []
    lgbm_models = []
    
    print(f"\n  Training with {N_SPLITS}-fold GroupKFold...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, cv_groups)):
        print(f"\n  === Fold {fold + 1}/{N_SPLITS} ===")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        gc.collect()
        
        # --- Ridge ---
        print(f"    Training Ridge (α={RIDGE_ALPHA})...")
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_val)
        ridge_models.append(ridge)
        
        gc.collect()
        
        # --- XGBRFRegressor (GPU-accelerated Random Forest) ---
        print(f"    Training XGB-RF GPU (n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH})...")
        rf = xgb.XGBRFRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            subsample=RF_SUBSAMPLE,
            colsample_bynode=RF_COLSAMPLE_BYNODE,
            reg_lambda=RF_REG_LAMBDA,
            tree_method='hist',
            device='cuda',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict(X_val)
        rf_models.append(rf)
        
        gc.collect()
        
        # --- Huber LightGBM ---
        print(f"    Training Huber-LightGBM (leaves={LGBM_NUM_LEAVES}, depth={LGBM_MAX_DEPTH})...")
        lgbm_model = lgb.LGBMRegressor(
            objective='huber',
            num_leaves=LGBM_NUM_LEAVES,
            max_depth=LGBM_MAX_DEPTH,
            learning_rate=0.05,
            n_estimators=500,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=2
        )
        lgbm_model.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = lgbm_model.predict(X_val)
        lgbm_models.append(lgbm_model)
        
        gc.collect()
        
        # Fold metrics
        fold_r2_ridge = 1 - np.var(y_val - oof_ridge[val_idx]) / np.var(y_val)
        fold_r2_rf = 1 - np.var(y_val - oof_rf[val_idx]) / np.var(y_val)
        fold_r2_lgbm = 1 - np.var(y_val - oof_lgbm[val_idx]) / np.var(y_val)
        
        print(f"    Fold R² - Ridge: {fold_r2_ridge:.6f}, XGB-RF: {fold_r2_rf:.6f}, LGBM: {fold_r2_lgbm:.6f}")
    
    # Ensemble OOF predictions
    oof_ensemble = (RIDGE_WEIGHT * oof_ridge + 
                    RF_WEIGHT * oof_rf + 
                    LGBM_WEIGHT * oof_lgbm)
    
    # Overall CV metrics
    cv_r2_ridge = 1 - np.var(y_train - oof_ridge) / np.var(y_train)
    cv_r2_rf = 1 - np.var(y_train - oof_rf) / np.var(y_train)
    cv_r2_lgbm = 1 - np.var(y_train - oof_lgbm) / np.var(y_train)
    cv_r2_ensemble = 1 - np.var(y_train - oof_ensemble) / np.var(y_train)
    
    print("\n" + "=" * 60)
    print("OVERALL CV R² SCORES:")
    print(f"  Ridge:    {cv_r2_ridge:.6f}")
    print(f"  XGB-RF:   {cv_r2_rf:.6f}")
    print(f"  LightGBM: {cv_r2_lgbm:.6f}")
    print(f"  Ensemble: {cv_r2_ensemble:.6f}")
    print("=" * 60)
    
    return {
        'ridge_models': ridge_models,
        'rf_models': rf_models,
        'lgbm_models': lgbm_models,
        'cv_scores': {
            'ridge': cv_r2_ridge,
            'rf': cv_r2_rf,
            'lgbm': cv_r2_lgbm,
            'ensemble': cv_r2_ensemble
        }
    }
# =============================================================================
# STEP 15-16: PREDICT AND GENERATE SUBMISSION
# =============================================================================
def predict_ensemble(X_test, models):
    """
    Generate predictions using ensemble of models.
    Average predictions across folds.
    """
    print("=" * 60)
    print("STEP 15: Generating ensemble predictions...")
    
    X_test = X_test.astype(np.float32)
    
    # Ridge predictions
    ridge_preds = np.mean([m.predict(X_test) for m in models['ridge_models']], axis=0)
    
    # XGB-RF predictions
    rf_preds = np.mean([m.predict(X_test) for m in models['rf_models']], axis=0)
    
    # LightGBM predictions
    lgbm_preds = np.mean([m.predict(X_test) for m in models['lgbm_models']], axis=0)
    
    # Ensemble
    ensemble_preds = (RIDGE_WEIGHT * ridge_preds + 
                      RF_WEIGHT * rf_preds + 
                      LGBM_WEIGHT * lgbm_preds)
    
    print(f"  Prediction stats - mean: {ensemble_preds.mean():.6f}, std: {ensemble_preds.std():.6f}")
    
    return ensemble_preds
def create_submission(test_ids, predictions, filename='submission.csv'):
    """Create submission file."""
    print("=" * 60)
    print(f"STEP 16: Creating submission '{filename}'...")
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"  Saved {len(submission)} rows")
    print(f"  Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    return submission
# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("V6.1 PIPELINE - GPU-ACCELERATED XGBRFRegressor")
    print("=" * 60)
    
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Extract IDs and metadata
    train_ids = train_df['ID'].values
    test_ids = test_df['ID'].values
    target = train_df['TARGET'].values
    cv_groups = train_df['CV_GROUP'].values
    
    # Step 2: Identify features with lags
    features_with_lags = identify_lag_features(train_df.columns)
    
    # Step 3: Compute orthogonal windows for train
    print("\n  Processing TRAIN data...")
    w_train = compute_orthogonal_windows(train_df, features_with_lags)
    
    # Step 3b: Compute orthogonal windows for test
    print("\n  Processing TEST data...")
    w_test = compute_orthogonal_windows(test_df, features_with_lags)
    
    # Free memory
    del train_df
    gc.collect()
    
    # Step 5: Cluster W features
    cluster_df, _ = cluster_w_features(w_train, CLUSTER_DISTANCE_THRESHOLD)
    
    # Step 6: Select representatives by variance
    representative_features = select_representatives_by_variance(w_train, cluster_df)
    
    # Subset to representative features
    X_train_w = w_train[representative_features].copy()
    X_test_w = w_test[representative_features].copy()
    
    del w_train, w_test
    gc.collect()
    
    # Step 7: SO3_T band detection and one-hot encoding
    # Reload SO3_T from test (we deleted train_df)
    test_df_so3t = pd.read_parquet(TEST_PATH, columns=['SO3_T'])
    train_df_so3t = pd.read_parquet(TRAIN_PATH, columns=['SO3_T'])
    
    train_so3t = train_df_so3t['SO3_T'].values
    test_so3t = test_df_so3t['SO3_T'].values
    
    # Detect bands from training data
    band_edges = detect_so3t_bands(train_so3t)
    
    # One-hot encode
    train_so3t_onehot, train_band_indices = one_hot_encode_so3t(train_so3t, band_edges)
    test_so3t_onehot, test_band_indices = one_hot_encode_so3t(test_so3t, band_edges)
    
    print(f"  Train SO3_T one-hot shape: {train_so3t_onehot.shape}")
    print(f"  Test SO3_T one-hot shape: {test_so3t_onehot.shape}")
    
    # Step 8: Compute regime stats
    regime_stats = compute_regime_stats(target, train_band_indices)
    
    # Step 9: Scale target by regime
    scaled_target = scale_target_by_regime(target, train_band_indices, regime_stats)
    
    # Combine W features with SO3_T one-hot
    X_train_final = pd.concat([
        X_train_w.reset_index(drop=True), 
        train_so3t_onehot.reset_index(drop=True)
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test_w.reset_index(drop=True), 
        test_so3t_onehot.reset_index(drop=True)
    ], axis=1)
    
    print(f"\n  Final train features: {X_train_final.shape}")
    print(f"  Final test features: {X_test_final.shape}")
    
    del X_train_w, X_test_w, train_so3t_onehot, test_so3t_onehot
    gc.collect()
    
    # Steps 10-14: Train ensemble on SCALED target
    models = train_ensemble(X_train_final, scaled_target, cv_groups, representative_features)
    
    # Step 15: Generate predictions
    scaled_predictions = predict_ensemble(X_test_final, models)
    
    # Rescale predictions by regime std
    print("\n  Rescaling predictions by regime volatility...")
    final_predictions = rescale_predictions(scaled_predictions, test_band_indices, regime_stats)
    
    print(f"  Final prediction stats - mean: {final_predictions.mean():.6f}, std: {final_predictions.std():.6f}")
    
    # Step 16: Create submission
    submission = create_submission(test_ids, final_predictions, 'submission_v6.1.csv')
    
    print("\n" + "=" * 60)
    print("V6.1 PIPELINE COMPLETE!")
    print("=" * 60)
    
    return models, submission
if __name__ == '__main__':
    models, submission = main()


