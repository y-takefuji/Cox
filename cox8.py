import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=20):
    """Generate synthetic data for COPD mortality and air pollution analysis"""
    
    # Generate features (air pollution and other environmental variables)
    X = np.random.randn(n_samples, n_features)
    
    # Name the features
    feature_names = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Temperature', 'Humidity', 
        'Wind_Speed', 'Precipitation', 'Season', 'Population_Density',
        'Green_Space', 'Traffic_Volume', 'Industrial_Proximity', 'Smoking_Rate',
        'Avg_Age', 'Healthcare_Access', 'Socioeconomic_Status', 'Previous_Respiratory_Issues'
    ]
    
    # Create DataFrame with features
    df = pd.DataFrame(X, columns=feature_names)
    
    # Create additional columns for survival analysis
    # Time until death or censoring (in months)
    df['time'] = np.abs(5 * np.random.randn(n_samples) + 
                        0.3 * df['PM2.5'] + 
                        0.2 * df['NO2'] + 
                        0.25 * df['Smoking_Rate'] + 
                        0.15 * df['Previous_Respiratory_Issues'] +
                        0.1 * df['Avg_Age']) + 1
    
    # Event indicator (1 = death, 0 = censored)
    linear_pred = (0.4 * df['PM2.5'] + 
                   0.3 * df['NO2'] + 
                   0.35 * df['Smoking_Rate'] + 
                   0.25 * df['Previous_Respiratory_Issues'] +
                   0.2 * df['Avg_Age'])
    
    prob = 1 / (1 + np.exp(-linear_pred))
    df['event'] = (np.random.random(n_samples) < prob).astype(int)
    
    # Binary target for classification (for random forest)
    df['mortality'] = df['event']
    
    return df

# Generate data
data = generate_synthetic_data(1000, 20)
print("Dataset shape:", data.shape)
print("\nData preview:")
print(data.head())

# Split the data
X = data.drop(['time', 'event', 'mortality'], axis=1)
y = data['mortality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Cox Proportional Hazards Model
def cox_feature_selection(data, n_features=8):
    """Feature selection using Cox Proportional Hazards model"""
    
    # Prepare data for Cox model
    cox_data = data.copy()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='time', event_col='event')
    
    # Get summary and sort by p-value
    cox_summary = cph.summary
    cox_summary = cox_summary.sort_values('p')
    
    print("\nCox Proportional Hazards Model Summary:")
    print(cox_summary[['coef', 'exp(coef)', 'p']])
    
    # Select top n_features by significance (lowest p-value)
    # Filter out 'mortality' if it appears in the summary
    cox_features = [col for col in cox_summary.index if col != 'mortality']
    top_features = cox_features[:n_features]
    
    return list(top_features), cox_summary

# Method 2: Feature Agglomeration - Fixed
def feature_agglomeration(X, feature_names, n_clusters=8):
    """Feature selection using Feature Agglomeration"""
    
    X_array = X.values if hasattr(X, 'values') else X
    
    # Apply feature agglomeration
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X_array)
    
    # Get feature clusters
    clusters = {}
    for i, label in enumerate(agglo.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(feature_names[i])
    
    # Select one representative feature from each cluster (highest variance)
    selected_features = []
    for cluster_idx, features in clusters.items():
        if len(features) == 1:
            selected_features.append(features[0])
        else:
            # Get indices for features in this cluster
            feature_indices = [feature_names.index(f) for f in features]
            
            # Calculate variance for each feature in the cluster
            feature_variances = np.var(X_array[:, feature_indices], axis=0)
            
            # Select feature with highest variance
            selected_feature = features[np.argmax(feature_variances)]
            selected_features.append(selected_feature)
    
    print("\nFeature Agglomeration Clusters:")
    for cluster_idx, features in clusters.items():
        print(f"Cluster {cluster_idx}: {features}")
    
    return selected_features

# Method 3: Highly Variable Gene Selection (adapted for environmental features)
def highly_variable_selection(X, feature_names, n_features=8):
    """Select features with highest variance"""
    
    # Calculate variance for each feature
    variances = X.var().sort_values(ascending=False)
    
    print("\nFeatures ranked by variance:")
    print(variances)
    
    # Select top n_features with highest variance
    top_features = variances.index[:n_features].tolist()
    
    return top_features

# Method 4: Spearman Correlation with target
def spearman_correlation_selection(X, y, feature_names, n_features=8):
    """Select features with highest absolute Spearman correlation with target"""
    
    correlations = []
    p_values = []
    
    # Calculate Spearman correlation for each feature
    for feature in feature_names:
        corr, p_value = spearmanr(X[feature], y)
        correlations.append(abs(corr))  # Take absolute correlation
        p_values.append(p_value)
    
    # Create DataFrame with correlations and p-values
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations,
        'p_value': p_values
    })
    
    # Sort by absolute correlation
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    print("\nSpearman Correlations with target:")
    print(corr_df)
    
    # Select top features with highest absolute correlation
    top_features = corr_df['feature'][:n_features].tolist()
    
    return top_features, corr_df

# Apply all feature selection methods
print("\n--- Feature Selection Methods ---")

# Set number of features to select
n_features = 8

# Method 1: Cox model
cox_features, cox_summary = cox_feature_selection(data, n_features)
print(f"\nTop {n_features} features selected by Cox model: {cox_features}")

# Method 2: Feature Agglomeration
fa_features = feature_agglomeration(X, X.columns.tolist(), n_features)
print(f"\nTop {n_features} features selected by Feature Agglomeration: {fa_features}")

# Method 3: Highly Variable Features
hvgs_features = highly_variable_selection(X, X.columns.tolist(), n_features)
print(f"\nTop {n_features} features selected by highest variance: {hvgs_features}")

# Method 4: Spearman Correlation
spearman_features, spearman_df = spearman_correlation_selection(X, y, X.columns.tolist(), n_features)
print(f"\nTop {n_features} features selected by Spearman correlation: {spearman_features}")

# Evaluate feature sets with Random Forest for all methods
def evaluate_feature_set(X, y, features, name):
    """Evaluate a feature set using Random Forest with cross-validation"""
    
    # Select only the specified features
    X_selected = X[features]
    
    # Initialize Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
    
    print(f"\n{name} - Random Forest Cross-Validation:")
    print(f"Accuracy scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_scores.mean(), cv_scores.std()

# Evaluate each feature set
print("\n--- Feature Set Evaluation ---")
cox_score, cox_std = evaluate_feature_set(X, y, cox_features, "Cox Model Features")
fa_score, fa_std = evaluate_feature_set(X, y, fa_features, "Feature Agglomeration Features")
hvgs_score, hvgs_std = evaluate_feature_set(X, y, hvgs_features, "Highly Variable Features")
spearman_score, spearman_std = evaluate_feature_set(X, y, spearman_features, "Spearman Correlation Features")

# Compare results
results = pd.DataFrame({
    'Feature Selection Method': ['Cox Model', 'Feature Agglomeration', 'Highly Variable', 'Spearman Correlation'],
    'Selected Features': [cox_features, fa_features, hvgs_features, spearman_features],
    'Mean Accuracy': [cox_score, fa_score, hvgs_score, spearman_score],
    'Standard Deviation': [cox_std, fa_std, hvgs_std, spearman_std]
})

print("\n--- Comparison of Feature Selection Methods ---")
print(results.sort_values('Mean Accuracy', ascending=False))

# Print feature sets
print("\n--- Selected Feature Sets (Top 8) ---")
print(f"\nCox Model Features: {cox_features}")
print(f"\nFeature Agglomeration Features: {fa_features}")
print(f"\nHighly Variable Features: {hvgs_features}")
print(f"\nSpearman Correlation Features: {spearman_features}")
