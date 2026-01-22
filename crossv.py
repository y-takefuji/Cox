import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Function to implement Cox model feature selection
def cox_model_selection(X, y, k=10):
    # Create a dataframe for Cox model
    df = X.copy()
    df['time'] = y
    df['event'] = 1  # Assuming all events are observed
    
    # Calculate concordance index and p-values for each feature
    scores = []
    p_values = {}
    
    for i, col in enumerate(X.columns):
        # Create a univariate Cox model for each feature
        feature_df = df[['time', 'event', col]].copy()
        
        try:
            cph = CoxPHFitter()
            cph.fit(feature_df, duration_col='time', event_col='event')
            
            # Extract p-value and concordance index
            p_val = cph.summary.p[0]
            c_index = concordance_index(y, X[col])
            
            p_values[col] = p_val
            scores.append((col, c_index, p_val))
        except Exception as e:
            # Handle errors (e.g., convergence issues)
            print(f"Error calculating Cox p-value for {col}: {e}")
            scores.append((col, 0.5, 1.0))  # Default to neutral values
    
    # Sort features by their concordance index and select top k
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected_features = [col for col, _, _ in scores[:k]]
    
    # Print p-values for selected features
    print(f"\nCox Model Selected Features with p-values (k={k}):")
    for feature in selected_features:
        print(f"{feature}: p-value = {p_values.get(feature, 'N/A'):.6f}")
    
    return selected_features, [score for _, score, _ in scores[:k]]

# Function to implement feature agglomeration
def feature_agglomeration_selection(X, y, k=10):
    n_features = min(k, X.shape[1])
    fa = FeatureAgglomeration(n_clusters=n_features)
    fa.fit(X)
    
    # Identify the most important feature in each cluster
    cluster_importances = {}
    for i, cluster in enumerate(fa.labels_):
        if cluster not in cluster_importances:
            cluster_importances[cluster] = []
        # Calculate correlation with target for ranking
        corr = abs(np.corrcoef(X.iloc[:, i], y)[0, 1])
        cluster_importances[cluster].append((X.columns[i], corr))
    
    # Select the most important feature from each cluster
    selected_features = []
    scores = []
    for cluster, features in cluster_importances.items():
        features.sort(key=lambda x: x[1], reverse=True)
        selected_features.append(features[0][0])
        scores.append(features[0][1])
    
    return selected_features[:k], scores[:k]

# Function to implement highly variable gene selection
def hvgs_selection(X, y, k=10):
    # Calculate variance for each feature
    variances = X.var().sort_values(ascending=False)
    selected_features = variances.index[:k].tolist()
    scores = variances.values[:k].tolist()
    
    return selected_features, scores

# Function to implement Spearman correlation-based selection - MODIFIED FOR POSITIVE CORRELATIONS ONLY
def spearman_selection(X, y, k=10):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        # Only use positive correlation value, not absolute value
        correlations.append((col, corr))
    
    # Sort by correlation coefficient and select top k
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [col for col, _ in correlations[:k]]
    scores = [score for _, score in correlations[:k]]
    
    return selected_features, scores

# Function to evaluate features using Random Forest
def evaluate_features(X, y, feature_names, k, cv=5):
    features = X[feature_names[:k]]
    model = RandomForestRegressor(random_state=42)
    scores = cross_val_score(model, features, y, cv=cv, scoring='r2')
    return np.mean(scores)

# Function to evaluate Cox model using C-index
def evaluate_cox_features(X, y, feature_names, k, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    features = X[feature_names[:k]]
    c_indices = []
    
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train Cox Proportional Hazards model
        df_train = X_train.copy()
        df_train['time'] = y_train
        df_train['event'] = 1  # Assuming all events are observed
        
        cph = CoxPHFitter()
        try:
            cph.fit(df_train, duration_col='time', event_col='event')
            
            # Predict partial hazard
            preds = cph.predict_partial_hazard(X_test)
            
            # Calculate C-index
            c_index = concordance_index(y_test, preds)
            c_indices.append(c_index)
        except Exception as e:
            # In case of convergence issues or other errors
            print(f"Cox model fitting error: {e}")
            c_indices.append(0.5)  # Use neutral value for failed runs    
    return np.mean(c_indices)

# Load dataset
data = pd.read_csv('AQI_Respiratory_2000_2019.csv')
print(data.shape)

# Drop 'FIPS,YEAR' (single variable name)
data = data.drop(['FIPS,YEAR'], axis=1, errors='ignore')

# Drop NaN values
data = data.dropna()

# Separate features and target
X = data.drop(['Resp Death Rate', 'location_name'], axis=1)
y = data['Resp Death Rate']

# Show shape of dataset
print(f"Dataset shape: {data.shape}")

# Show distribution of target
print("\nTarget distribution:")
print(y.describe())

# Feature selection methods
selection_methods = {
    'Cox Model': cox_model_selection,
    'Feature Agglomeration': feature_agglomeration_selection,
    'HVGS': hvgs_selection,
    'Spearman': spearman_selection
}

# Number of features to select
feature_counts = [5, 8, 9, 10]

# Results storage
results = {method: {'cv5': None, 'cv8': None, 'cv9': None, 'cv10': None} for method in selection_methods}

# For each method
for method_name, selection_function in selection_methods.items():
    print(f"\n{'-'*40}\nProcessing {method_name}\n{'-'*40}")
    
    # Select top 10 features from full set
    top_10_features, _ = selection_function(X, y, 10)
    print(f"Top 10 features from full set: {top_10_features}")
    
    # Evaluate for cv10
    if method_name != 'Cox Model':
        accuracy_cv10 = evaluate_features(X, y, top_10_features, 10)
    else:
        accuracy_cv10 = evaluate_cox_features(X, y, top_10_features, 10)
    results[method_name]['cv10'] = (accuracy_cv10, top_10_features)
    
    # For cv9: Remove the highest feature from full set and re-select top 9
    highest_feature = top_10_features[0]
    print(f"\nRemoving highest feature: {highest_feature}")
    reduced_X = X.drop(highest_feature, axis=1)
    
    # Re-select top 9 features from reduced dataset
    print(f"Re-selecting top 9 features from reduced dataset (shape: {reduced_X.shape})")
    top_9_features, _ = selection_function(reduced_X, y, 9)
    print(f"Top 9 features from reduced set: {top_9_features}")
    
    if method_name != 'Cox Model':
        accuracy_cv9 = evaluate_features(reduced_X, y, top_9_features, 9)
    else:
        accuracy_cv9 = evaluate_cox_features(reduced_X, y, top_9_features, 9)
    results[method_name]['cv9'] = (accuracy_cv9, top_9_features)
    
    # For cv8 and cv5: Select from full dataset
    if method_name != 'Cox Model':
        accuracy_cv8 = evaluate_features(X, y, top_10_features, 8)
        accuracy_cv5 = evaluate_features(X, y, top_10_features, 5)
    else:
        accuracy_cv8 = evaluate_cox_features(X, y, top_10_features, 8)
        accuracy_cv5 = evaluate_cox_features(X, y, top_10_features, 5)    
    
    results[method_name]['cv8'] = (accuracy_cv8, top_10_features[:8])
    results[method_name]['cv5'] = (accuracy_cv5, top_10_features[:5])
    
    print(f"Performance summary for {method_name}:")
    print(f"CV5: {accuracy_cv5:.4f}, CV8: {accuracy_cv8:.4f}, CV9: {accuracy_cv9:.4f}, CV10: {accuracy_cv10:.4f}")

# Create summary table
summary_data = []
for method_name in selection_methods:
    row = {
        'Method': method_name
    }
    
    for k in feature_counts:
        accuracy, features = results[method_name][f'cv{k}']
        feature_list = ', '.join(features[:k])
        row[f'cv{k}'] = f"{accuracy:.4f}, {feature_list}"
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Reorder columns to match desired format: Method, cv5, cv8, cv9, cv10
summary_df = summary_df[['Method', 'cv5', 'cv8', 'cv9', 'cv10']]

# Display the summary table
print("\nSummary Table:")
print(summary_df)

# Save the table
summary_df.to_csv('result.csv', index=False)
