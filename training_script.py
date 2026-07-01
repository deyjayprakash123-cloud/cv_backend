import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, silhouette_score
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_parser import comma_tokenizer

def run_training():
    print("Loading recruitment dataset...")
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "data", "recruitment_dataset.csv")
    if not os.path.exists(csv_path):
        # Fallback to local workspace if run differently
        csv_path = "recruitment_dataset.csv"
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # ------------------ PREPROCESSING ------------------
    # Map Education to numeric values
    education_mapping = {
        'Diploma': 2,
        'B.Sc': 3,
        'B.Tech': 3,
        'B.E': 3,
        'MBA': 4,
        'M.Sc': 4,
        'M.Tech': 4,
        'PhD': 5
    }
    df['Education_Enc'] = df['Education'].map(education_mapping).fillna(3).astype(int)
    
    # Count skills from Skills column
    df['Skill_Count'] = df['Skills'].apply(
        lambda x: len([s.strip() for s in str(x).split(',') if s.strip()]) if pd.notna(x) else 0
    )
    
    # Calculate Skill Overlap Ratio in the dataset (overlap between candidate skills and job required skills)
    def get_overlap(row):
        if pd.isna(row['Skills']) or pd.isna(row['Required_Skills']):
            return 0
        s1 = set([s.strip().lower() for s in row['Skills'].split(',') if s.strip()])
        s2 = set([s.strip().lower() for s in row['Required_Skills'].split(',') if s.strip()])
        return len(s1.intersection(s2))

    df['Skill_Overlap'] = df.apply(get_overlap, axis=1)
    df['Skill_Overlap_Ratio'] = df.apply(
        lambda r: r['Skill_Overlap'] / len([s.strip() for s in str(r['Required_Skills']).split(',') if s.strip()]) 
        if pd.notna(r['Required_Skills']) and len([s.strip() for s in str(r['Required_Skills']).split(',') if s.strip()]) > 0 
        else 0, 
        axis=1
    )

    # Convert Outcome to binary label (1 for Hired, 0 for Not Hired/Rejected/Shortlisted)
    df['Outcome_Binary'] = (df['Outcome'] == 'Hired').astype(int)
    
    # Base Features (Scale only candidate-specific numeric variables)
    X_base = df[['Experience_Years', 'Education_Enc', 'Skill_Count']]
    y_clf = df['Outcome_Binary']
    y_reg = df['Job_Fit_Score']
    
    # Scale Features
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    
    # Stack the Skill_Overlap_Ratio as an unscaled 4th feature (since it is already bounded in [0, 1])
    X_features = np.hstack([X_base_scaled, df[['Skill_Overlap_Ratio']].values])
    
    # ------------------ MODULE 1 (CLASSIFICATION) ------------------
    print("\n--- Module 1: Classification (Predict Hired/Not-Hired) ---")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_features, y_clf, test_size=0.2, random_state=42
    )
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_c, y_train_c)
    knn_preds = knn.predict(X_test_c)
    print(f"KNN Accuracy: {accuracy_score(y_test_c, knn_preds):.4f}")
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_c, y_train_c)
    dt_preds = dt.predict(X_test_c)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test_c, dt_preds):.4f}")
    
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_clf.fit(X_features, y_clf)  # Fit on full dataset for production usage
    
    # Validation evaluation for outputting metrics
    rf_val = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_val.fit(X_train_c, y_train_c)
    rf_val_preds = rf_val.predict(X_test_c)
    rf_val_probs = rf_val.predict_proba(X_test_c)[:, 1]
    val_acc = accuracy_score(y_test_c, rf_val_preds)
    val_auc = roc_auc_score(y_test_c, rf_val_probs)
    print(f"Random Forest Accuracy: {val_acc:.4f}")
    print(f"Random Forest ROC-AUC: {val_auc:.4f}")
    
    # ------------------ MODULE 2 (REGRESSION) ------------------
    print("\n--- Module 2: Regression (Predict Job Fit Score) ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_features, y_reg, test_size=0.2, random_state=42
    )
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_r, y_train_r)
    lr_preds = lr.predict(X_test_r)
    print(f"Linear Regression R2: {r2_score(y_test_r, lr_preds):.4f}")
    
    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_reg.fit(X_features, y_reg)  # Fit on full dataset for production usage
    
    # Validation evaluation
    rf_reg_val = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_reg_val.fit(X_train_r, y_train_r)
    rf_reg_preds = rf_reg_val.predict(X_test_r)
    val_r2 = r2_score(y_test_r, rf_reg_preds)
    print(f"Random Forest Regressor R2: {val_r2:.4f}")
    
    # ------------------ MODULE 3 (CLUSTERING) ------------------
    print("\n--- Module 3: Clustering (Talent Segmentation) ---")
    # Base features + Job Fit Score
    X_cluster = np.hstack([X_base_scaled, y_reg.values.reshape(-1, 1)])
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X_cluster)
    print(f"K-Means Silhouette Score: {silhouette_score(X_cluster, kmeans_labels):.4f}")
    
    # Hierarchical (Ward)
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
    hier_labels = hierarchical.fit_predict(X_cluster)
    print(f"Hierarchical Clustering Silhouette Score: {silhouette_score(X_cluster, hier_labels):.4f}")
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_cluster)
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN Clusters Found: {n_clusters_db}")
    
    # ------------------ MODULE 4 (DIMENSIONALITY REDUCTION) ------------------
    print("\n--- Module 4: Dimensionality Reduction (PCA on TF-IDF skills) ---")
    tfidf = TfidfVectorizer(analyzer=comma_tokenizer)
    X_tfidf = tfidf.fit_transform(df['Skills'].fillna(''))
    
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    print(f"PCA Total Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # ------------------ MODULE 5 (COLLABORATIVE FILTERING VIA SVD) ------------------
    print("\n--- Module 5: Collaborative Filtering (TruncatedSVD Matrix Factorization) ---")
    # Build candidate-job pivot matrix using pivot_table to handle duplicate entries
    pivot_df = df.pivot_table(index='Candidate_ID', columns='Job_Title', values='Job_Fit_Score', aggfunc='mean').fillna(0)
    job_titles_list = list(pivot_df.columns)
    print(f"Candidate-Job Matrix shape: {pivot_df.shape} (Jobs: {job_titles_list})")
    
    # Run TruncatedSVD
    n_svd_components = min(5, len(job_titles_list) - 1)
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    candidate_latent = svd.fit_transform(pivot_df)
    print(f"SVD Total Explained Variance Ratio: {np.sum(svd.explained_variance_ratio_):.4f}")
    
    # Aggregate content features per unique candidate, matching the index order of pivot_df
    print("Aggregating candidate features for latent SVD mapping...")
    candidate_features = []
    for cand_id in pivot_df.index:
        cand_rows = df[df['Candidate_ID'] == cand_id]
        
        # Aggregate base variables
        exp = cand_rows['Experience_Years'].max()
        edu = cand_rows['Education_Enc'].max()
        
        # Aggregate unique skills
        all_skills = []
        for s_list in cand_rows['Skills'].dropna():
            for s in str(s_list).split(','):
                s_clean = s.strip()
                if s_clean and s_clean not in all_skills:
                    all_skills.append(s_clean)
        skills_str = ", ".join(all_skills)
        skill_cnt = len(all_skills)
        
        candidate_features.append({
            'Experience_Years': exp,
            'Education_Enc': edu,
            'Skill_Count': skill_cnt,
            'Skills': skills_str
        })
        
    cand_feat_df = pd.DataFrame(candidate_features)
    
    # Preprocess and scale aggregated features
    X_cand_base = scaler.transform(cand_feat_df[['Experience_Years', 'Education_Enc', 'Skill_Count']])
    
    # Vectorize and reduce aggregated skills
    X_cand_tfidf = tfidf.transform(cand_feat_df['Skills'])
    X_cand_pca = pca.transform(X_cand_tfidf.toarray())
    
    # Combine for latent regressor input
    X_cand_base_pca = np.hstack([X_cand_base, X_cand_pca])
    
    # Train regressor to predict the SVD latent factors of any candidate
    latent_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    latent_regressor.fit(X_cand_base_pca, candidate_latent)
    print("Cold-start Latent Regressor trained successfully.")
    
    # ------------------ SAVE MODELS ------------------
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save Scaler
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    # Save Random Forest Classifier (both names for compatibility)
    with open(os.path.join(models_dir, "rf_clf.pkl"), "wb") as f:
        pickle.dump(rf_clf, f)
    with open(os.path.join(models_dir, "random_forest_clf.pkl"), "wb") as f:
        pickle.dump(rf_clf, f)

    # Save Random Forest Regressor
    with open(os.path.join(models_dir, "rf_reg.pkl"), "wb") as f:
        pickle.dump(rf_reg, f)
    with open(os.path.join(models_dir, "random_forest_reg.pkl"), "wb") as f:
        pickle.dump(rf_reg, f)
        
    # Save TF-IDF Vectorizer
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
        
    # Save PCA (Module 4)
    with open(os.path.join(models_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
        
    # Save Latent Regressor (Module 5 bridge)
    with open(os.path.join(models_dir, "latent_regressor.pkl"), "wb") as f:
        pickle.dump(latent_regressor, f)
        
    # Save TruncatedSVD Model (directly)
    with open(os.path.join(models_dir, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)
        
    # Construct and save full SVD predicted interaction matrix DataFrame
    predicted_matrix = np.dot(candidate_latent, svd.components_)
    svd_predicted_df = pd.DataFrame(predicted_matrix, index=pivot_df.index, columns=pivot_df.columns)
    with open(os.path.join(models_dir, "svd_predicted.pkl"), "wb") as f:
        pickle.dump(svd_predicted_df, f)
        
    # SVD Reconstruction Analysis
    svd_rmse = np.sqrt(np.mean((pivot_df.values - predicted_matrix) ** 2))
    
    # Save Metadata and Encoder lists
    with open(os.path.join(models_dir, "job_titles.pkl"), "wb") as f:
        pickle.dump(job_titles_list, f)
        
    with open(os.path.join(models_dir, "education_encoder.pkl"), "wb") as f:
        pickle.dump(education_mapping, f)
        
    print("\nAll models (including random_forest_clf.pkl, svd_model.pkl, and svd_predicted.pkl) saved successfully.")
    print(f"Collaborative SVD Reconstruction RMSE: {svd_rmse:.5f}")
    
    # PRINT THE ACTUAL ACHIEVED METRICS
    print("\n" + "="*50)
    print("Baseline Production Metrics Achieved:")
    print(f"Classification Accuracy: {val_acc*100:.1f}%")
    print(f"ROC-AUC: {val_auc:.3f}")
    print(f"Regression R²: {val_r2:.3f}")
    print("="*50)

if __name__ == "__main__":
    run_training()
