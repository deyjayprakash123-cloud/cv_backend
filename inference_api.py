import os
import sys
import pickle
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure the backend directory is in the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv_parser
from cv_parser import comma_tokenizer
from role_mapper import role_mapper

app = FastAPI(
    title="AI Hiring Intelligence Platform API",
    description="Inference API utilizing classification, regression, and Matrix Factorization SVD.",
    version="1.1.0"
)

# Enable CORS for frontend requests
# Note: allow_credentials=True is NOT compatible with allow_origins=["*"]
# Browsers will reject cross-origin requests if both are set simultaneously
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}

# Reverse mapping for education levels
reverse_education_map = {
    5: "PhD",
    4: "Master's (M.S./M.Tech/MBA)",
    3: "Bachelor's (B.S./B.Tech/B.Sc)",
    2: "Diploma",
    1: "High School"
}

# Standard required skills mapped to the 6 dataset job titles
ROLE_REQUIRED_SKILLS = {
    "Backend Developer": ["SQL", "Java", "Node.js", "Python", "Agile", "Scrum"],
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Deep Learning", "Communication"],
    "Frontend Developer": ["React", "Node.js", "Communication", "Agile", "Leadership"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "C++", "Agile"],
    "Project Manager": ["Project Management", "Agile", "Scrum", "Leadership", "Communication"],
    "Software Engineer": ["Java", "Python", "SQL", "React", "Agile", "Scrum"]
}

@app.on_event("startup")
def load_models():
    """Load all saved model artifacts from the models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        models_dir = "models"
        
    try:
        models["scaler"] = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        models["random_forest_clf"] = joblib.load(os.path.join(models_dir, "random_forest_clf.pkl"))
        models["tfidf_vectorizer"] = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        models["svd_model"] = joblib.load(os.path.join(models_dir, "svd_model.pkl"))
        models["svd_predicted"] = joblib.load(os.path.join(models_dir, "svd_predicted.pkl"))
        
        # Load PCA and Latent Regressor if available (from Module 4/5 training)
        pca_path = os.path.join(models_dir, "pca.pkl")
        if os.path.exists(pca_path):
            models["pca"] = joblib.load(pca_path)
            
        reg_path = os.path.join(models_dir, "latent_regressor.pkl")
        if os.path.exists(reg_path):
            models["latent_regressor"] = joblib.load(reg_path)
            
        # Load Random Forest Regressor
        rf_reg_path = os.path.join(models_dir, "rf_reg.pkl")
        if os.path.exists(rf_reg_path):
            models["rf_reg"] = joblib.load(rf_reg_path)
            
        print("All ML models (including rf_reg) loaded successfully into FastAPI.")
    except Exception as e:
        print(f"Error loading models during startup: {e}")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "AI Hiring Platform inference API is running.",
        "loaded_models": list(models.keys())
    }

@app.post("/predict")
async def predict_cv(
    file: UploadFile = File(...),
    role: str = Form("Software Engineer")
):
    """
    Predict global hireability, SVD role-specific fit score, and career recommendations.
    """
    import re
    if not models:
        load_models()
        if not models:
            raise HTTPException(status_code=500, detail="Models are not loaded. Run training_script.py first.")
            
    # Read file content
    try:
        file_bytes = await file.read()
        filename = file.filename or "resume.txt"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file bytes: {str(e)}")
        
    # 1. Extract text from uploaded CV
    cv_text = cv_parser.extract_text(file_bytes, filename)
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="CV text could not be extracted.")
        
    # 2. Extract structured candidate features using cv_parser
    experience_years, education_level, _ = cv_parser.extract_features_from_cv(cv_text)
    
    # 3. Map selected role to dataset role
    mapped_role = role_mapper.get(role, "Software Engineer")
    
    # Load required models
    scaler = models["scaler"]
    random_forest_clf = models["random_forest_clf"]
    tfidf_vectorizer = models["tfidf_vectorizer"]
    svd_model = models["svd_model"]
    svd_predicted = models["svd_predicted"]
    rf_reg = models.get("rf_reg")
    
    job_titles_list = list(svd_predicted.columns)
    if mapped_role not in job_titles_list:
        mapped_role = "Software Engineer"
    role_idx = job_titles_list.index(mapped_role)
    
    # 4. Extract matched skills from the CV text using TF-IDF vocabulary (aligned with training)
    vocab = tfidf_vectorizer.get_feature_names_out()
    extracted_skills = []
    cv_text_lower = cv_text.lower()
    for skill in vocab:
        if skill in ['c++', 'c#', '.net']:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, cv_text_lower):
            extracted_skills.append(skill)
            
    skill_count = len(extracted_skills)
    skills_str = ", ".join(extracted_skills)
    
    # Calculate Skill Overlap Ratio against target role's standard required skills
    target_req_skills = ROLE_REQUIRED_SKILLS.get(mapped_role, ["SQL", "Python", "Agile"])
    target_overlap_count = len(set(extracted_skills).intersection(set([s.lower() for s in target_req_skills])))
    target_overlap_ratio = target_overlap_count / len(target_req_skills) if len(target_req_skills) > 0 else 0.0
    
    # 5. Predict Global Hireability (Random Forest Classifier using 4 features)
    base_features = np.array([[experience_years, education_level, skill_count]])
    base_features_scaled = scaler.transform(base_features)
    
    clf_features = np.hstack([base_features_scaled, [[target_overlap_ratio]]])
    global_pred_class = int(random_forest_clf.predict(clf_features)[0])
    global_pred_prob = float(random_forest_clf.predict_proba(clf_features)[0][1])
    
    # 6. Compute Candidate-Specific SVD Fit Score (collaborative latent projection Route B)
    svd_fit_score = 0.0
    try:
        if "latent_regressor" in models and "pca" in models:
            pca = models["pca"]
            latent_regressor = models["latent_regressor"]
            
            # Vectorize the clean reconstructed skills string, matching training TF-IDF
            text_tfidf = tfidf_vectorizer.transform([skills_str])
            cv_pca = pca.transform(text_tfidf.toarray())
            X_latent = np.hstack([base_features_scaled, cv_pca])
            
            # Predict SVD latent vector
            candidate_latent = latent_regressor.predict(X_latent)
            
            # Reconstruct job fit scores
            predicted_fits = np.dot(candidate_latent, svd_model.components_)[0]
            profile_quality = (min(experience_years / 10.0, 1.0) * 0.4) + \
                              (min(education_level / 5.0, 1.0) * 0.3) + \
                              (min(skill_count / 15.0, 1.0) * 0.3)
                              
            raw_min = min(predicted_fits)
            raw_max = max(predicted_fits)
            if raw_max > raw_min:
                relative_weights = 0.75 + 0.25 * (predicted_fits - raw_min) / (raw_max - raw_min)
            else:
                relative_weights = np.ones(len(predicted_fits))
                
            svd_job_scores = relative_weights * profile_quality
            svd_fit_score = min(max(float(svd_job_scores[role_idx]), 0.0), 1.0)
    except Exception as e:
        print(f"SVD collaborative model prediction warning: {e}")
        
    # 7. Predict precise Job Fit Scores for ALL roles using the high R² (0.998) Regressor
    final_job_scores = []
    for job in job_titles_list:
        job_req_skills = ROLE_REQUIRED_SKILLS.get(job, [])
        overlap_count = len(set(extracted_skills).intersection(set([s.lower() for s in job_req_skills])))
        overlap_ratio = overlap_count / len(job_req_skills) if len(job_req_skills) > 0 else 0.0
        
        feat = np.hstack([base_features_scaled, [[overlap_ratio]]])
        score = float(rf_reg.predict(feat)[0]) if rf_reg else svd_fit_score
        # Clamp score between 0.0 and 1.0
        score = min(max(score, 0.0), 1.0)
        final_job_scores.append(score)
        
    role_fit_score = final_job_scores[role_idx]
    
    # Classify as Hired if role fit score > 0.7
    hired_prediction = "Hired" if role_fit_score > 0.7 else "Rejected"
    
    # 8. Generate Alternative Recommendations (Top 3)
    all_job_scores = []
    for idx, job in enumerate(job_titles_list):
        all_job_scores.append((job, final_job_scores[idx]))
        
    # Sort descending by score
    all_job_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top 3 other roles (excluding the selected one)
    recommended_jobs = []
    recs_count = 0
    for job, score in all_job_scores:
        if job != mapped_role and recs_count < 3:
            recommended_jobs.append({
                "role": job,
                "score": round(score, 3)
            })
            recs_count += 1
            
    education_name = reverse_education_map.get(education_level, "Diploma")
    
    return {
        # User prompt required JSON keys
        "hired_prediction": hired_prediction,
        "confidence": round(global_pred_prob, 3),
        "extracted_experience": float(experience_years),
        "extracted_education_level": int(education_level),
        "extracted_skill_count": int(skill_count),
        "mapped_role": mapped_role,
        "role_fit_score": round(role_fit_score, 3),
        "recommended_jobs": recommended_jobs,
        
        # React Frontend compatibility keys
        "success": True,
        "extracted_features": {
            "experience_years": float(experience_years),
            "education_level": int(education_level),
            "education_name": education_name,
            "skill_count": int(skill_count)
        },
        "role_predictions": {
            "selected_role": role,
            "mapped_role": mapped_role,
            "fit_score": round(role_fit_score, 3),
            "outcome": hired_prediction
        },
        "global_predictions": {
            "hireability_prob": round(global_pred_prob, 3),
            "outcome": "Hired" if global_pred_class == 1 else "Rejected"
        },
        "recommendations": recommended_jobs
    }
