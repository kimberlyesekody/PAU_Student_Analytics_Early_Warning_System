import os
import json
from flask import Flask, render_template, jsonify, request, Response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report

app = Flask(__name__)
DATA_PATH = "large_student_dataset.csv"

FEATURES = ['Previous_GPA', 'Attendance_Rate', 'CA_Score', 'Exam_Score', 
            'Weekly_Study_Hours', 'Assignments_Missed', 'LMS_Logins_Per_Week', 'Participation_Score']

def train_framework():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df['Risk_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Selected Model Core
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]
    
    # Basic Metrics
    metrics_log = {
        'accuracy': round(accuracy_score(y_test, preds) * 100, 2),
        'precision': round(precision_score(y_test, preds) * 100, 2),
        'recall': round(recall_score(y_test, preds) * 100, 2),
        'f1_score': round(f1_score(y_test, preds) * 100, 2)
    }
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cm = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    
    # ROC Curve Data Computation
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    # Downsample ROC points for efficient Chart.js transmission (max 20 points)
    step = max(1, len(fpr) // 20)
    roc_data = [{'x': round(float(f), 3), 'y': round(float(t), 3)} for f, t in zip(fpr[::step], tpr[::step])]
    roc_data.append({'x': 1.0, 'y': 1.0}) # Ensure endpoint binding
    
    # Feature Importances
    weights = {FEATURES[i]: round(float(rf.feature_importances_[i]) * 100, 2) for i in range(len(FEATURES))}
    
    # Generate Full Classification Report Data Stream
    report_dict = classification_report(y_test, preds, output_dict=True)
    
    return metrics_log, weights, cm, roc_data, round(roc_auc, 3), report_dict, rf

metrics, feature_weights, cm, roc_points, auc_score, class_report, rf_model = train_framework()

@app.route('/', methods=['GET', 'POST'])
def index():
    df = pd.read_csv(DATA_PATH)
    single_pred = None
    
    if request.method == 'POST':
        try:
            vector = np.array([[
                float(request.form.get('gpa', 2.5)), float(request.form.get('attendance', 80)),
                float(request.form.get('ca', 60)), float(request.form.get('exam', 60)),
                float(request.form.get('study_hours', 15)), float(request.form.get('missed_assign', 0)),
                float(request.form.get('lms_logins', 10)), float(request.form.get('participation', 70))
            ]])
            pred = int(rf_model.predict(vector)[0])
            prob = round(rf_model.predict_proba(vector)[0][1] * 100, 2)
            single_pred = {'name': request.form.get('name'), 'status': pred, 'probability': prob}
        except Exception as e:
            single_pred = {'error': str(e)}

    return render_template('dashboard.html', total=len(df), at_risk=int(df['Risk_Status'].sum()), safe=len(df)-int(df['Risk_Status'].sum()), weights=feature_weights, prediction=single_pred)

@app.route('/model-performance')
def model_performance():
    return render_template('model_perf.html', metrics=metrics, cm=cm, roc=roc_points, auc=auc_score, report=class_report)

@app.route('/batch-predict', methods=['GET', 'POST'])
def batch_predict():
    table_headers = []
    table_rows = []
    csv_raw_data = ""
    
    if request.method == 'POST':
        file = request.files.get('batch_file')
        if file and file.filename.endswith('.csv'):
            try:
                uploaded_df = pd.read_csv(file)
                table_headers = list(uploaded_df.columns)
                
                if all(col in table_headers for col in FEATURES):
                    preds = rf_model.predict(uploaded_df[FEATURES])
                    probs = rf_model.predict_proba(uploaded_df[FEATURES])[:, 1] * 100
                    
                    uploaded_df['Inference_Probability_%'] = np.round(probs, 2)
                    uploaded_df['Prediction_Status'] = ['AT RISK' if p == 1 else 'SAFE' for p in preds]
                    
                    table_headers.extend(['Inference_Probability_%', 'Prediction_Status'])
                    csv_raw_data = uploaded_df.to_csv(index=False)
                
                table_rows = uploaded_df.values.tolist()
            except Exception as e:
                table_headers = ["Error Log"]
                table_rows = [[str(e)]]

    return render_template('batch.html', headers=table_headers, rows=table_rows, csv_data=csv_raw_data)

@app.route('/download-csv', methods=['POST'])
def download_csv():
    raw_data = request.form.get('csv_content', '')
    return Response(
        raw_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=batch_predictions_output.csv"}
    )

@app.route('/eda-analytics')
def eda_analytics():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Distribution Histogram Data
    att_hist, att_edges = np.histogram(df['Attendance_Rate'], bins=10)
    gpa_hist, gpa_edges = np.histogram(df['Previous_GPA'], bins=10)
    
    # 2. Score Distributions separated by Risk Group (Averages)
    risk_df = df[df['Risk_Status'] == 1]
    safe_df = df[df['Risk_Status'] == 0]
    
    score_groups = {
        'labels': ['Exam Avg', 'CA Avg', 'Participation Avg'],
        'risk_scores': [round(risk_df['Exam_Score'].mean(), 1), round(risk_df['CA_Score'].mean(), 1), round(risk_df['Participation_Score'].mean(), 1)],
        'safe_scores': [round(safe_df['Exam_Score'].mean(), 1), round(safe_df['CA_Score'].mean(), 1), round(safe_df['Participation_Score'].mean(), 1)]
    }
    
    # 3. Correlation Heatmap Grid Ingestion (GPA, Attendance, CA, Exam, Risk)
    corr_cols = ['Previous_GPA', 'Attendance_Rate', 'CA_Score', 'Exam_Score', 'Risk_Status']
    corr_matrix = df[corr_cols].corr().round(2).values.tolist()
    
    eda_plots = {
        'att_hist': att_hist.tolist(), 'att_bins': [round(x,1) for x in att_edges[:-1]],
        'gpa_hist': gpa_hist.tolist(), 'gpa_bins': [round(x,1) for x in gpa_edges[:-1]],
        'score_groups': score_groups,
        'corr_matrix': corr_matrix, 'corr_labels': corr_cols
    }
    return render_template('eda.html', eda=eda_plots)

if __name__ == '__main__':
    app.run(debug=True, port=5000)