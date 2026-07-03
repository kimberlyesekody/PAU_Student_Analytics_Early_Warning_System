import pandas as pd
import numpy as np

def generate_expanded_dataset(filename="large_student_dataset.csv", num_records=1500):
    np.random.seed(42)
    
    # Generate an expanded 8-feature data matrix
    gpa = np.random.uniform(1.5, 4.0, num_records)
    attendance = np.random.uniform(50, 100, num_records)
    ca_score = np.random.uniform(30, 95, num_records)
    exam_score = np.random.uniform(35, 98, num_records)
    study_hours = np.random.uniform(2, 30, num_records)
    missed_assign = np.random.randint(0, 7, num_records)
    lms_logins = np.random.uniform(1, 45, num_records)
    participation = np.random.uniform(20, 100, num_records)
    
    # Complex, non-linear risk calculation using the expanded feature space
    base_risk = ((4.0 - gpa) * 20 + (100 - attendance) * 0.25 + 
                 (100 - exam_score) * 0.2 + (missed_assign * 8) - 
                 (study_hours * 0.4) - (lms_logins * 0.15))
    
    noise = np.random.normal(0, 6, num_records)
    final_score = base_risk + noise
    risk_status = (final_score > 32).astype(int)
    
    df = pd.DataFrame({
        'Previous_GPA': np.round(gpa, 2),
        'Attendance_Rate': np.round(attendance, 2),
        'CA_Score': np.round(ca_score, 2),
        'Exam_Score': np.round(exam_score, 2),
        'Weekly_Study_Hours': np.round(study_hours, 2),
        'Assignments_Missed': missed_assign,
        'LMS_Logins_Per_Week': np.round(lms_logins, 2),
        'Participation_Score': np.round(participation, 2),
        'Risk_Status': risk_status
    })
    
    df.to_csv(filename, index=False)
    print(f"Success! Generated expanded 8-feature dataset inside '{filename}'.")

if __name__ == "__main__":
    generate_expanded_dataset()