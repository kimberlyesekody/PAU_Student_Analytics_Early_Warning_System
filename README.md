# PAU Student Academic Performance Predictive Early Warning Platform (V2)

An enterprise-grade, lightweight predictive analytics microservice designed to collapse departmental data latency barriers and move academic advising from a retrospective exercise to an automated, data-driven framework.

## 🛠️ System Architecture & Feature Space

The platform transitions away from low-dimensional baselines by training an ensemble **Random Forest Classifier** across **8 distinct academic and behavioral feature vectors**:

- Previous GPA, Attendance Rate, Cumulative CA Marks, Summative Exam Scores, Weekly Study Hours, Missed Assignment Counts, LMS Logins, and Class Participation Indicies.

## 🚀 Local Installation Framework

1. Clone the repository workspace:
   ```bash
   git clone [https://github.com/kimberlyesekody/PAU_Student_Analytics_Early_Warning_System.git](https://github.com/kimberlyesekody/PAU_Student_Analytics_Early_Warning_System.git)
   cd PAU_Student_Analytics_Early_Warning_System
   ```
   Build the training matrix data stream:

Bash
python synthetic_data.py

Boot the production server:

Bash
python app.py

---
