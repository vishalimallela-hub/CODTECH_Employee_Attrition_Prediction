# Employee Attrition Prediction

## Project Overview
This project uses machine learning to predict employee attrition based on employee-related features. It includes a Streamlit web application where users can enter employee details, view predicted attrition probabilities, and explore patterns in the dataset.

Developed as part of the CODTECH Data Science Internship.

## Features
- Machine learning model for employee attrition prediction.
- Model performance metrics displayed in the application.
- Predicted probability breakdown for attrition and no attrition.
- Interactive analytics by department, overtime, and job role.
- What-if simulator comparing predictions with and without overtime.
- Visualizations for attrition distribution and model performance.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Dataset
The project uses the IBM HR Analytics Employee Attrition dataset, stored in the `dataset/` directory.

## Project Structure
```text
Employee_Attrition_Prediction/
├── app.py
├── Employee_Attrition_Prediction.ipynb
├── attrition_model.joblib
├── model_results.json
├── dataset/
├── visualizations/
├── screenshots/
└── README_new.md
```

## Model Performance
- Test accuracy: 75.2%
- ROC-AUC: 0.803
- Test records: 294

These results describe performance on a held-out test set and do not guarantee correct predictions for individual employees.

## Run the Application
1. Install Python.
2. Install the required packages:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```
3. Open a terminal in the project directory and run:
   ```bash
   streamlit run app.py
   ```

## Screenshots
Screenshots of the prediction form, prediction results, interactive analytics, and What-if Simulator are available in the `screenshots/` folder.

## Disclaimer
This project is for educational and demonstration purposes only. Predictions are uncertain statistical estimates and should not be used as the sole basis for employment decisions.
