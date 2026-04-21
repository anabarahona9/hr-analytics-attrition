HR Analytics: Predictive Model for Employee Attrition

English
Overview
End-to-end machine learning project to predict employee attrition using the IBM HR Analytics Dataset. The project combines exploratory data analysis, class imbalance handling with SMOTE, and a Random Forest classification model to identify employees at risk of leaving the company.
🔗 Interactive Dashboard
👉 View on Tableau Public
Key Findings

16.1% overall attrition rate across 1,470 employees
Sales department has the highest attrition rate at 20.63%
Employees with overtime are significantly more likely to leave
Stock Option Level, Monthly Income and Job Satisfaction are the top 3 attrition drivers
Employees who leave have a considerably lower average monthly income

Model Performance
ModelAccuracyRecall (Attrition=Yes)Random Forest88%0.13Random Forest + SMOTE85%0.26
SMOTE was applied to handle class imbalance (84% No / 16% Yes). The balanced model doubles recall for at-risk employees — more valuable for HR decision making than raw accuracy.
Tech Stack
ToolPurposePython / PandasData cleaning and EDAScikit-learnRandom Forest modelimbalanced-learnSMOTE for class balancingSeaborn / MatplotlibEDA visualizationsTableau PublicInteractive dashboard
Project Structure
hr-analytics-attrition/
├── churn_analysis.py        # Full pipeline: EDA, model, export
├── dashboard_preview.png    # Dashboard screenshot
└── README.md
Workflow
Raw CSV → EDA → Label Encoding → Train/Test Split
→ Random Forest → SMOTE → Random Forest + SMOTE
→ Feature Importance → Export → Tableau Dashboard
Dataset

Source: IBM HR Analytics — Kaggle
Observations: 1,470 employees
Features: 35 variables (31 after cleaning)
Target: Attrition (Yes/No)


Español
Descripción
Proyecto de machine learning end-to-end para predecir la deserción de empleados usando el IBM HR Analytics Dataset. Combina análisis exploratorio, manejo de desbalance de clases con SMOTE y un modelo de clasificación Random Forest para identificar empleados en riesgo de abandonar la empresa.
🔗 Dashboard Interactivo
👉 Ver en Tableau Public
Hallazgos Principales

Tasa de deserción general del 16.1% en 1,470 empleados
El departamento de Sales tiene la mayor tasa con 20.63%
Empleados con horas extra tienen significativamente más probabilidad de irse
Stock Option Level, Ingreso Mensual y Satisfacción Laboral son los 3 principales factores de deserción
Los empleados que se van tienen un ingreso mensual promedio considerablemente menor

Desempeño del Modelo
ModeloAccuracyRecall (Deserción=Sí)Random Forest88%0.13Random Forest + SMOTE85%0.26
Se aplicó SMOTE para manejar el desbalance de clases (84% No / 16% Sí). El modelo balanceado duplica el recall para empleados en riesgo — más valioso para decisiones de RRHH que la precisión bruta.
Stack Tecnológico
HerramientaUsoPython / PandasLimpieza de datos y EDAScikit-learnModelo Random Forestimbalanced-learnSMOTE para balanceo de clasesSeaborn / MatplotlibVisualizaciones EDATableau PublicDashboard interactivo
