<<<<<<< HEAD
# thyroid-cancer-project
Differentiated Thyroid Cancer Recurrence EDA Project
=======
# Differentiated Thyroid Cancer Recurrence EDA Project

This project aims to analyze the Differentiated Thyroid Cancer Recurrence dataset using Exploratory Data Analysis (EDA) and build predictive models to classify whether cancer is likely to recur.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Details](#dataset-details)
- [Installation](#installation)
- [Usage](#usage)
- [Features Explored](#features-explored)
- [Modeling](#modeling)
- [Insights](#insights)
- [Author](#author)

---

## Overview
This project employs **Streamlit** for interactive visualization, using Pandas for data analysis and Plotly for visualizations. Machine learning models are developed and evaluated using cross-validation techniques with recall as the primary metric.

---

## Dataset Details
- **Name**: [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)
- **Duration**: Data collected over 15 years; each patient followed for at least 10 years.
- **Features**: 13 clinical attributes and one target variable (`Recurred`).
- **Objective**: To predict the likelihood of thyroid cancer recurrence based on clinical data.

---

## Installation
To run this project, ensure you have Python installed and set up the environment:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows

3. Clone the repository:
   ```bash
   pip install -r requirements.txt

---

## Usage
Run the Streamlit application:
   ```bash
   pip install -r requirements.txt

---

## Application Features
1. EDA: Visualize data distribution, correlation, and other trends.
2. Modeling: Compare machine learning models and evaluate their performance.
3. Interactive Plots: Explore and analyze features dynamically.

---

## Features Explored
1. Age Distribution:
   - Recurrence peaks in middle-aged adults (30–40 years).
   - Preventative measures may help reduce risks in these critical age groups.
2. Gender Insights:
   - Age distribution patterns are similar across genders, but recurrence is slightly more common in older patients.
3. Correlation Analysis:
   - Strong correlations between Stage, Tumor Size (T), Node Involvement (N), and recurrence.
   - Demographic factors like Gender and Smoking show minimal correlation.

---

## Modeling

### Models Evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors
6. Naive Bayes
7. Linear Discriminant Analysis

### Evaluation Criteria:
   - Cross-validation ensures reliable model performance across splits.
   - Recall is used to prioritize minimizing false negatives.

### Best Models:
1. Random Forest:
   - Best Parameters: {max_depth=3, max_features=15, n_estimators=25}
   - Performance:
      - AUC: 0.92
2. SVM:
   - Best Parameters: {C=1000, gamma=0.0001, kernel='rbf'}
   - Performance:
      - AUC: 0.89

## Insights
   - Age Group Analysis:
      - Middle-aged adults are at higher risk of recurrence.
      - Cancer recurrence drops significantly in patients above 70 years.
   - Key Features:
      - Tumor size (T), node involvement (N), and stage have the strongest correlations with recurrence.
   - Model Performance:
      - Random Forest and SVM provide stable and reliable results, making them suitable for predicting recurrence.
## Author
   João Victor Aragão
   - More Projects: [joaoaragao](https://joaovictor.onrender.com/)
   - GitHub: [joaovictor-aragao](https://github.com/joaovictor-aragao)
>>>>>>> ae0c0a2 (Add files)
