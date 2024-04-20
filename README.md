# Breast Cancer Prediction Model

## Project Overview

This repository contains the code and documentation for a machine learning project focused on predicting whether breast cancer tumors are benign or malignant. The project uses multiple machine learning algorithms processed through a Python-based data science stack.

## Getting Started

### Prerequisites

Ensure you have the following installed:

-   Python 3.8 or later
-   pip (Python package installer)

### Installation

Clone the repository and install the required Python packages:

bashCopy code

    git clone https://github.com/your-username/breast-cancer-prediction.git
    cd breast-cancer-prediction
    pip install -r requirements.txt

### Running the Application

To run the Streamlit application locally:

`streamlit run app.py` 

## Repository Structure

-   `EDA/` - Jupyter notebook (`EDA.ipynb`) containing exploratory data analysis.
-   `assets/` - Contains CSS files (`style.css`) for styling the Streamlit application.
-   `data/` - Dataset file (`data.csv`) used for model training.
-   `model/` - Python scripts and model files:
    -   `main.py` - Main script for model training and evaluation. 
-   `app.py` - Streamlit application script for the web interface.
-   `requirements.txt` - Required packages for the project.

## Models

The project evaluates several models to ensure robustness and accuracy:

-   **Logistic Regression**: Provides a good baseline for model comparison.
-   **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
-   **K-Nearest Neighbors (KNN)**: Captures the locality of data points for prediction.
-   **Random Forest**: Utilizes an ensemble method for improved prediction stability and accuracy.

Model performance is evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC metrics.

## Data Collection and Preprocessing

The dataset is sourced from the UCI Machine Learning Repository and involves several preprocessing steps:

-   Removal of irrelevant identifiers.
-   Normalization using `StandardScaler`.
-   Encoding of categorical variables.

## Feature Engineering and Model Development

Feature engineering techniques employed include:

-   **Interaction Features** to explore potential synergies between variables.
- **Experimenting with different models**

## Usage

The Streamlit application provides an interactive interface to explore the models' predictions based on user input. It demonstrates the model's ability to classify tumors and allows for dynamic adjustment of predictor values.


## Team Members (NUID):
1. Ashwini Khedkar (002738717)
2. Tanmay Zope (002767087)
3. Vipul Rajderkar (002700991)
