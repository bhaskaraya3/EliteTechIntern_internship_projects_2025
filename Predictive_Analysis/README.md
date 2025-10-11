# 🚀  Project Overview
This project demonstrates the end-to-end process of building, training, and evaluating a machine learning model to predict outcomes (classification task) from a large-scale dataset.

The dataset consisted of millions of text entries, requiring scalable preprocessing and incremental learning techniques. The goal was to create a robust and reproducible ML pipeline that can handle high-volume data while still providing accurate predictions.

This project was built as part of my internship at EliteTech, where I applied practical ML techniques to solve a real-world classification problem.

# 🛠️  What I Did
1. Data Preprocessing
Cleaned and normalized raw text (removing punctuation, stopwords, and lowercasing).
Converted text into numerical features using:
    TF-IDF Vectorization (baseline model).
    HashingVectorizer (for scalable incremental learning).
Applied dimensionality reduction (TruncatedSVD) for feature selection.

2. Model Training
Built multiple models:
Baseline Logistic Regression (with TF-IDF).
Streaming SGDClassifier (trained incrementally on 3.1M rows).
Balanced Class-Weighted Model to handle severe class imbalance.
Used partial_fit() to train on streaming data without overloading memory.

3. Evaluation
Assessed performance using:
Classification reports (precision, recall, F1-score).
Confusion matrices.
Accuracy comparisons on balanced test sets vs original imbalanced sets.
Analyzed failure cases (esp. poor recall on minority classes).

4. Model Persistence

Trained models and pipelines were saved using joblib:
baseline_model.joblib
streaming_model.joblib
balanced_streaming_model.joblib

5. Reproducibility
All code is in a Jupyter Notebook with step-by-step explanations.
Exported a standalone Python script (predictive_analysis.py) for quick re-runs.

# Practical uses of the model outputs:

Business insights → identifying underrepresented categories, trends, or anomalies.
Automation → real-time classification of streaming data (e.g., customer support, social media monitoring).
Decision support → assisting humans by pre-labeling massive datasets.

# Results:
Baseline and streaming models provide robust text classification on large datasets.
Streaming SGDClassifier achieves ~75% accuracy on balanced test sets.
Handles large-scale datasets efficiently without exhausting memory.
Outputs predictions ready for use in dashboards, applications, or further analysis.

# Recommendations:
For production use, focus on binary classification (main classes 0 vs 1).
For rare classes (like 2), consider collecting more data.
Models can be used as APIs for text classification applications.
Update preprocessing if using different datasets (column names, stopwords, etc.).

# Author
Name: Bhaskar Arya
LinkedIn : www.linkedin.com/in/bhaskar-arya-b854b9245