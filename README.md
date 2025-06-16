# Intelligent-document-analysis-system
Intelligent Document Analysis System is an ML-powered solution with the ability to categorize and analyze the sentiment of business documents automatically. It mimics an actual ML project optimized for OpenText, a world leader in Enterprise Information Management (EIM). It facilitates document classification into pre-configured categories—like Legal, Financial, Technical, Marketing, and HR—and sentiment analysis of the document content.
🎯 Objective
To build a production-ready document analysis tool that:

Automatically classifies documents based on their textual content.

Performs sentiment analysis (positive, neutral, negative).

Provides visual insights and comparative performance metrics.

Demonstrates scalability and integration-readiness for enterprise environments like OpenText's ECM suite.

⚙️ Key Components
Text Preprocessing:

Lowercasing, special character removal, and stop-word filtering.

Feature Engineering:

TF-IDF vectorization for converting text into numerical features.

Classification Models:

Trains and compares multiple models:

Multinomial Naive Bayes

Random Forest

Support Vector Machine

Logistic Regression

Selects the best-performing model for deployment.

Sentiment Analysis:

Separate logistic regression model trained to identify document sentiment.

Evaluation Metrics:

Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Cross-Validation.

Visual Dashboard:

Bar charts, heatmaps, and comparison graphs to interpret model performance.

Real-Time Prediction Interface:

Accepts new document input and outputs category & sentiment predictions with confidence scores.

