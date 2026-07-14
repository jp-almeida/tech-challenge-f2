# Model Card: E-Commerce Recommender

## Model Details
- **Architecture:** Multi-Layer Perceptron (MLP)
- **Framework:** PyTorch
- **Task:** Binary Classification (Purchase Prediction based on interactions)
- **Design Pattern:** Factory Method pattern allows dynamically swapping to Scikit-Learn baselines.

## Intended Use
- **Primary Use Case:** Recommend e-commerce products based on user browsing behavior and normalized interaction times.

## Performance
- **Metrics Evaluated:** 
  - RMSE (Root Mean Squared Error)
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - ROC-AUC
- **Tracking:** A Random Forest baseline and two MLP configurations are tracked in MLflow.
- **Registry:** The best MLP by F1 score is registered as `EcommerceRecommender` and promoted to Production.

## Limitations & Biases
- **Cold Start Problem:** The model might struggle recommending items to completely new users who have zero interaction history.
- **Popularity Bias:** Highly interacted items may skew the model weights, leading to mostly popular recommendations over personalized niches.
