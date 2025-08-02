
# 🌾 Crop Recommendation System using Machine Learning

This project implements and compares multiple **classification algorithms** to recommend the **most suitable crop** for cultivation based on soil and environmental parameters using a publicly available dataset. The models are trained, optimized, and evaluated using Python and popular libraries such as `scikit-learn`, `pandas`, `numpy`, and more.

---

## 📁 Dataset

The dataset used is:  
**Crop_Recommendation.csv**  
It contains **2200+ samples** with the following features:

| Feature         | Description                                 |
|-----------------|---------------------------------------------|
| Nitrogen        | Content in the soil                         |
| Phosphorus      | Content in the soil                         |
| Potassium       | Content in the soil                         |
| Temperature     | Temperature in °C                           |
| Humidity        | Humidity percentage                         |
| pH_Value        | pH of the soil                              |
| Rainfall        | Rainfall in mm                              |
| Crop (Target)   | Recommended crop label (rice, wheat, etc.)  |

---

## 🧼 Data Preprocessing

1. **Feature Scaling**: Standardized features using `StandardScaler`.
2. **Label Encoding**: Converted crop names to numeric values using `LabelEncoder`.
3. **Train-Test Split**: Data split into 80% training and 20% testing using `train_test_split`.

---

## 🤖 Models Implemented

| Model                      | Description |
|---------------------------|-------------|
| **Naive Bayes**           | Probabilistic model assuming feature independence. Very fast and accurate on this dataset. |
| **Logistic Regression**   | Linear model used for multiclass classification with softmax. |
| **Random Forest**         | Ensemble of decision trees using bagging and majority voting. |
| **Decision Tree (Entropy & Gini)** | Tree-based models using entropy and Gini index to split nodes. |
| **Support Vector Machine**| Kernel-based classifier that separates data using hyperplanes. |
| **K-Nearest Neighbors**   | Distance-based model classifying based on K closest neighbors. |
| **Voting Classifier**     | Soft-voting ensemble of Logistic Regression, Random Forest, and SVC. |

Some models were tuned using **GridSearchCV** to optimize parameters such as:
- `max_depth`, `n_estimators` (for trees and forests)
- `C`, `gamma`, `kernel` (for SVM)
- `n_neighbors` (for KNN)

---

## 📈 Evaluation Metrics

| Metric     | Description                                                                 |
|------------|-------------------------------------------------------------------------------|
| **Accuracy**   | Overall correctness of predictions (most intuitive metric).             |
| **Precision**  | Out of predicted crops, how many were actually correct.                |
| **Recall**     | Out of actual crops, how many were correctly predicted.                |
| **F1 Score**   | Harmonic mean of precision and recall (best when data is imbalanced). |

---

## 🧪 Results Summary

| Model                      | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| **Naive Bayes**           | 0.9955   | 0.9958    | 0.9955 | 0.9954   |
| Logistic Regression       | 0.9636   | 0.9644    | 0.9636 | 0.9635   |
| Random Forest             | 0.9931   | 0.9937    | 0.9931 | 0.9931   |
| Decision Tree (Entropy)   | 0.9795   | 0.9802    | 0.9795 | 0.9793   |
| Decision Tree (Gini)      | 0.9863   | 0.9868    | 0.9863 | 0.9863   |
| Support Vector Classifier | 0.9795   | 0.9820    | 0.9795 | 0.9796   |
| K-Nearest Neighbors       | 0.9681   | 0.9727    | 0.9681 | 0.9682   |
| **Voting Classifier**     | 0.9841   | 0.9857    | 0.9841 | 0.9844   |

---

## 🔍 Key Takeaways

- **Naive Bayes** outperformed all models due to the independent nature of input features and Gaussian distributions.
- **Random Forest** and **Voting Classifier** also showed excellent performance due to their robustness and ensemble nature.
- Feature scaling helped improve performance, especially for distance-based models like KNN and SVM.
- GridSearchCV tuning improved the performance of SVM and KNN notably.

---

## 📦 Libraries Used

- `pandas`, `numpy` – Data handling
- `scikit-learn` – ML models, metrics, preprocessing, hyperparameter tuning
- `xgboost` *(optional)* – For future model comparison
- `tabulate` – Pretty console table output

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/crop-recommendation-ml.git
   cd crop-recommendation-ml
