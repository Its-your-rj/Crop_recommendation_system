
# 🌾 Crop Recommendation System using Machine Learning

This project implements and compares multiple **classification algorithms** to recommend the **most suitable crop** for cultivation based on soil and environmental parameters using a publicly available dataset. The models are trained, optimized, and evaluated using Python and popular libraries such as `scikit-learn`, `pandas`, `numpy`, and more.

---

## 📁 Dataset

The dataset used is:
**Crop_Recommendation.csv**
It contains **2200+ samples** with the following features:

| Feature       | Description                                |
|---------------|--------------------------------------------|
| Nitrogen      | Content in the soil                        |
| Phosphorus    | Content in the soil                        |
| Potassium     | Content in the soil                        |
| Temperature   | Temperature in °C                        |
| Humidity      | Humidity percentage                        |
| pH_Value      | pH of the soil                             |
| Rainfall      | Rainfall in mm                             |
| Crop (Target) | Recommended crop label (rice, wheat, etc.) |

---

## 🧼 Data Preprocessing

- **Feature Scaling**: Standardized features using `StandardScaler`.
- **Label Encoding**: Converted crop names to numeric values using `LabelEncoder`.
- **Train-Test Split**: Data split into 80% training and 20% testing using `train_test_split`.

---

## 🤖 Classification Algorithms and When to Use Them

### 1. Naive Bayes
- **When**: Features are statistically independent; need a fast, simple model.
- **Why**: GaussianNB assumes normal distribution; fits this context.
- **Math**: Applies Bayes' Theorem to calculate crop probabilities.

### 2. Logistic Regression
- **When**: Data is linearly separable; focus on interpretability.
- **Why**: Provides baseline multiclass predictions.
- **Math**: Uses sigmoid/softmax for class probabilities.

### 3. Random Forest
- **When**: Complex, nonlinear data; importance analysis needed.
- **Why**: Handles mixed data types; prevents overfitting.
- **Math**: Combines multiple decision trees using majority voting.

### 4. Decision Tree (Entropy & Gini)
- **When**: Interpretability is key.
- **Why**: Simple rules like "If pH < 6.5 and N > 80, then rice".
- **Math**: Splits data based on entropy or Gini impurity.

### 5. Support Vector Classifier (SVC)
- **When**: High-dimensional data; clear class margin.
- **Why**: Captures complex boundaries with kernels.
- **Math**: Finds hyperplane maximizing margin.

### 6. K-Nearest Neighbors (KNN)
- **When**: Small datasets; intuitive logic.
- **Why**: Predicts based on local data similarity.
- **Math**: Uses distance (e.g., Euclidean) and majority voting.

### 7. Voting Classifier (Ensemble)
- **When**: Need more robust results.
- **Why**: Combines strengths of multiple models.
- **Math**: Averages class probabilities (soft voting).

---

## 📈 Evaluation Metrics

| Metric     | Description                                                       |
|------------|-------------------------------------------------------------------|
| Accuracy   | Correct predictions vs. total predictions.                        |
| Precision  | Correctly predicted crops vs. total predicted crops.             |
| Recall     | Correctly predicted crops vs. actual crops.                      |
| F1 Score   | Harmonic mean of precision and recall.                           |

---

## 🧪 Results Summary

| Model                      | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| Naive Bayes               | 0.9955   | 0.9958    | 0.9955 | 0.9954   |
| Logistic Regression       | 0.9636   | 0.9644    | 0.9636 | 0.9635   |
| Random Forest             | 0.9931   | 0.9937    | 0.9931 | 0.9931   |
| Decision Tree (Entropy)   | 0.9795   | 0.9802    | 0.9795 | 0.9793   |
| Decision Tree (Gini)      | 0.9863   | 0.9868    | 0.9863 | 0.9863   |
| Support Vector Classifier | 0.9795   | 0.9820    | 0.9795 | 0.9796   |
| K-Nearest Neighbors       | 0.9681   | 0.9727    | 0.9681 | 0.9682   |
| Voting Classifier         | 0.9841   | 0.9857    | 0.9841 | 0.9844   |

---

## 📦 Libraries Used

- `pandas`, `numpy` – Data handling
- `scikit-learn` – ML models, metrics, preprocessing, hyperparameter tuning
- `tabulate` – Pretty console table output

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/crop-recommendation-ml.git
cd crop-recommendation-ml
pip install -r requirements.txt
python crop_prediction.py
```

> The script will preprocess the data, train all models, perform evaluation, and print the results table.

---

## 📚 Future Work

- Deploy best model using Flask or FastAPI
- Add real-time farmer input support
- Integrate sensor/satellite data
- Add comparisons with XGBoost, LightGBM

---

## 🧑‍🔬 Author

**Raj Mehta**  
B.Tech in CSE with specialization in Cybersecurity  


---

## ⭐ Star this repo if you found it useful!
