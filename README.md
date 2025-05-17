# ghost_jobs_detection


#  Logistic Regression Explanation

**Logistic Regression** is a machine learning algorithm used for binary classification tasks. In the context of ghost job detection, it helps classify whether a job posting is a **ghost job (1)** or a **legit job (0)** by learning relationships between engineered features and the target label.

---

##  How Logistic Regression Works:

1. **Takes input features** (e.g., `missing_url`, `desc_length`, TF-IDF vectors)
2. **Applies weights to each feature**
3. **Computes a weighted sum**
4. **Applies the sigmoid function** to convert the sum into a probability between 0 and 1
5. **Predicts class label** based on a threshold (e.g., 0.5)

This makes logistic regression interpretable and ideal for baseline classification problems.

---

##  Key Concepts:

| Concept            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Sigmoid Function   | Converts a linear output into a probability                                 |
| Weights (Coefficients) | Represent the importance of each feature                                 |
| Regularization (L2) | Prevents overfitting by penalizing large weights                           |
| Class Weights      | Adjusts for imbalanced data by giving more importance to minority class     |
| TF-IDF             | Converts job descriptions into feature vectors based on word importance     |

---

##  Why Logistic Regression Is Useful for Ghost Job Detection:

- Simple and interpretable
- Performs well on linearly separable data
- Fast to train and evaluate
- Works well with sparse text data (TF-IDF)
- Supports class weighting for imbalance
- Great baseline before using complex models

---

## Challenges in Ghost Job Detection

###  Challenge 1: **Labeling Ghost Jobs Reliably**
- No true labels available
- Relied on proxy logic: match between company name and application URL

###  Challenge 2: **Handling Class Imbalance**
- More legit jobs than ghost jobs
- Solution: `class_weight='balanced'` to equalize impact

###  Challenge 3: **Feature Extraction from Text Data**
- Job descriptions are unstructured and noisy
- Applied TF-IDF with bigrams to extract meaningful features

###  Challenge 4: **Model Underfitting or Overfitting**
- Initial models overfit due to rule mirroring
- Later models underfit due to weak signal
- Balanced via regularization and feature tuning

---

##  Tools Used and Steps Performed

###  Tools:

- **Python**: Primary scripting language
- **pandas**: For data loading and transformation
- **scikit-learn**:
  - `LogisticRegression` for model training
  - `train_test_split` for data splitting
  - `classification_report` for evaluation
- **TfidfVectorizer**: For converting job descriptions into numerical vectors

---

##  Steps to Perform

###  1. Data Preprocessing:
- Load dataset using pandas
- Handle missing values
- Clean descriptions and trim URL formats

###  2. Ghost Job Labeling:
- Use business rule:
  - Ghost = URL missing or domain mismatch with company name

###  3. Feature Engineering:
- `missing_url`: 1 if URL is missing
- `desc_length`: number of characters in description
- `TF-IDF`: convert text to 500-dimensional sparse vector

###  4. Train-Test Split:
- Stratified split with 80% training, 20% testing

###  5. Model Building:
```python
LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
