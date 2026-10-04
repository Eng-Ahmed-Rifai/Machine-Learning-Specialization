# Course 1: Supervised Machine Learning: Regression and Classification

**Instructor:** Andrew Ng (DeepLearning.AI & Stanford Online)  
**Credential Status:** Completed (Grade: 100.00%)  
**Verification ID:** [YG4LQQ8DT7IQ](https://www.coursera.org/account/accomplishments/verify/YG4LQQ8DT7IQ)  
**Learner:** Eng. Ahmed Mohamed Abdullatif Rifai  

---

## 🎯 Executive Summary
Course 1 establishes the mathematical, statistical, and computational foundations of modern supervised machine learning. Spanning linear models for continuous prediction to non-linear classification via logistic functions, the curriculum emphasizes vectorization in NumPy, rigorous gradient descent derivation, objective loss minimization, feature engineering, and regularization against overfitting.

---

## 📐 Mathematical Architecture & Core Algorithms

### 1. Univariate & Multivariate Linear Regression
Linear regression models the relationship between an input feature vector $\vec{x} \in \mathbb{R}^n$ and a continuous scalar output $y \in \mathbb{R}$:

$$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = \sum_{j=1}^n w_j x_j + b$$

#### Mean Squared Error (MSE) Cost Function
To quantify prediction error across $m$ training examples:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)^2$$

The division by $2$ cancels algebraically during differentiation, simplifying gradient updates.

#### Batch Gradient Descent Updates
Simultaneous parameter updates in each iteration:

$$w_j := w_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j} = w_j - \alpha \frac{1}{m} \sum_{i=1}^m \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$b := b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b} = b - \alpha \frac{1}{m} \sum_{i=1}^m \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

---

### 2. Feature Scaling & Gradient Descent Convergence
When features differ by orders of magnitude (e.g., $x_1 \in [0, 2000]$ vs. $x_2 \in [1, 5]$), cost contours become steep ellipses, causing gradient descent to oscillate inefficiently.

#### Normalization Techniques:
1. **Min-Max Normalization:**
   $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
2. **Z-score Standardization:**
   $$x_{scaled} = \frac{x - \mu}{\sigma}$$
   Ensures all features have mean $\mu = 0$ and standard deviation $\sigma = 1$.

---

### 3. Classification via Logistic Regression
For discrete binary classification targets $y \in \{0, 1\}$, linear regression fails because outputs are unbounded. Logistic regression maps linear combinations to probabilities via the Sigmoid (logistic) function:

$$g(z) = \frac{1}{1 + e^{-z}}, \quad z = \vec{w} \cdot \vec{x} + b$$

$$f_{\vec{w}, b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = P(y=1 \mid \vec{x}; \vec{w}, b)$$

- **Decision Boundary:** Predict $\hat{y} = 1$ when $f_{\vec{w}, b}(\vec{x}) \ge 0.5 \iff \vec{w} \cdot \vec{x} + b \ge 0$.

#### Binary Cross-Entropy Cost Function
Using MSE on sigmoid outputs yields a non-convex cost surface riddled with local minima. Instead, logistic loss uses convex negative log-likelihood:

$$L(f_{\vec{w}, b}(\vec{x}), y) = -y \log(f_{\vec{w}, b}(\vec{x})) - (1 - y) \log(1 - f_{\vec{w}, b}(\vec{x}))$$

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(f_{\vec{w}, b}(\vec{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\vec{w}, b}(\vec{x}^{(i)})) \right]$$

---

### 4. Overfitting, Bias/Variance, & Regularization
- **Underfitting (High Bias):** Model is too simplistic to capture underlying patterns.
- **Overfitting (High Variance):** Model fits training data noise and fails to generalize.

#### L2 Regularization (Ridge)
Applies a penalty proportional to the sum of squared weights:

$$J_{reg}(\vec{w}, b) = J(\vec{w}, b) + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$$

- $\lambda$: Regularization parameter balancing fit versus model simplicity.
- The intercept $b$ is intentionally **not** regularized.

#### Regularized Gradient Updates:
$$w_j := w_j (1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^m \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$b := b - \alpha \frac{1}{m} \sum_{i=1}^m \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

---

## 💻 Core Python & NumPy Implementation Reference

```python
import numpy as np

def sigmoid(z):
    """
    Compute sigmoid of z
    """
    z = np.clip(z, -500, 500)  # Numerical stability
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost_reg(X, y, w, b, lambda_reg=1.0):
    """
    Computes regularized logistic regression cost
    """
    m, n = X.shape
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Loss computation with epsilon clipping against log(0)
    eps = 1e-15
    f_wb = np.clip(f_wb, eps, 1 - eps)
    cost = - (1.0 / m) * np.sum(y * np.log(f_wb) + (1.0 - y) * np.log(1.0 - f_wb))
    
    # L2 Regularization penalty
    reg_cost = (lambda_reg / (2.0 * m)) * np.sum(w ** 2)
    return cost + reg_cost

def compute_gradient_reg(X, y, w, b, lambda_reg=1.0):
    """
    Computes regularized gradient for logistic regression
    """
    m, n = X.shape
    f_wb = sigmoid(np.dot(X, w) + b)
    err = f_wb - y
    
    dj_dw = (1.0 / m) * np.dot(X.T, err) + (lambda_reg / m) * w
    dj_db = (1.0 / m) * np.sum(err)
    
    return dj_db, dj_dw

def predict(X, w, b):
    """
    Predict binary labels {0, 1} using learned logistic regression parameters
    """
    p = sigmoid(np.dot(X, w) + b)
    return (p >= 0.5).astype(int)
```

---

## 📊 Course Assessment Performance

| Week | Assessment Title | Type | Weight | Grade Achieved |
| :--- | :--- | :--- | :---: | :---: |
| **Week 1** | Practice quiz: Supervised vs unsupervised learning | Quiz | 2% | **100.00%** |
| **Week 1** | Practice quiz: Regression | Quiz | 2% | **100.00%** |
| **Week 1** | Practice quiz: Train the model with gradient descent | Quiz | 2% | **100.00%** |
| **Week 2** | Practice quiz: Multiple linear regression | Quiz | 2% | **100.00%** |
| **Week 2** | Practice quiz: Gradient descent in practice | Quiz | 2% | **100.00%** |
| **Week 2** | Practice Lab: Linear Regression | Lab (Jupyter) | 41% | **100.00%** |
| **Week 3** | Practice quiz: Classification with logistic regression | Quiz | 2% | **100.00%** |
| **Week 3** | Practice quiz: Cost function for logistic regression | Quiz | 2% | **100.00%** |
| **Week 3** | Practice quiz: Gradient descent for logistic regression | Quiz | 2% | **100.00%** |
| **Week 3** | Practice quiz: The problem of overfitting | Quiz | 2% | **100.00%** |
| **Week 3** | Practice Lab: Logistic Regression | Lab (Jupyter) | 41% | **100.00%** |
| **Total** | **Overall Course Score** | **Combined** | **100%** | **100.00%** 🏆 |

---

## 📜 Verified Certificate & Artifacts
- **Certificate Verification URL:** [https://www.coursera.org/account/accomplishments/verify/YG4LQQ8DT7IQ](https://www.coursera.org/account/accomplishments/verify/YG4LQQ8DT7IQ)
- **Official Vector PDF:** `certificates/DeepLearning_ML_Course1_Supervised_Machine_Learning_YG4LQQ8DT7IQ.pdf`
- **High-Resolution Verification Screenshot:** `certificates/DeepLearning_ML_Course1_Supervised_Machine_Learning_YG4LQQ8DT7IQ.png`
- **Grade Transcript Screenshot:** `certificates/DeepLearning_ML_Course1_grades_100pct.png`
