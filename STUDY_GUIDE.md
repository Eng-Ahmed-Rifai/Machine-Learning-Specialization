# 🧠 Machine Learning Specialization — Comprehensive Study Guide

**Author:** Eng. Ahmed Rifai  
**Credential:** [CVSVNIM5HXSN](https://www.coursera.org/account/accomplishments/specialization/CVSVNIM5HXSN) (Verified 100% Mastery across all 3 Courses)  
**Instructors:** Andrew Ng, Eddy Shyu, Aarti Bagul, Geoff Ladwig (DeepLearning.AI / Stanford Online)  
**Target:** Theoretical Foundations, Mathematical Equations, Algorithm Workflows, and Code Implementations.

---

## 📑 Table of Contents
1. [Course 1: Supervised Machine Learning (Regression & Classification)](#course-1-supervised-machine-learning)
2. [Course 2: Advanced Learning Algorithms (Neural Networks & Decision Trees)](#course-2-advanced-learning-algorithms)
3. [Course 3: Unsupervised Learning, Recommenders & Reinforcement Learning](#course-3-unsupervised-learning-recommenders--reinforcement-learning)

---

## Course 1: Supervised Machine Learning

### 1. Linear Regression with One & Multiple Variables
- **Model Representation**:
  $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
- **Cost Function (Squared Error)**:
  $$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)^2$$
- **Gradient Descent Algorithm**:
  Repeat until convergence:
  $$w_j := w_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j} = w_j - \alpha \frac{1}{m} \sum_{i=1}^m \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right) x_j^{(i)}$$
  $$b := b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b} = b - \alpha \frac{1}{m} \sum_{i=1}^m \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)$$
- **Feature Scaling**: Z-score normalization transforms each feature to mean $\mu = 0$ and standard deviation $\sigma = 1$:
  $$x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}$$

### 2. Logistic Regression for Classification
- **Sigmoid Function**:
  $$g(z) = \frac{1}{1 + e^{-z}}, \quad 0 < g(z) < 1$$
- **Decision Boundary**: Predict $y = 1$ if $f_{\vec{w}, b}(\vec{x}) \ge 0.5 \iff \vec{w} \cdot \vec{x} + b \ge 0$.
- **Binary Cross-Entropy Loss**:
  $$L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\vec{w}, b}(\vec{x}^{(i)})\right) - (1 - y^{(i)}) \log\left(1 - f_{\vec{w}, b}(\vec{x}^{(i)})\right)$$
- **Regularization (L2 Penalty)**:
  $$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^m L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$$

---

## Course 2: Advanced Learning Algorithms

### 1. Neural Networks & Multiclass Softmax
- **Layer Computation**:
  $$a^{[l]} = g\left(W^{[l]} a^{[l-1]} + b^{[l]}\right)$$
- **Softmax Regression**: Outputs probabilities across $K$ classes:
  $$a_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = P(y = k \mid \vec{x})$$
  Loss: $-\log(a_y)$. Computed stably using `from_logits=True`.

### 2. ML System Diagnostics & Bias-Variance
| Metric Condition | Diagnosis | Primary Solutions |
|:---|:---|:---|
| $J_{train}$ high, $J_{cv} \approx J_{train}$ | **High Bias (Underfitting)** | Increase model size, add features, decrease $\lambda$ |
| $J_{train}$ low, $J_{cv} \gg J_{train}$ | **High Variance (Overfitting)** | Get more training data, reduce features, increase $\lambda$ |

### 3. Decision Trees & Ensemble Methods
- **Entropy & Information Gain**:
  $$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$
  $$\text{Information Gain} = H(p_1^{node}) - \left(w^{left} H(p_1^{left}) + w^{right} H(p_1^{right})\right)$$
- **Random Forests**: Bagging (Bootstrap Aggregation) + random feature subsampling at each split.
- **XGBoost**: Trains new decision trees sequentially to predict the residual errors of preceding trees.

---

## Course 3: Unsupervised Learning, Recommenders & Reinforcement Learning

### 1. K-Means Clustering & Anomaly Detection
- **K-Means Optimization**: Minimizes distortion $J(c, \mu) = \frac{1}{m} \sum \|x^{(i)} - \mu_{c^{(i)}}\|^2$.
- **Anomaly Detection (Gaussian Estimation)**:
  $$p(\vec{x}) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^n \frac{1}{\sqrt{2\pi}\sigma_j} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$
  Flag as anomaly if $p(\vec{x}) < \epsilon$.

### 2. Collaborative Filtering & Content-Based Recommenders
- **Collaborative Filtering Cost Function**:
  $$J(X, W, b) = \frac{1}{2} \sum_{(i, j): r(i, j)=1} \left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \|\vec{w}^{(j)}\|^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \|\vec{x}^{(i)}\|^2$$

### 3. Reinforcement Learning & Deep Q-Networks (DQN)
- **Markov Decision Process (MDP)**: $(S, A, P, R, \gamma)$.
- **Return with Discount $\gamma$**:
  $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$
- **Bellman Equation**:
  $$Q(s, a) = R(s) + \gamma \max_{a'} Q(s', a')$$
- **Deep Q-Learning**: Uses a neural network to approximate $Q(s, a; \theta)$, updated via Mean Squared Bellman Error with experience replay and target networks.
