# Course 2: Advanced Learning Algorithms

**Instructor:** Andrew Ng (DeepLearning.AI & Stanford Online)  
**Credential Status:** Completed (Grade: 100.00%)  
**Verification ID:** [J65E9KQ92PYL](https://www.coursera.org/account/accomplishments/verify/J65E9KQ92PYL)  
**Learner:** Eng. Ahmed Mohamed Abdullatif Rifai  

---

## 🎯 Executive Summary
Course 2 deepens machine learning mastery into non-linear architectures and practical engineering workflows. It bridges modern deep learning (multi-layer artificial neural networks, activation mechanics, multiclass softmax, and the Adam optimizer in TensorFlow) with disciplined diagnostic methodologies (bias/variance decomposition, cross-validation, learning curves, and error analysis) and non-parametric ensemble models (Decision Trees, Random Forests, and XGBoost). Every theoretical concept was verified with 100.00% empirical precision across 17 graded assessments and 4 comprehensive programming labs.

---

## 📐 Mathematical Architecture & Core Algorithms

### 1. Neural Network Architecture & Forward Propagation
A neural network composes successive affine transformations and non-linear activation functions across $L$ layers:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

Where:
- $W^{[l]} \in \mathbb{R}^{s_l \times s_{l-1}}$: Weight matrix for layer $l$.
- $b^{[l]} \in \mathbb{R}^{s_l}$: Bias vector for layer $l$.
- $g^{[l]}(\cdot)$: Element-wise activation function.
- $a^{[0]} = \vec{x}$: Input feature vector.

#### Activation Functions:
1. **Rectified Linear Unit (ReLU):**
   $$g(z) = \max(0, z)$$
   Default activation for hidden layers; avoids gradient saturation for $z > 0$, speeding up gradient descent compared to sigmoid.
2. **Sigmoid:**
   $$g(z) = \frac{1}{1 + e^{-z}}$$
   Used in binary classification output nodes ($y \in \{0, 1\}$).
3. **Linear Activation:**
   $$g(z) = z$$
   Used in regression output nodes ($y \in \mathbb{R}$).
4. **Softmax (Multiclass Output):**
   $$a_j = \frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}}, \quad j = 1, \dots, C$$

---

### 2. Multiclass Loss & Numerical Stability (Logits Formulation)
In multiclass classification across $C$ categories, standard Categorical Cross-Entropy is:

$$L(\vec{y}, \vec{a}) = -\sum_{j=1}^C y_j \log(a_j)$$

#### Numerical Instability in Naive Softmax:
Computing $a_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$ directly leads to floating-point underflow when $z_j \ll 0$ or overflow when $z_j \gg 0$. Furthermore, taking $\log(a_j)$ compounds rounding errors.

#### Numerically Stable `from_logits=True`:
Combining the softmax activation directly with the loss function allows TensorFlow to algebraically simplify the expression before evaluation:

$$\log(a_j) = z_j - \log\left(\sum_{k=1}^C e^{z_k}\right) = z_j - \left(z_{max} + \log\left(\sum_{k=1}^C e^{z_k - z_{max}}\right)\right)$$

This mathematically eliminates intermediate extreme exponents, preventing `NaN` and `Inf` numerical crashes.

---

### 3. Machine Learning Diagnostics & Diagnostics Workflow

#### Data Splitting Strategy:
- **Training Set (60%):** Minimize $J_{train}(\vec{w}, b)$ to optimize parameters.
- **Cross-Validation Set (20%):** Minimize $J_{cv}(\vec{w}, b)$ to select model architecture, polynomial degree $d$, and regularization parameter $\lambda$.
- **Test Set (20%):** Compute $J_{test}(\vec{w}, b)$ for unbiased generalization performance estimate.

#### Bias vs. Variance Diagnostics:
| Diagnostic Symptom | Root Cause | Remedies |
| :--- | :--- | :--- |
| $J_{train}$ high, $J_{cv} \approx J_{train}$ | **High Bias (Underfitting)** | Add features, increase model complexity (larger neural network), decrease $\lambda$ |
| $J_{train}$ low, $J_{cv} \gg J_{train}$ | **High Variance (Overfitting)** | Gather more training data, reduce feature count, increase $\lambda$ |
| $J_{train}$ high, $J_{cv} \gg J_{train}$ | **High Bias & High Variance** | Re-evaluate architecture, adjust baseline human error assumptions |

#### Performance on Skewed Datasets:
Classification accuracy is deceptive when class distributions are highly imbalanced.
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1\text{-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 4. Decision Trees & Ensemble Methods

#### Shannon Entropy & Information Gain
At node $S$, the impurity of binary classes (edible $p_1$ vs poisonous $p_0$) is quantified by Shannon entropy:

$$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1), \quad \text{with } H(0) = H(1) = 0$$

Splitting on feature $j$ creates left ($S_{left}$) and right ($S_{right}$) partitions. The Information Gain ($IG$) measures the reduction in entropy:

$$IG(S, j) = H(p_1^{node}) - \left( w^{left} H(p_1^{left}) + w^{right} H(p_1^{right}) \right)$$
$$\text{where } w^{left} = \frac{|S_{left}|}{|S|}, \quad w^{right} = \frac{|S_{right}|}{|S|}$$

The algorithm greedily selects the feature that maximizes $IG(S, j)$.

#### Stopping Criteria:
1. When node entropy reaches $0$ (100% pure node).
2. When the tree reaches predefined `max_depth`.
3. When the number of examples in a node falls below a threshold `min_samples_split`.
4. When information gain from further splits is below a threshold $\epsilon$.

#### Tree Ensembles:
1. **Bagged Decision Trees:** Train $B$ decision trees on bootstrap samples (sampling $m$ examples with replacement). Average predictions for regression or vote for classification.
2. **Random Forest:** Bagging combined with random feature subspace sampling: at each split, only a random subset of $k = \sqrt{n}$ features is evaluated, decorrelating the trees.
3. **XGBoost (Extreme Gradient Boosted Trees):** Successive trees fit directly to the residual errors of preceding trees with built-in regularization and highly optimized memory scaling.

#### Decision Trees vs. Neural Networks Trade-Off:
- **Decision Trees / Ensembles:** Excel on **structured / tabular data**. Fast training, interpretable, handle mixed feature types without feature scaling.
- **Neural Networks:** Excel on **unstructured data** (images, audio, text) and mixed complex pipelines. Seamless transfer learning and end-to-end differentiability.

---

## 💻 Core Implementation Reference

```python
import numpy as np

def compute_entropy(y):
    """Computes Shannon entropy for binary labels {0, 1}."""
    entropy = 0.0
    if len(y) > 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    return entropy

def split_dataset(X, node_indices, feature):
    """Splits data at the given node into left (feature==1) and right (feature==0)."""
    left_indices = [i for i in node_indices if X[i][feature] == 1]
    right_indices = [i for i in node_indices if X[i][feature] == 0]
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    """Calculates information gain achieved by splitting on a specific feature."""
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(left_indices) / len(node_indices)
    w_right = len(right_indices) / len(node_indices)
    
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    return node_entropy - weighted_entropy

def get_best_split(X, y, node_indices):
    """Greedily identifies the feature yielding maximum information gain."""
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0.0
    
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            
    return best_feature
```

---

## 📊 Course Assessment Performance

| Week | Assessment Title | Type | Weight | Grade Achieved |
| :--- | :--- | :--- | :---: | :---: |
| **Week 1** | Practice quiz: Neural networks intuition | Quiz | 2% | **100.00%** |
| **Week 1** | Practice quiz: Neural network model | Quiz | 2% | **100.00%** |
| **Week 1** | Practice quiz: TensorFlow implementation | Quiz | 2% | **100.00%** |
| **Week 1** | Practice quiz: Neural network implementation in Python | Quiz | 2% | **100.00%** |
| **Week 1** | Practice Lab: Neural Networks for Binary Classification | Lab (Jupyter) | 17% | **100.00%** |
| **Week 2** | Practice quiz: Neural Network Training | Quiz | 2% | **100.00%** |
| **Week 2** | Practice quiz: Activation Functions | Quiz | 2% | **100.00%** |
| **Week 2** | Practice quiz: Multiclass Classification | Quiz | 2% | **100.00%** |
| **Week 2** | Practice quiz: Additional Neural Network Concepts | Quiz | 2% | **100.00%** |
| **Week 2** | Practice Lab: Neural Networks for Multiclass classification | Lab (Jupyter) | 17% | **100.00%** |
| **Week 3** | Practice quiz: Advice for applying machine learning | Quiz | 2% | **100.00%** |
| **Week 3** | Practice quiz: Bias and variance | Quiz | 2% | **100.00%** |
| **Week 3** | Practice quiz: Machine learning development process | Quiz | 2% | **100.00%** |
| **Week 3** | Practice Lab: Advice for Applying Machine Learning | Lab (Jupyter) | 19% | **100.00%** |
| **Week 4** | Practice quiz: Decision trees | Quiz | 2% | **100.00%** |
| **Week 4** | Practice quiz: Decision tree learning | Quiz | 2% | **100.00%** |
| **Week 4** | Practice quiz: Tree ensembles | Quiz | 2% | **100.00%** |
| **Week 4** | Practice Lab: Decision Trees | Lab (Jupyter) | 19% | **100.00%** |
| **Total** | **Overall Course Score** | **Combined** | **100%** | **100.00%** 🏆 |

---

## 📜 Verified Certificate & Artifacts
- **Certificate Verification URL:** [https://www.coursera.org/account/accomplishments/verify/J65E9KQ92PYL](https://www.coursera.org/account/accomplishments/verify/J65E9KQ92PYL)
- **Official Vector PDF:** `certificates/DeepLearning_ML_Course2_Advanced_Learning_Algorithms_J65E9KQ92PYL.pdf`
- **High-Resolution Verification Screenshot:** `certificates/DeepLearning_ML_Course2_Advanced_Learning_Algorithms_J65E9KQ92PYL.png`
- **Grade Transcript Screenshot:** `certificates/DeepLearning_ML_Course2_grades_100pct.png`
