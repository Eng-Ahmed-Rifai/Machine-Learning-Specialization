# Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning

**Instructor:** Andrew Ng (DeepLearning.AI & Stanford Online)  
**Credential Status:** Completed (Grade: 100.00%)  
**Verification ID:** [P1FH0Q4RG5X6](https://www.coursera.org/account/accomplishments/verify/P1FH0Q4RG5X6)  
**Learner:** Eng. Ahmed Mohamed Abdullatif Rifai  

---

## 🎯 Executive Summary
Course 3 concludes the prestigious Stanford / DeepLearning.AI Machine Learning Specialization by mastering modern unsupervised learning paradigms, industrial-scale recommender systems, and reinforcement learning. Spanning 3 intensive weeks, 13 graded assessments, and 5 programming assignments, every model was built and verified with 100.00% precision:
1. **Unsupervised Learning:** K-Means clustering, image vector quantization compression, multi-dimensional Gaussian density estimation, and anomaly detection with automated $F_1$ threshold tuning.
2. **Recommender Systems:** Collaborative filtering matrix factorization with mean normalization, followed by modern two-tower deep neural networks for content-based retrieval and embedding generation.
3. **Reinforcement Learning:** Discrete and continuous Markov Decision Processes (MDPs), Bellman equation dynamics, and Deep Q-Networks (DQN) with replay buffers and target network stabilization controlling the OpenAI Gym Lunar Lander.

---

## 📐 Mathematical Architecture & Core Algorithms

### 1. K-Means Clustering & Distortion Function
K-Means partitions an unlabeled dataset $\{x^{(1)}, \dots, x^{(m)}\} \subset \mathbb{R}^n$ into $K$ disjoint clusters.

#### Distortion Function (Inertia):
$$J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$

Where:
- $c^{(i)} \in \{1, \dots, K\}$: Index of cluster centroid assigned to sample $x^{(i)}$.
- $\mu_k \in \mathbb{R}^n$: Centroid coordinates for cluster $k$.

#### Optimization Steps:
1. **Centroid Assignment Step:** Minimize $J$ with respect to $c^{(1)}, \dots, c^{(m)}$ holding $\mu$ fixed:
   $$c^{(i)} := \arg\min_k ||x^{(i)} - \mu_k||^2$$
2. **Centroid Update Step:** Minimize $J$ with respect to $\mu_1, \dots, \mu_K$ holding $c$ fixed:
   $$\mu_k := \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$

#### Application — 16-Color Vector Quantization:
In the lab, K-Means was implemented from scratch to compress a 24-bit RGB photo (16,777,216 possible colors) down to 16 centroid colors, drastically compressing image payload from $128 \times 128 \times 24$ bits to 16 RGB values plus 4-bit cluster index references.

---

### 2. Anomaly Detection via Multi-Variate Gaussian Density
Given an unlabeled training set of normal operating examples $\{x^{(1)}, \dots, x^{(m)}\}$:

#### Model Formulation:
Assuming independent features (or post-decorrelation):
$$p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^n \frac{1}{\sqrt{2\pi \sigma_j^2}} \exp\left( -\frac{(x_j - \mu_j)^2}{2\sigma_j^2} \right)$$

#### Maximum Likelihood Parameter Estimation:
$$\mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)}, \quad \sigma_j^2 = \frac{1}{m} \sum_{i=1}^m (x_j^{(i)} - \mu_j)^2$$

#### Automated Threshold Selection ($\epsilon$) via $F_1$-Score:
On a small cross-validation set containing labeled anomalies:
$$\text{Predict } y = 1 \text{ (Anomaly) if } p(x) < \epsilon$$
$$\text{Precision} = \frac{tp}{tp + fp}, \quad \text{Recall} = \frac{tp}{tp + fn}$$
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
The optimal $\epsilon$ maximizes $F_1$ on cross-validation data.

---

### 3. Collaborative Filtering Recommender Systems
Predicting unknown rating $y^{(i, u)}$ of movie/item $i$ by user $u$ using latent item vectors $x^{(i)} \in \mathbb{R}^n$ and user preference vectors $w^{(u)} \in \mathbb{R}^n$:

$$\hat{y}^{(i, u)} = w^{(u)} \cdot x^{(i)} + b^{(u)}$$

#### Cost Function with $L_2$ Regularization:
$$J(w, b, x) = \frac{1}{2} \sum_{(i, u): r(i,u)=1} \left( w^{(u)} \cdot x^{(i)} + b^{(u)} - y^{(i, u)} \right)^2 + \frac{\lambda}{2} \sum_{u=1}^{n_u} \sum_{k=1}^n (w_k^{(u)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$$

#### Mean Normalization:
Subtracting each movie's mean observed rating $\mu_i$ allows the model to predict reasonable average defaults ($\mu_i$) for new users who have not yet rated any items.

---

### 4. Content-Based Deep Filtering (Two-Tower Architecture)
For systems rich in metadata, user profiles $x_u$ and item profiles $x_m$ are projected through parallel neural networks into a shared latent space $\mathbb{R}^d$:

$$v_u = f_{\text{user}}(x_u) \in \mathbb{R}^d, \quad v_m = f_{\text{item}}(x_m) \in \mathbb{R}^d$$
$$\hat{y}_{u, m} = v_u \cdot v_m$$

#### Loss Formulation:
$$J = \frac{1}{m} \sum_{(u, m)} (v_u \cdot v_m - y_{u, m})^2$$

#### Scalable Production Serving:
Item vectors $v_m$ are pre-computed offline and indexed into approximate nearest neighbor (ANN) vector stores (such as ScaNN / FAISS). At runtime, computing $v_u$ requires a single inference pass followed by rapid dot-product top-$k$ vector retrieval.

---

### 5. Reinforcement Learning & Deep Q-Networks (DQN)

#### Markov Decision Process (MDP):
Defined by $(S, A, P, R, \gamma)$:
- States $S$, Actions $A$
- Reward $R(s, a, s')$
- Discount factor $\gamma \in [0, 1)$
- Return $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$

#### Bellman Equation for Optimal Action-Value $Q^*(s, a)$:
$$Q^*(s, a) = R(s) + \gamma \max_{a'} Q^*(s', a')$$

#### Deep Q-Learning Loss Function:
In continuous state environments where $|S| = \infty$, $Q(s, a; \theta)$ is parameterized by neural network weights $\theta$:
$$y_j = R_j + (1 - \text{done}_j) \gamma \max_{a'} Q(s'_j, a'; \theta^-)$$
$$L(\theta) = \frac{1}{B} \sum_{j=1}^B \left( y_j - Q(s_j, a_j; \theta) \right)^2$$

Where:
- $\theta^-$: Parameters of the Target Network (frozen or soft-updated via Polyak averaging: $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$) to stabilize training.
- Experience Replay: Uniformly sampling batches $(s, a, R, s', \text{done})$ breaks temporal correlation among sequential steps.
- $\epsilon$-Greedy Policy: Balances exploration and exploitation ($a = \arg\max_a Q(s, a)$ with probability $1-\epsilon$, random action with probability $\epsilon$).

---

## 🔬 Assignments & Grades Matrix

| Module | Graded Component | Type | Points / Weight | Score | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Week 1** | Clustering | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 1** | k-means (`C3_W1_KMeans_Assignment.ipynb`) | Programming Lab | 14% | **100 / 100** | Passed |
| **Week 1** | Anomaly detection | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 1** | Anomaly Detection (`C3_W1_Anomaly_Detection.ipynb`) | Programming Lab | 14% | **100 / 100** | Passed |
| **Week 2** | Collaborative Filtering | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 2** | Collaborative Filtering Recommender Systems (`C3_W2_Collaborative_RecSys_Assignment.ipynb`) | Programming Lab | 14% | **100 / 100** | Passed |
| **Week 2** | Recommender systems implementation | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 2** | Content-based filtering | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 2** | Deep Learning for Content-Based Filtering (`C3_W2_RecSysNN_Assignment.ipynb`) | Programming Lab | 14% | **100 / 100** | Passed |
| **Week 3** | Reinforcement learning introduction | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 3** | State-action value function | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 3** | Continuous state spaces | Practice Quiz | 3% | **100.00%** | Passed |
| **Week 3** | Reinforcement Learning (`C3_W3_A1_Assignment.ipynb`) | Programming Lab | 20% | **100 / 100** | Passed |
| **Overall** | **Course Final Grade** | **Total** | **100%** | **100.00%** | **Certified** |

---

## 🏆 Official Certification Artifacts
- **Official Vector PDF:** [`DeepLearning_ML_Course3_Unsupervised_Learning_Recommenders_Reinforcement_Learning_P1FH0Q4RG5X6.pdf`](../certificates/DeepLearning_ML_Course3_Unsupervised_Learning_Recommenders_Reinforcement_Learning_P1FH0Q4RG5X6.pdf)
- **High-Resolution PNG:** [`DeepLearning_ML_Course3_Unsupervised_Learning_Recommenders_Reinforcement_Learning_P1FH0Q4RG5X6.png`](../certificates/DeepLearning_ML_Course3_Unsupervised_Learning_Recommenders_Reinforcement_Learning_P1FH0Q4RG5X6.png)
- **Official Coursera Verification:** [https://www.coursera.org/account/accomplishments/verify/P1FH0Q4RG5X6](https://www.coursera.org/account/accomplishments/verify/P1FH0Q4RG5X6)
