# maths vs ai
```
Sure! Here are some equivalent mathematical formulas for each of the points mentioned in the context of their applications in AI:

### 1. **Linear Algebra**
- **Matrix Operations**:
  - **Matrix Multiplication**: \( C = A \cdot B \)
  - **Dot Product**: \( \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i \)
- **Principal Component Analysis (PCA)**:
  - **Covariance Matrix**: \( \Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \mathbf{\mu})(\mathbf{x}_i - \mathbf{\mu})^T \)
  - **Eigenvalue Decomposition**: \( \Sigma \mathbf{v} = \lambda \mathbf{v} \)
- **Singular Value Decomposition (SVD)**:
  - \( A = U \Sigma V^T \)

### 2. **Calculus**
- **Optimization Algorithms**:
  - **Gradient Descent Update Rule**: \( \theta := \theta - \eta \nabla_{\theta} J(\theta) \)
- **Backpropagation**:
  - **Chain Rule**: \( \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} \)

### 3. **Probability and Statistics**
- **Bayesian Networks**:
  - **Bayes' Theorem**: \( P(A|B) = \frac{P(B|A)P(A)}{P(B)} \)
- **Markov Chains**:
  - **Transition Matrix**: \( P(X_{t+1} = s_j | X_t = s_i) = P_{ij} \)
- **Hypothesis Testing**:
  - **Z-Score**: \( z = \frac{\bar{x} - \mu}{\frac{\sigma}{\sqrt{n}}} \)
  - **P-Value Calculation**: Depends on the specific test being used (e.g., t-test, chi-squared test).

### 4. **Discrete Mathematics**
- **Graph Theory**:
  - **Adjacency Matrix**: \( A_{ij} = \begin{cases} 1 & \text{if there is an edge from } i \text{ to } j \\ 0 & \text{otherwise} \end{cases} \)
- **Combinatorics**:
  - **Binomial Coefficient**: \( \binom{n}{k} = \frac{n!}{k!(n-k)!} \)

### 5. **Optimization**
- **Linear Programming**:
  - **Objective Function**: \( \text{maximize} \ c^T x \)
  - **Constraints**: \( Ax \leq b \)
- **Convex Optimization**:
  - **Convex Function**: \( f(\theta) \text{ is convex if } f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y) \text{ for all } \lambda \in [0, 1] \)

### 6. **Information Theory**
- **Entropy**:
  - \( H(X) = -\sum_{i=1}^{n} P(x_i) \log P(x_i) \)
- **Mutual Information**:
  - \( I(X; Y) = \sum_{y \in Y} \sum_{x \in X} P(x, y) \log \frac{P(x, y)}{P(x)P(y)} \)
- **Kullback-Leibler Divergence**:
  - \( D_{KL}(P \| Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)} \)

### 7. **Numerical Methods**
- **Numerical Differentiation**:
  - **Forward Difference Approximation**: \( f'(x) \approx \frac{f(x+h) - f(x)}{h} \)
- **Numerical Integration**:
  - **Trapezoidal Rule**: \( \int_{a}^{b} f(x) \, dx \approx \frac{b-a}{2} [f(a) + f(b)] \)
- **Solving Linear Systems**:
  - **Gauss-Seidel Method**: \( x^{(k+1)}_i = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x^{(k+1)}_j - \sum_{j=i+1}^{n} a_{ij} x^{(k)}_j \right) \)

### 8. **Graph Theory**
- **Graph Neural Networks (GNNs)**:
  - **Node Update Rule**: \( h_v^{(k)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \frac{1}{c_{vu}} W^{(k)} h_u^{(k-1)} \right) \)
- **PageRank Algorithm**:
  - **PageRank Update Rule**: \( PR(i) = \frac{1-d}{N} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)} \)
  - Here, \(d\) is the damping factor, \(N\) is the total number of nodes, \(M(i)\) is the set of pages linking to \(i\), and \(L(j)\) is the number of outbound links on page \(j\).

These formulas represent the core mathematical concepts used in various AI applications and algorithms. Understanding these formulas helps in grasping how AI models and techniques are constructed and optimized.

```
