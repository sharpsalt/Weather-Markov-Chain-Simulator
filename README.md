#  Weather Markov Chain Simulator

A complete implementation of a **Discrete-Time Markov Chain (DTMC)** that models weather transitions between three states — **Sunny, Rainy, and Cloudy**. The project computes the **stationary distribution** using three independent methods and visually proves their convergence.

![Weather Markov Chain Simulator](weather_markov_results.png)

---

##  Table of Contents

- [What is a Markov Chain?](#what-is-a-markov-chain)
- [The Markov Property](#the-markov-property)
- [State Space & Transition Matrix](#state-space--transition-matrix)
- [How the Simulation Works](#how-the-simulation-works)
- [Three Methods to Find Stationary Distribution](#three-methods-to-find-stationary-distribution)
  - [1. Monte Carlo Simulation](#1-monte-carlo-simulation)
  - [2. Power Iteration](#2-power-iteration)
  - [3. Analytical Solution](#3-analytical-solution)
- [The Ergodic Theorem](#the-ergodic-theorem)
- [Results](#results)
- [Visualizations](#visualizations)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

---

## What is a Markov Chain?

A **Markov Chain** is a stochastic process that transitions between a finite set of states, where the probability of moving to the next state depends **only on the current state** — not on the sequence of states that preceded it.

It is defined by:
- A finite set of **states** $S = \{s_1, s_2, ..., s_n\}$
- A **transition matrix** $T$ where $T_{ij} = P(X_{t+1} = s_j \mid X_t = s_i)$

---

## The Markov Property

The core assumption — **memorylessness**:

$$P(X_{t+1} = s \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} = s \mid X_t)$$

> "The future depends only on the present, not the past."

This means the entire history of the chain is irrelevant; only the current state matters for predicting the next state.

---

## State Space & Transition Matrix

### States
$$S = \{\text{Sunny},\; \text{Rainy},\; \text{Cloudy}\}$$

### Transition Matrix

The transition matrix $T$ is a $3 \times 3$ row-stochastic matrix (each row sums to 1):

$$T = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.2 & 0.3 & 0.5 \end{pmatrix}$$

| From \ To | Sunny | Rainy | Cloudy | Row Sum |
|-----------|-------|-------|--------|---------|
| **Sunny** | 0.70 | 0.20 | 0.10 | 1.0 |
| **Rainy** | 0.30 | 0.40 | 0.30 | 1.0 |
| **Cloudy**| 0.20 | 0.30 | 0.50 | 1.0 |

**Reading the matrix:** If today is **Sunny** (row 1), there's a 70% chance tomorrow is Sunny, 20% Rainy, and 10% Cloudy.

### Properties of this chain:
- **Irreducible** — every state can be reached from every other state (fully connected graph)
- **Aperiodic** — self-loops exist ($T_{ii} > 0$), so states are not periodic
- **Ergodic** — irreducible + aperiodic → a **unique stationary distribution exists**

---

## How the Simulation Works

The model represents a **fully connected directed graph** with 3 nodes, each having 3 outgoing edges (including self-loops) weighted by transition probabilities:

```
         0.7
    ┌──────────┐
    │   Sunny   │
    └──────────┘
   ↗ 0.3    ↘ 0.2
  0.2↗        ↘0.1
┌──────┐    ┌──────┐
│Rainy │←──→│Cloudy│
└──────┘0.3 └──────┘
   0.4↺         0.5↺
```

**Algorithm (Monte Carlo):**
1. Pick an initial state (Sunny, index 0)
2. For each day $t = 1, 2, \ldots, 10000$:
   - Look at current state's row in $T$
   - Sample next state using those probabilities (weighted random choice)
   - Record the state
3. Count visits to each state
4. Divide by total steps → **empirical stationary distribution**

---

## Three Methods to Find Stationary Distribution

The **stationary distribution** $\pi$ is the probability vector satisfying:

$$\pi \cdot T = \pi \quad \text{subject to} \quad \sum_i \pi_i = 1$$

Once the chain reaches this distribution, it stays there forever.

---

### 1. Monte Carlo Simulation

**Approach:** Simulate the chain for $N = 10,000$ days and count how often each state is visited.

$$\hat{\pi}_i = \frac{\text{count of state } i}{N}$$

```python
def monte_carlo(T, num_days=10000, start_state=0):
    current = start_state
    visits = [current]
    for _ in range(num_days):
        current = np.random.choice(len(states), p=T[current])
        visits.append(current)
    counts = np.bincount(visits, minlength=3)
    return counts / len(visits)
```

**Result:** $\hat{\pi} \approx [0.4689,\; 0.2822,\; 0.2489]$

> Approximate due to random sampling variance. Increasing $N$ → more accurate.

---

### 2. Power Iteration

**Approach:** Raise the transition matrix to a large power $T^n$. As $n \to \infty$, all rows of $T^n$ converge to $\pi$.

$$\lim_{n \to \infty} T^n = \begin{pmatrix} \pi \\ \pi \\ \pi \end{pmatrix}$$

Uses **fast matrix exponentiation** ($O(\log n)$ matrix multiplications):

```python
def matrix_power(T, n):
    result = np.eye(len(T))
    base = T.copy()
    while n > 0:
        if n % 2 == 1:
            result = result @ base
        base = base @ base
        n //= 2
    return result

Tn = matrix_power(T, 1000)
pi = Tn[0]  # Any row works
```

**Result:** $\pi = [0.4565,\; 0.2826,\; 0.2609]$

---

### 3. Analytical Solution

**Approach:** Directly solve the linear system $\pi T = \pi$ with the constraint $\sum \pi_i = 1$.

Rewrite as:

$$\pi (T - I) = 0 \quad \Rightarrow \quad (T^\top - I)\pi^\top = 0$$

Replace one equation with the normalization constraint $\pi_1 + \pi_2 + \pi_3 = 1$:

$$A = (T^\top - I), \quad A[\text{last row}] = [1, 1, 1], \quad b = [0, 0, 1]$$

Then solve $A \cdot \pi = b$:

```python
def analytic_steady_state(T):
    n = len(T)
    A = (T.T - np.eye(n))
    A[-1, :] = 1.0      # normalization constraint
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)
```

**Result:** $\pi = [0.4565,\; 0.2826,\; 0.2609]$

---

## The Ergodic Theorem

For an ergodic Markov chain, the **Ergodic Theorem** guarantees:

$$\lim_{n \to \infty} \frac{1}{n} \sum_{t=1}^{n} \mathbf{1}(X_t = i) = \pi_i \quad \text{(almost surely)}$$

**In plain words:**
> The fraction of time spent in state $i$ over a long simulation converges to the steady-state probability $\pi_i$, regardless of the starting state.

This is why Monte Carlo, Power Iteration, and the Analytical method all converge to the **same answer** — they are computing the same fundamental quantity from different angles.

---

## Results

### Stationary Distribution Comparison

| Method | Sunny | Rainy | Cloudy |
|--------|-------|-------|--------|
| **Monte Carlo** (10K days) | 0.4689 | 0.2822 | 0.2489 |
| **Power Iteration** ($T^{1000}$) | 0.4565 | 0.2826 | 0.2609 |
| **Analytical** ($\pi T = \pi$) | 0.4565 | 0.2826 | 0.2609 |

### Interpretation
In the long run:
- **~45.65%** of days will be **Sunny** 
- **~28.26%** of days will be **Rainy** 
- **~26.09%** of days will be **Cloudy** 

### Distribution Evolution (starting from 100% Sunny)

| Step | Sunny | Rainy | Cloudy |
|------|-------|-------|--------|
| 0 | 1.0000 | 0.0000 | 0.0000 |
| 1 | 0.7000 | 0.2000 | 0.1000 |
| 5 | 0.4685 | 0.2794 | 0.2521 |
| 10 | 0.4568 | 0.2825 | 0.2607 |
| 20 | 0.4565 | 0.2826 | 0.2609 |
| 50 | 0.4565 | 0.2826 | 0.2609 |

> Convergence happens by ~step 15–20. After that, the distribution does not change.

---

## Visualizations

The project generates a 6-panel dashboard:

| Plot | Description |
|------|-------------|
| **Transition Matrix Heatmap** | Visual representation of $T$ with color-coded probabilities |
| **Simulated Weather (100 Days)** | Color-coded bar chart showing daily weather over first 100 days |
| **Running Frequency → Convergence** | Shows how empirical frequencies converge to $\pi$ over 10K days |
| **Distribution Evolution ($\pi \times T^t$)** | Probability vector evolving from $[1,0,0]$ to steady state |
| **Stationary Distribution — All Methods** | Side-by-side bar comparison of all 3 methods |
| **State Transitions (30 Days)** | Step plot showing state-to-state transitions |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations, random sampling, linear algebra |
| `matplotlib` | All 6 visualization plots |
| `collections.defaultdict` | State visit counting |

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Weather_Markov_Chain_Simulator.git
   cd Weather_Markov_Chain_Simulator
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```

3. **Open and run the notebook:**
   ```bash
   jupyter notebook Markov.ipynb
   ```
   Or open `Markov.ipynb` in **VS Code** and run all cells (`Ctrl+Shift+P` → "Notebook: Run All Cells").

---

## Key Takeaway

> **No matter how you compute it — brute-force simulation, matrix exponentiation, or solving linear equations — an ergodic Markov Chain always converges to the same unique stationary distribution.** This project demonstrates that convergence both numerically and visually.

---




