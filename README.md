# Bio-Inspired Optimisation & Neural Networks

A Python implementation of three foundational computational intelligence techniques, exploring how nature-inspired and analytically-driven methods solve optimisation and learning problems that brute force cannot handle efficiently.

---

## Contents

| Section | Technique | Problem Solved |
|---|---|---|
| 1 | Particle Swarm Optimisation (PSO) | Global optimisation on the Griewank benchmark |
| 2 | RBF Neural Network | Function approximation via Moore-Penrose pseudoinverse |
| 3 | Multi-Armed Bandit | Sequential decision making under uncertainty |

---

## Part 1 — Particle Swarm Optimisation

### What it does
PSO is a population-based metaheuristic that searches for the global minimum of a function by simulating the collective behaviour of a swarm. Each particle tracks its own best position and the swarm's global best, adjusting its velocity at every step based on both.

### Benchmark function — Griewank
```
f(x) = Σ(xᵢ²/4000) − Π(cos(xᵢ/√i)) + 1
```
- **Global minimum:** 0 at the origin
- **Why Griewank:** it has thousands of local minima designed to trap naive search methods. The product term creates regular oscillations that make nearly every valley look like the deepest one. PSO's swarm communication helps particles escape local traps and continue searching.

### Velocity update (core equation)
```
v = w·v + c1·r1·(personal_best − position) + c2·r2·(global_best − position)
```
| Parameter | Value | Role |
|---|---|---|
| `w = 0.7` | Inertia weight | How much past velocity persists |
| `c1 = 1.5` | Cognitive coefficient | Pull toward particle's own best |
| `c2 = 1.5` | Social coefficient | Pull toward swarm's global best |
| 30 particles | — | Swarm size |
| 200 iterations | — | Search budget |
| Bounds: ±600 | — | Search space |

### Result
```
Best position : [3.14, -4.44]
Best value    : 0.0074  (true minimum = 0.0)
```

---

## Part 2 — RBF Neural Network

### Architecture
```
Input → [RBF Hidden Layer: Gaussian neurons] → [Linear Output Layer]
```
Unlike standard neural networks where neurons respond globally, each RBF neuron has a centre point and fires strongly when the input is close to that centre, weakly when far away. The response is a Gaussian bell curve:
```
φ(x, cⱼ) = exp(−‖x − cⱼ‖² / 2σ²)
```

### Three-step training process

**Step 1 — Find centres (k-means)**
K-means clustering places 15 representative points across the training data. These become the RBF neuron centres, ensuring coverage of the input space.

**Step 2 — Set widths (heuristic)**
```
σ = d_max / √(2K)
```
Where `d_max` is the maximum distance between any two centres and `K` is the number of centres. This ensures neurons cover the input space without excessive overlap or gaps.

**Step 3 — Solve weights (Moore-Penrose pseudoinverse)**
Once centres and widths are fixed, the output is a linear combination of RBF activations. The optimal weights are solved in a single closed-form step:
```
weights = Φ⁺ · y = pinv(Φ) @ y
```
This gives the globally optimal least-squares solution without iterative gradient descent, no learning rate to tune, no epochs to run, deterministic result every time.

### Demo dataset
Noisy sine wave: `y = sin(x) + N(0, 0.15)` over `[0, 2π]`

### Result
```
Centres : 15
MSE     : 0.0025
RMSE    : 0.0496
```

---

## Part 3 — Multi-Armed Bandit

### The problem
10 slot machines, each with an unknown true reward distribution. Given a fixed number of pulls, find the strategy that maximises total reward — balancing **exploration** (trying new arms to gather information) against **exploitation** (pulling the currently best-known arm).

### Strategies compared

**Greedy (ε = 0)** — always pull the arm with the highest current estimated value. Commits too early; gets trapped on suboptimal arms if early samples are misleading.

**Epsilon-Greedy (ε = 0.1)** — exploit the best known arm 90% of the time, explore randomly 10% of the time. Best balance of exploration and exploitation.

**Epsilon-Greedy (ε = 0.01)** — exploits more aggressively. Converges faster but misses better arms if not explored early enough.

**Random** — ignore all information, pull randomly. Baseline. Never converges.

### Incremental mean update
```
Q[a] += (reward − Q[a]) / N[a]
```
Maintains a running average without storing all historical rewards. Efficient for online learning — only needs the current estimate, the new reward, and the pull count.

### Results (500 runs × 1000 steps, 10-arm bandit)

| Strategy | Avg Reward (last 100) | % Optimal Action (last 100) |
|---|---|---|
| Epsilon-Greedy ε=0.1 | 1.385 | 79.6% |
| Epsilon-Greedy ε=0.01 | 1.363 | 62.5% |
| Greedy ε=0 | 1.040 | 34.0% |
| Random | -0.013 | 9.9% |

**Key finding:** ε=0.1 outperforms because it maintains consistent exploration throughout. Pure greedy commits too early and never recovers. Random never learns at all.

---

## What connects all three

All three methods solve a version of the same fundamental challenge: **searching or learning under uncertainty without being able to evaluate every possibility.**

- **PSO** — escapes local minima through collective swarm intelligence
- **RBF Network** — approximates unknown functions using local Gaussian responses, solved analytically in one shot
- **Multi-Armed Bandit** — learns optimal actions through structured trial and experience

---

## How to run

```bash
pip install numpy matplotlib
python bio_inspired_optimisation.py
```

**Output files generated:**
- `pso_convergence.png` — PSO fitness over iterations
- `rbf_regression.png` — RBF fit vs true sine curve
- `bandit_comparison.png` — reward and optimal action rate per strategy

---

## Requirements

```
numpy
matplotlib
```

---

## Author

**Vishnu Koushik Tekuru**
MSc Health Data Science, University of Liverpool
[github.com/VishnuKoushikTekuru](https://github.com/VishnuKoushikTekuru)
[linkedin.com/in/vishnu-koushik-t-46aa1b12a](https://linkedin.com/in/vishnu-koushik-t-46aa1b12a)
