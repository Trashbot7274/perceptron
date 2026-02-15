
# Perceptron Visualizer 

Perceptron learning visualization that I was inspired to make after completing the MIT OCW 6.36 chapter on perceptrons!

## Demo

<p align="center">
  <img src="https://github.com/Trashbot7274/perceptron/blob/main/gifs/one.gif" width="45%" />
  <img src="https://github.com/Trashbot7274/perceptron/blob/main/gifs/two.gif" width="45%" />
</p>


<p align="center">
  <em>Left: 3rd order polynomial | Right: 15th order polynomial</em>
</p>

## What It Does

Trains a perceptron on 2D data by expanding features to polynomial order, creating non-linear decision boundaries in the original space. Visualizes training in real-time.

## Installation
```bash
pip install numpy matplotlib
```

## Usage
```python
python perceptron.py
```

Default behavior trains on 6D linearly separable data with order-15 polynomial features.

### Custom Training
```python
from perceptron import perceptron
from data import gen_lin_separable

# Generate data
x, y = gen_lin_separable(num_points=30)

# Train
weights, bias = perceptron(
    x, y,
    lr=0.01,          # Learning rate
    iterations=100,   # Epochs
    order=2,          # Polynomial order
    shuffle=True      # Shuffle each epoch
)
```

## Polynomial Features

For order `n`, generates all monomials x₁ⁱ · x₂ʲ where i + j ≤ n.

**Example (order=2):**
```
1, x₁, x₂, x₁², x₁x₂, x₂²
```

This transforms 2D points into 6D space where a linear boundary (in 6D) appears as a conic section (in 2D).

## Data Generators

The data.py file comes from MIT OCW 6.063 and is designed to generate a mix of separable and non-separable datasets 


**From `data.py`:**

- `gen_lin_separable()` - Linearly separable 2D data
- `gen_flipped_lin_separable()` - Noisy data with flipped labels
- `big_higher_dim_separable()` - 6D linearly separable data
- `big_data` - Fixed 100-point dataset

## Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `lr` | Learning rate | 0.001 - 0.1 |
| `iterations` | Training epochs | 50 - 1000 |
| `order` | Polynomial order | 1 - 15 |
| `shuffle` | Randomize samples | True/False |

## How It Works

1. **Feature Expansion:** Transform (x₁, x₂) → (1, x₁, x₂, x₁², x₁x₂, x₂², ...)
2. **Perceptron Update:** For each misclassified point:
```
   w ← w + lr · y · Φ(x)
   b ← b + lr · y
```
3. **Visualization:** Plot decision boundary and regions after each epoch

## Notes

- Higher order = more complex boundaries, risk of overfitting
- Learning rate too high → unstable, too low → slow convergence
- Code from MIT 6.036 OpenCourseWare

## Files

- `perceptron.py` - Main implementation
- `data.py` - Dataset generators (from MIT OCW)

---

**Simple example:**
```python
from perceptron import perceptron
from data import gen_lin_separable

x, y = gen_lin_separable(num_points=20)
weights, bias = perceptron(x, y, lr=0.01, iterations=50, order=2)
```
