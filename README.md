# Micrograd & PyTorch Validation Engine

A lightweight, scalar-valued automatic differentiation engine with dynamic DAG construction, custom neural network primitives, and numerical gradient verification against PyTorch (`torch.autograd`).

## Key Features

- **Scalar Autograd Engine**: Reverse-mode automatic differentiation over dynamic computational graphs (`Value` abstraction).
- **Rich Activation Functions**: Built-in support for `ReLU`, `LeakyReLU`, `ELU`, `SiLU`, `GELU` (tanh approximation), `sigmoid`, and `tanh`.
- **Modular Neural Networks**: Custom implementation of `Neuron`, `Layer`, and `MLP` architectures mimicking PyTorch's `nn.Module` design.
- **PyTorch Equivalence Benchmark**: Strict step-by-step verification comparing forward loss and exact parameter gradients against `torch.nn` and `torch.autograd`.
- **DAG Visualization**: Export computational graphs to SVG/PNG using Graphviz (`graph.py`).

## Mathematical Foundation

The backward pass computes exact partial derivatives $\frac{\partial L}{\partial x_i}$ via topological sorting of the computational DAG:

$$v_i.\text{grad} = \sum_{j \in \text{children}(i)} v_j.\text{grad} \cdot \frac{\partial v_j}{\partial v_i}$$

Activation functions such as GELU are constructed using scalar operations directly:

$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

## Quick Start

```python
from value import Value
from nn import MLP

# Initialize model (2 inputs -> hidden layers [16, 16] -> 1 output)
model = MLP(2, [16, 16, 1])

# Forward pass
x = [Value(2.0), Value(-3.0)]
out = model(x)

# Backward pass
out.backward()
print(f"Output: {out.data:.4f}, Gradient wrt x[0]: {x[0].grad:.4f}")
```

## Running the Benchmark

The `test.py` script trains both the custom `MLP` and an identical `torch.nn.Sequential` model on the `make_moons` dataset using Binary Cross-Entropy (BCE) loss:

```bash
python comparison_with_torch.py
```

### Verification Results

Maximum absolute gradient deviation ($\max |g_{\text{custom}} - g_{\text{pytorch}}|$) across all weights and biases:

| Epoch | Custom BCE Loss | PyTorch BCE Loss | Max Grad Diff |
| :---: | :-------------: | :--------------: | :-----------: |
|  10   |    1.301180     |     1.301179     |   1.67e-06    |
|  20   |    0.981055     |     0.981054     |   6.26e-07    |
|  50   |    0.526500     |     0.526500     |   5.59e-08    |

*Gradients match PyTorch autograd outputs within single-precision floating-point tolerances ($< 10^{-6}$).*

## Visualizing Computation Graphs

To render the computational graph for a loss scalar into an SVG vector image:

```python
from graph import draw_dot

# Generate Graphviz Digraph
dot = draw_dot(total_loss)
dot.render("graph", format="svg", view=False)
```

## Dependencies

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- tqdm
- graphviz (Python package & system binary)
