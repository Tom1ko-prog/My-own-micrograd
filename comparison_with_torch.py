import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn
from tqdm import tqdm

from nn import MLP
from value import Value

torch_model = nn.Sequential(
    nn.Linear(2, 16),
    nn.GELU(approximate="tanh"),
    nn.Linear(16, 16),
    nn.LeakyReLU(0.01),
    nn.Linear(16, 1),
)

funcs = [lambda x: x.GELU(), lambda x: x.LeakyReLU(0.01), None]
my_model = MLP(2, [16, 16, 1], funcs)

linear_layers = [m for m in torch_model if isinstance(m, nn.Linear)]

for layer_idx, custom_layer in enumerate(my_model.layers):
    torch_layer = linear_layers[layer_idx]

    w_matrix = []
    b_vector = []

    for neuron in custom_layer.neurons:
        w_matrix.append([w.data for w in neuron.w])
        b_vector.append(neuron.b.data)

    with torch.no_grad():
        torch_layer.weight.copy_(torch.tensor(w_matrix, dtype=torch.float32))
        torch_layer.bias.copy_(torch.tensor(b_vector, dtype=torch.float32))

X_test, y_test = make_moons(n_samples=100, noise=0.1, random_state=42)
X_test = X_test.astype(np.float32)  # type: ignore
y_test = y_test.astype(np.float32)  # type: ignore

lr = 0.01
epochs = 100

pbar = tqdm(range(1, epochs + 1), desc="Training")
for epoch in pbar:
    my_model.zero_grad()
    total_loss = Value(0.0)

    for xi, yi in zip(X_test, y_test):
        logits = my_model(xi)
        predict = logits.sigmoid()  # type: ignore
        loss_i = -(yi * predict.log() + (1 - yi) * (1 - predict).log())
        total_loss += loss_i

    total_loss = total_loss / len(X_test)
    total_loss.backward()

    torch_model.zero_grad()
    X_Torch = torch.tensor(X_test)
    y_torch = torch.tensor(y_test).unsqueeze(1)

    torch_pred = torch.sigmoid(torch_model(X_Torch))
    criterion = nn.BCELoss()
    torch_loss = criterion(torch_pred, y_torch)
    torch_loss.backward()

    max_grad_diff = 0.0

    torch_layers = [m for m in torch_model.modules() if isinstance(m, torch.nn.Linear)]

    for custom_layer, torch_layer in zip(my_model.layers, torch_layers):
        for i, neuron in enumerate(custom_layer.neurons):
            for j, w_custom in enumerate(neuron.w):
                assert torch_layer.weight.grad is not None
                torch_w_grad = torch_layer.weight.grad[i, j].item()
                diff = abs(w_custom.grad - torch_w_grad)
                max_grad_diff = max(max_grad_diff, diff)

            if torch_layer.bias is not None and torch_layer.bias.grad is not None:
                torch_b_grad = torch_layer.bias.grad[i].item()
                diff = abs(neuron.b.grad - torch_b_grad)
                max_grad_diff = max(max_grad_diff, diff)

    for p in my_model.parameters():
        p.data -= lr * p.grad

    with torch.no_grad():
        for p in torch_model.parameters():
            assert p.grad is not None
            p -= lr * p.grad

    if epoch % 10 == 0:
        pbar.set_postfix(
            {
                "Custom": f"{total_loss.data:.4f}",
                "PyTorch": f"{torch_loss.item():.4f}",
                "GradDiff": f"{max_grad_diff:.2e}",
            }
        )
