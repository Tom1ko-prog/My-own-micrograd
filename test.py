import numpy as np
import torch
from torch import nn

from nn import MLP
from value import Value

torch_model = nn.Sequential(
    nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16), nn.LeakyReLU(0.01), nn.Linear(16, 1)
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
        torch_layer.weight.copy_(torch.tensor(w_matrix))
        torch_layer.bias.copy_(torch.tensor(b_vector))


X_test = np.random.randn(16, 2)
y_test = np.random.randint(0, 2, size=(16,))


torch_model.zero_grad()
X_Torch = torch.tensor(X_test, dtype=torch.float32)
y_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

torch_logits = torch_model(X_Torch)
torch_pred = torch.sigmoid(torch_logits)
criterion = nn.BCELoss()
torch_loss = criterion(torch_pred, y_torch)
torch_loss.backward()


my_model.zero_grad()
total_loss = Value(0.0)

for xi, yi in zip(X_test, y_test):
    logits = my_model(xi)
    predict = logits.sigmoid()  # type: ignore
    loss_i = -(yi * predict.log() + (1 - yi) * (1 - predict).log())
    total_loss += loss_i

custom_loss = total_loss / len(X_test)
custom_loss.backward()

loss_diff = abs(custom_loss.data - torch_loss.item())
print(f"Custom Loss:  {custom_loss.data:.8f}")
print(f"PyTorch Loss: {torch_loss.item():.8f}")
print(f"Разница в Loss: {loss_diff:.8e}\n")

max_grad_diff = 0.0
for layer_idx, custom_layer in enumerate(custom_model.layers):
    torch_layer = linear_layers[layer_idx]

    # Сравниваем градиенты весов W
    for i, neuron in enumerate(custom_layer.neurons):
        for j, w in enumerate(neuron.w):
            torch_grad = torch_layer.weight.grad[i, j].item()
            diff = abs(w.grad - torch_grad)
            max_grad_diff = max(max_grad_diff, diff)

        # Сравниваем градиенты сдвигов B
        torch_b_grad = torch_layer.bias.grad[i].item()
        diff_b = abs(neuron.b.grad - torch_b_grad)
        max_grad_diff = max(max_grad_diff, diff_b)

print(f"Максимальное расхождение градиентов: {max_grad_diff:.8e}")
