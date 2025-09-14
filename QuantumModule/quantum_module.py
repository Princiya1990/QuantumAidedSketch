import numpy as np
import pandas as pd
from qiskit import Aer
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.connectors import TorchConnector
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

df = pd.read_csv("features.csv")
X = df.drop(columns=["filename", "label"]).values
y = df["label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_qubits = 10   # 7 amplitude + 3 angle encoding

def build_hybrid_circuit():
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    for i in range(7):
        qc.h(qr[i])   

    qc.ry(np.pi/4, qr[7])
    qc.ry(np.pi/3, qr[8])
    qc.ry(np.pi/6, qr[9])
    ansatz = RealAmplitudes(num_qubits, reps=2)
    qc.compose(ansatz, inplace=True)

    return qc

feature_map = build_hybrid_circuit()
ansatz = RealAmplitudes(num_qubits, reps=2)


backend = Aer.get_backend("aer_simulator_statevector")
qi = QuantumInstance(backend)

vqc = VQC(
    feature_map=ansatz,     
    ansatz=ansatz,
    optimizer=None,
    quantum_instance=qi
)

torch_model = TorchConnector(vqc)

class HybridNet(nn.Module):
    def __init__(self, quantum_model, n_classes=5):
        super().__init__()
        self.q_layer = quantum_model
        self.fc = nn.Linear(quantum_model.output_shape[0], n_classes)

    def forward(self, x):
        out = self.q_layer(x)
        out = self.fc(out)
        return out

model = HybridNet(torch_model, n_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_torch = torch.tensor(X_train[:, :num_qubits], dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)

for epoch in range(100):   
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

X_test_torch = torch.tensor(X_test[:, :num_qubits], dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

with torch.no_grad():
    preds = model(X_test_torch).argmax(dim=1)
    acc = (preds == y_test_torch).sum().item() / len(y_test_torch)
    print("✅ Test Accuracy:", acc)
