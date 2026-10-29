# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Entradas
x = torch.tensor([ 
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32) 

# Saídas
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        # Camadas --> 2 Entradas e 4 Neurônios 
        self.linearHidden = nn.Linear(2, 4)

        # Ativacao
        self.activation = nn.Sigmoid()

        # Camada Saida
        self.linearOutput = nn.Linear(4, 1)

    def forward(self, x): # x e o neuronio que esta sendo ativado
        x = self.linearHidden(x)
        x = self.activation(x)
        x = self.linearOutput(x)
        x = self.activation(x)
        return x

modelo = QNet()

criterioPerda = nn.MSELoss()

otimizador = optim.Adam(modelo.parameters(), lr=0.1)

epocas = 1000
historicoPerda = []

print("Treinando a rede neural")
for epoca in range(epocas):
    saida = modelo(x)
    perda = criterioPerda(saida, y)
    historicoPerda.append(perda.item())

    otimizador.zero_grad() # Limpa os gradientes 
    perda.backward() # Regra de cadeia
    otimizador.step() # Atualizar Bias

    if (epoca + 1) % 200 == 0:
        print(f"Época [{epoca + 1}/{epocas}], Perda: {perda.item():.6f}")   

print("Treinamento concluído!")


print("Testando a rede neural")

modelo.eval() # Modo de avaliação
with torch.no_grad():
    saidasFinal = modelo(x)
    for i in range(len(x)):
        entrada = x[i].tolist()
        esperada = int(y[i].item())
        previstoContinuo = saidasFinal[i].item()
        preivsao = 1 if previstoContinuo >= 0.5 else 0
        print(f"Entrada: {entrada}--> Esperado: {esperada} | Previsto: {previstoContinuo:.4f} | Classificação: {preivsao}")

plt.figure(figsize=(10, 5))
plt.plot(historicoPerda, color='#2563EB', linewidth=2, label='Perda (MSE)')
plt.title('Evolução da Perda durante o Treinamento', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=11)
plt.ylabel('Erro (MSE)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
