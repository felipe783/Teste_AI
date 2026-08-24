import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Config HARDWARE
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")
LOG_DIR = os.path.join(BASE_DIR, "Log")

# Rede Neural
# nn.Module é a classe base para todas as redes neurais do PyTorch.
class Linear_QNET(nn.Module): 
    # Qntd que entra, qntd de neurônios na camada escondida, qntd que sai
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # Conceito de Heranca

        # Camadas Lineares
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size)

        for p in self.parameters(): # Pega todos os parametros e biases
            p.requires_grad_(False) # Nao ira calcula gradiente(Backpropagation)

        self.to(DEVICE) # Coloca a rede no HARDWARE 
        self.eval() 

    def forward(self, x): 
        # input --> linear1 --> ReLU --> linear2 --> output
        x = F.relu(self.linear1(x)) # input --> hidden e aplica o relu
        x = self.linear2(x) # hidden --> output
        return x 

    # Manipulação de pesos 

    # Pega todos os pesos e transforma em um unico vatot
    def get_flat_weights(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    # Pega o vetor com Tudo e coloca os valores de volta na Camada
    def set_flat_weights(self, flat: torch.Tensor):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view(p.shape))
            idx += n

    # Cria uma copia da rede 
    def clone(self) -> "Linear_QNET":

        child = Linear_QNET(
            self.linear1.in_features,
            self.linear1.out_features,
            self.linear2.out_features,
        )
        # Pega os pesos da rede atual e coloca na copia 
        # Pai --> Filho
        child.set_flat_weights(self.get_flat_weights())
        return child

    # Salvar o modelo da geracao
    def save_checkpoint(self, generation, record):
        os.makedirs(MODEL_DIR, exist_ok=True)

        filename = f"generation_{generation:03d}.pth"

        filepath = os.path.join(MODEL_DIR, filename)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "generation": generation,
            "record": record,
        }

        torch.save(checkpoint, filepath)

        best_filepath = os.path.join(MODEL_DIR, "best_genetic.pth")
        torch.save(checkpoint, best_filepath)

        print(f"Melhor Genetica da Geracao {generation}")
        print(f"Checkpoint salvo em: {filepath}")

    def load_checkpoint(self, filename="best_genetic.pth"):
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):
            print(f"Nenhum checkpoint encontrado em: {filepath}")
            return 0, 0

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])

        generation = checkpoint["generation"]
        record = checkpoint["record"]

        print("Checkpoint carregado!")
        print(f"Geração anterior: {generation}")
        print(f"Recorde anterior: {record}")

        return generation, record

    """""
    def save_generation_log(
        self,
        generation,
        best_score,
        best_fitness,
        average_score,
        average_fitness,
        population_size,
        mutation_rate,
        mutation_strength,
        elapsed_time,
        population
    ):

        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, "history.json")

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                history = json.load(file)
        else:
            history = []

        # Desempenho de cada indivíduo
        individuals_data = []

        for i, agent in enumerate(population):
            individuals_data.append({
                "id": i,
                "score": agent.score,
                "fitness": agent.fitness
            })

        # Dados da geração
        generation_data = {
            "generation": generation,
            "best_score": best_score,
            "best_fitness": best_fitness,
            "average_score": average_score,
            "average_fitness": average_fitness,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
            "elapsed_time": elapsed_time,
            "individuals": individuals_data
        }

        history.append(generation_data)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)
        """