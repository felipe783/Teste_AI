import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ============================================================
# CPU — configuração específica para o contexto GENÉTICO
# ============================================================
#
# Diferente do DQN (que processa batches grandes e se beneficia de
# várias threads), aqui cada avaliação é um forward pass de UM único
# estado por vez. Multithreading nesse tamanho de tensor só adiciona
# overhead — e se você paralelizar a população com multiprocessing,
# múltiplos processos com várias threads cada causam oversubscription
# de CPU. Por isso: 1 thread.

DEVICE = torch.device("cpu")
torch.set_num_threads(1)


# ============================================================
# DIRETÓRIOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# REDE (arquitetura idêntica ao DQN — só o "redor" muda)
# ============================================================

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Nunca há backward no genético: desliga o autograd na raiz.
        # Evita overhead de construção de grafo mesmo que algum forward
        # seja chamado fora de um bloco torch.no_grad() por engano.
        for p in self.parameters():
            p.requires_grad_(False)

        self.to(DEVICE)
        self.eval()  # sem dropout/batchnorm aqui, mas é hábito correto

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # ========================================================
    # MANIPULAÇÃO DE PESOS (crossover / mutação)
    # ========================================================

    def get_flat_weights(self) -> torch.Tensor:
        """Retorna todos os pesos concatenados em um único vetor 1D."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_weights(self, flat: torch.Tensor):
        """Recarrega um vetor 1D de volta na estrutura da rede."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view(p.shape))
            idx += n

    def clone(self) -> "Linear_QNet":
        child = Linear_QNet(
            self.linear1.in_features,
            self.linear1.out_features,
            self.linear2.out_features,
        )
        child.set_flat_weights(self.get_flat_weights())
        return child

    # ========================================================
    # CHECKPOINT (sem otimizador — genoma não tem estado de treino)
    # ========================================================

    def save_checkpoint(self, generation, record, file_name="best_genetic.pth"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        file_path = os.path.join(MODEL_DIR, file_name)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "generation": generation,
            "record": record,
        }

        torch.save(checkpoint, file_path)
        print(f"Checkpoint salvo em: {file_path}")

    def load_checkpoint(self, file_name="best_genetic.pth"):
        file_path = os.path.join(MODEL_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Nenhum checkpoint encontrado em: {file_path}")
            return 0, 0

        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])

        generation = checkpoint["generation"]
        record = checkpoint["record"]

        print("Checkpoint carregado!")
        print(f"Geração anterior: {generation}")
        print(f"Recorde anterior: {record}")

        return generation, record
