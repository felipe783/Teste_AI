# Sobre o Arquivo

O arquivo `model.py` define a `Linear_QNET` do GA: a mesma arquitetura `entrada → 256 → 3` do DQN, adaptada para **manipulação direta dos pesos**.
# Funcionalidades

## Sem gradiente

````python
for p in self.parameters():
    p.requires_grad_(False)
self.to(DEVICE)
self.eval()
````

Como os pesos mudam por mutação e crossover, não há *backpropagation*: desligar o gradiente economiza memória e tempo.

## get_flat_weights e set_flat_weights

- `get_flat_weights`: concatena todos os tensores (`linear1.weight`, `linear1.bias`, …) num único vetor 1D. Esse vetor é o **genoma**.
- `set_flat_weights`: percorre os parâmetros e copia de volta a fatia correspondente, com `view(p.shape)`.

Para 15 entradas: `15×256 + 256 + 256×3 + 3 = 4 867` genes.

## clone

Cria uma nova rede com as mesmas dimensões e copia o genoma. São os mesmos pesos em objetos independentes, o que é necessário para o elitismo.

## Checkpoint

- `save_checkpoint` grava `GA/Models/generation_XXX.pth` e sobrescreve `GA/Models/best_genetic.pth` (caminho baseado em `__file__`).
- `load_checkpoint` lê `best_genetic.pth` e retorna `(generation, record)`.

> [!note] Constante não usada
> `LOG_DIR = BASE_DIR/"Log"` é definida aqui, mas o log real usa `LOG_DIR = "logs"`, de [[Agente Genético]].
