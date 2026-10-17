# Sobre o Arquivo

O arquivo `leitor_pth.py` é um **Inspetor de Checkpoints**: abre um `.pth`, lista cada tensor da rede (shape, dtype, número de parâmetros e tamanho em memória) e mostra os metadados salvos junto (partidas, recorde, epsilon…).
# Funcionalidades

````python
for nome, tensor in state_dict.items():
    n_params = tensor.numel()
    n_bytes = tensor.element_size() * n_params
````

- `numel()`: total de elementos do tensor. Ex.: `linear1.weight` com shape `(256, 21)` → 5 376 parâmetros.
- `element_size()`: bytes por elemento (`float32` → 4).
- `inspecionar_outros_campos`: imprime tudo que não é `model_state_dict`, inclusive o estado do otimizador.

Uso:

````bash
python leitor_pth.py                                   # padrão: SnakeGame\Test\Models\model.pth
python leitor_pth.py SnakeGame/DQN/models/checkpoint.pth
````

> [!note] Limitações
> - O caminho padrão usa `\` (estilo Windows) e é relativo à raiz do repositório.
> - Um `state_dict` puro (como `DQN/models/model.pth`) não tem a chave `model_state_dict`. Nesse caso o script só lista as chaves disponíveis.
> - Usa `weights_only=False`, então só deve abrir checkpoints de origem confiável.
