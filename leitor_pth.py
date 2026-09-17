import sys
import torch

CHECKPOINT_PATH = r"SnakeGame\Test\Models\model.pth"


def format_bytes(n_bytes: float) -> str:
    """Converte bytes para uma unidade legível (KB, MB, GB...)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} TB"


def print_header(titulo: str) -> None:
    largura = 90
    print("\n" + "=" * largura)
    print(titulo.center(largura))
    print("=" * largura)


def inspecionar_state_dict(state_dict: dict) -> None:
    print_header("PARÂMETROS DO MODELO")

    header = f"{'Camada':35} {'Shape':20} {'Dtype':12} {'Params':>12} {'Tamanho':>12}"
    print(header)
    print("-" * len(header))

    total_params = 0
    total_bytes = 0

    for nome, tensor in state_dict.items():
        n_params = tensor.numel()
        n_bytes = tensor.element_size() * n_params
        total_params += n_params
        total_bytes += n_bytes

        print(
            f"{nome:35} "
            f"{str(tuple(tensor.shape)):20} "
            f"{str(tensor.dtype):12} "
            f"{n_params:>12,} "
            f"{format_bytes(n_bytes):>12}"
        )

    print("-" * len(header))
    print(f"{'TOTAL':35} {'':20} {'':12} {total_params:>12,} {format_bytes(total_bytes):>12}")


def inspecionar_outros_campos(checkpoint: dict) -> None:
    outros = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    if not outros:
        return

    print_header("OUTROS DADOS DO CHECKPOINT")

    for chave, valor in outros.items():
        if isinstance(valor, dict):
            print(f"\n[{chave}] (dict com {len(valor)} chaves)")
            for sub_k, sub_v in valor.items():
                if hasattr(sub_v, "shape"):
                    print(f"    {sub_k:30} shape={tuple(sub_v.shape)} dtype={sub_v.dtype}")
                else:
                    print(f"    {sub_k:30} = {sub_v}")
        elif hasattr(valor, "shape"):
            print(f"{chave:20} shape={tuple(valor.shape)} dtype={valor.dtype}")
        else:
            print(f"{chave:20} = {valor}")


def main() -> None:
    caminho = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT_PATH

    print(f"Carregando checkpoint: {caminho}")
    checkpoint = torch.load(caminho, map_location="cpu", weights_only=False)

    if "model_state_dict" not in checkpoint:
        print("Aviso: chave 'model_state_dict' não encontrada. Chaves disponíveis:")
        print(list(checkpoint.keys()))
        return

    state_dict = checkpoint["model_state_dict"]

    inspecionar_state_dict(state_dict)
    inspecionar_outros_campos(checkpoint)

    print_header("FIM")


if __name__ == "__main__":
    main()