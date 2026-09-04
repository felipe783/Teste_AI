from datetime import datetime
import json
from pathlib import Path
import re

import cv2
import pygame
import torch
import numpy as np
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, DEVICE

INPUT_SIZE = 21
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3
VIDEO_DIR = Path(__file__).resolve().parent / "video"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "testes.json"
VIDEO_FPS = 40

print("------------\n")
print("1 -- Modelo treinado por Recompensa")
print("2 -- Modelo treinado por Geracoes\n")
print("------------")
escolha = int(input("Numero: "))

if escolha == 1:
    MODEL_PATH = "Models/model.pth"
elif escolha == 2:
    MODEL_PATH = "Models/best_genetic.pth"
else:
    print("Escolhe certo")
    exit()

while True:
    try:
        TOTAL_GAMES = int(input("Quantos jogos a IA deve jogar? "))
        if TOTAL_GAMES > 0:
            break
    except ValueError:
        pass
    print("Informe um numero inteiro maior que zero.")

def get_state(game):
        head, tail = game.head, game.snake[-1]
        actions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        analyses = [game.action_analysis(action) for action in actions]
        board_width, board_height = max(1, game.w - BLOCK_SIZE), max(1, game.h - BLOCK_SIZE)
        capacity = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)

        state = [
            *(game.direction == direction for direction in
              (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)),
            (game.food.x - head.x) / board_width, (game.food.y - head.y) / board_height,
            (tail.x - head.x) / board_width, (tail.y - head.y) / board_height,
            *(item[0] for item in analyses),       # colisão: frente, direita, esquerda
            *(item[1] for item in analyses),       # fração de espaço acessível
            *(item[2] for item in analyses),       # rota segura até a comida
            *(item[3] for item in analyses),       # rota até a cauda/ciclo de escape
            len(game.snake) / capacity,
        ]
        return np.asarray(state, dtype=np.float32)



model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


def obter_recorde_anterior():
    """Recupera o maior score registrado em videos e logs de testes anteriores."""
    VIDEO_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    recorde = -1

    for video in VIDEO_DIR.glob("*_score_*.mp4"):
        resultado = re.search(r"_score_(\d+)\.mp4$", video.name)
        if resultado:
            recorde = max(recorde, int(resultado.group(1)))

    for teste in carregar_testes().values():
        pontuacao = teste.get("recorde_final") if isinstance(teste, dict) else None
        if isinstance(pontuacao, int):
            recorde = max(recorde, pontuacao)

    # Mantem compatibilidade com logs criados pela versao anterior do programa.
    for log in LOG_DIR.glob("teste_*.json"):
        try:
            with log.open(encoding="utf-8") as arquivo:
                pontuacao = json.load(arquivo).get("recorde_final")
            if isinstance(pontuacao, int):
                recorde = max(recorde, pontuacao)
        except (json.JSONDecodeError, OSError):
            continue

    return recorde


def carregar_testes():
    """Retorna o historico indexado por teste salvo no JSON unico."""
    LOG_DIR.mkdir(exist_ok=True)

    if not LOG_FILE.exists():
        return []

    try:
        with LOG_FILE.open(encoding="utf-8") as arquivo:
            testes = json.load(arquivo)
        if not isinstance(testes, dict):
            return {}

        # Converte automaticamente o formato antigo {"testes": [...]}.
        if set(testes) == {"testes"} and isinstance(testes["testes"], list):
            return {
                f"teste_{indice}": teste
                for indice, teste in enumerate(testes["testes"], start=1)
            }

        return testes
    except (json.JSONDecodeError, OSError):
        return {}


def criar_gravador(game):
    """Cria um video temporario para a partida atual."""
    VIDEO_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    caminho = VIDEO_DIR / f".temporario_{timestamp}.mp4"
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(str(caminho), codec, VIDEO_FPS, (game.w, game.h))

    if not gravador.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video: {caminho}")

    print("Gravando partida temporariamente...")
    return gravador, caminho


def finalizar_gravacao(gravador, caminho, score, salvar):
    """Salva o video apenas se a partida superou o recorde anterior."""
    gravador.release()

    if not salvar:
        caminho.unlink(missing_ok=True)
        return None

    timestamp = caminho.stem.removeprefix(".temporario_")
    caminho_final = VIDEO_DIR / f"snake_{timestamp}_score_{score}.mp4"
    caminho.rename(caminho_final)
    print(f"Novo recorde! Video salvo em: {caminho_final}")
    return caminho_final


def gravar_frame(gravador, game):
    """Copia o frame mostrado pelo Pygame para o arquivo de video."""
    frame_rgb = pygame.surfarray.array3d(game.display)
    frame_bgr = cv2.cvtColor(np.transpose(frame_rgb, (1, 0, 2)), cv2.COLOR_RGB2BGR)
    gravador.write(frame_bgr)


def salvar_log(inicio, scores, recorde_anterior, recorde_final, videos_salvos):
    """Acrescenta o resumo deste teste ao arquivo JSON de historico."""
    LOG_DIR.mkdir(exist_ok=True)
    log = {
        "data_inicio": inicio.isoformat(timespec="seconds"),
        "modelo": MODEL_PATH,
        "jogos_solicitados": TOTAL_GAMES,
        "jogos_rodados": len(scores),
        "pontuacao_media": round(sum(scores) / len(scores), 2) if scores else None,
        "menor_pontuacao": min(scores) if scores else None,
        "maior_pontuacao": max(scores) if scores else None,
        "recorde_anterior": recorde_anterior if recorde_anterior >= 0 else None,
        "recorde_final": recorde_final if recorde_final >= 0 else None,
        "videos_salvos": [str(video.name) for video in videos_salvos],
    }
    testes = carregar_testes()
    numeros_existentes = [
        int(chave.removeprefix("teste_"))
        for chave in testes
        if chave.startswith("teste_") and chave.removeprefix("teste_").isdigit()
    ]
    proximo_teste = max(numeros_existentes, default=0) + 1
    testes[f"teste_{proximo_teste}"] = log

    with LOG_FILE.open("w", encoding="utf-8") as arquivo:
        json.dump(testes, arquivo, ensure_ascii=False, indent=2)
    print(f"Resumo salvo como teste_{proximo_teste} em: {LOG_FILE}")


game = SnakeGameAI()
inicio_teste = datetime.now()
recorde_anterior = obter_recorde_anterior()
recorde_atual = recorde_anterior
scores = []
videos_salvos = []
gravador = None
caminho_video = None

try:
    for partida in range(1, TOTAL_GAMES + 1):
        game.reset()
        gravador, caminho_video = criar_gravador(game)
        game_over = False

        while not game_over:
            state = get_state(game)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(state_tensor)

            action_index = torch.argmax(prediction).item()

            action = [0, 0, 0]
            action[action_index] = 1

            _, game_over, score = game.play_step(action)
            gravar_frame(gravador, game)

        scores.append(score)
        superou_recorde = score > recorde_atual
        video_salvo = finalizar_gravacao(gravador, caminho_video, score, superou_recorde)
        gravador, caminho_video = None, None

        if superou_recorde:
            recorde_atual = score
            videos_salvos.append(video_salvo)

        print(f"Jogo {partida}/{TOTAL_GAMES} finalizado. Score: {score}")
except KeyboardInterrupt:
    print("Teste Encerrado")
finally:
    if gravador is not None:
        gravador.release()
        caminho_video.unlink(missing_ok=True)
    salvar_log(inicio_teste, scores, recorde_anterior, recorde_atual, videos_salvos)
    pygame.quit()
