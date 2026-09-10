import json
import os

def saveResults(
    difficulty,
    resolve,
    numBoards,
    solved,
    impossible,
    total_quadrants,
    total_coverage,
    limit_attempts,
    total_attempts,
    execution_time
):

    difficulty_names = {
        1: "facil",
        2: "medio",
        3: "dificil"
    }
    resolve_name = {
        1: "Probabilidade sem Tentativa",
        2: "Probabilidade com Tentativa",
        3: "BackTraking" 
    }

    type = resolve_name[resolve]
    name = difficulty_names[difficulty]

    result = {
    "dificuldade": name,
    "tipo_resolucao": type,
    "jogos_gerados": numBoards,
    "jogos_resolvidos": solved,
    "jogos_impossiveis": impossible,
    "media_quadrantes_corretos": round(total_quadrants / numBoards, 2),
    "media_cobertura": round(total_coverage / numBoards,2),
    "total_tentativas": total_attempts,
    "limite_tentativas_por_game": limit_attempts,
    "tempo_total": round(execution_time/60, 2)
    }

    file_path = "results.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    test_number = len(data) + 1

    data[f"teste_{test_number}"] = result

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)