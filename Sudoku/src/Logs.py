import json
import os

def saveResults(
    difficulty,
    numBoards,
    solved,
    impossible,
    total_quadrants,
    total_coverage,
    execution_time
):

    difficulty_names = {
        1: "facil",
        2: "medio",
        3: "dificil"
    }

    name = difficulty_names[difficulty]

    result = {
        "dificuldade": name,
        "jogos_gerados": numBoards,
        "jogos_resolvidos": solved,
        "jogos_impossiveis": impossible,
        "media_quadrantes_corretos": total_quadrants / numBoards,
        "media_cobertura": total_coverage / numBoards,
        "tempo_total": execution_time
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