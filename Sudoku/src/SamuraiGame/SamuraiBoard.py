class Cor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CIANO = "\033[96m"
    VERDE = "\033[92m"
    VERMELHO = "\033[91m"
    CINZA = "\033[90m"

def showBoardSamurai(board, original):
    print()
    for r in range(len(board)):
        linha_str = ""
        tem_conteudo_na_linha = False
        
        for c in range(len(board[r])):
            val = board[r][c]
            is_orig = original[r][c]

            if val is not None:
                tem_conteudo_na_linha = True

            # Formatação do caractere e cor
            if val is None:
                char = " "  # Célula fora do grid de jogo
            elif val == 0:
                char = f"{Cor.CINZA}.{Cor.RESET}"
            elif is_orig:
                char = f"{Cor.CIANO}{Cor.BOLD}{val}{Cor.RESET}"
            else:
                char = f"{Cor.VERDE}{val}{Cor.RESET}"

            linha_str += char + " "

            # Espaçamento vertical para delimitar blocos 3x3 de forma agradável
            if c % 3 == 2 and val is not None:
                linha_str += " "
            elif c % 3 == 2 and val is None:
                linha_str += " "

        if tem_conteudo_na_linha:
            print(linha_str)
            # Espaçamento horizontal para delimitar blocos 3x3
            if r % 3 == 2:
                print()