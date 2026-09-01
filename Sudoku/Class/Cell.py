from .States import State

class Cell:
    def __init__(self, value=0): # Quando Inicializa o Objeto
        self.value = value 

        if value == 0:
            self.state = State.EMPTY
        else:
            self.state = State.FILLED

    def __str__(self): # toString do JAVA
        return str(self.value) if self.state == State.FILLED else " "