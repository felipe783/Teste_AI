from .States import State

class Cell:
    
    def __init__(self, value=0, fixed=False): # Quando Inicializa o Objeto
        self.value = value 
        self.fixed = fixed
        self.state = State.EMPTY if value == 0 else State.FILLED
    
    def __str__(self): # toString do JAVA
        return str(self.value) if self.state == State.FILLED else " "