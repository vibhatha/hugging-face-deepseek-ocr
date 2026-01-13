class Agent:
    def __init__(self, name: str):
        self.name = name

    def log(self, message: str):
        print(f"[{self.name}] {message}")
