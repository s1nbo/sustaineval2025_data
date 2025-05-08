import os

class Test:
    def __init__(self):
        self.name = "Test"
        self.value = 42
        self.result_path = './Result'

    def display(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.result_path, exist_ok=True)

        # Open the file (create it if it doesn't exist or overwrite if it does)
        with open(f"{self.result_path}/test.txt", "w") as f:
            f.write(f"{self.name}: {self.value}\n")

test = Test()
test.display()
