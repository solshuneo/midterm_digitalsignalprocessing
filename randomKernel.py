import numpy as np

kernel = np.random.randint(-10, 11, size=20)

with open("kernel.txt", "w") as f:
    f.write(" ".join(map(str, kernel)))

