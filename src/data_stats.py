import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "preProcessedData", "with_T", "dp11")

fields = {
    "press":  "press.csv",
    "temp":   "temp.csv",
    "vel_x":  "vel_x.csv",
    "vel_y":  "vel_y.csv",
    "vel_z":  "vel_z.csv",
}

print(f"{'Field':<10} {'Min':>20} {'Max':>20}")
print("-" * 52)

for name, filename in fields.items():
    path = os.path.join(DATA_DIR, filename)
    data = np.loadtxt(path, delimiter=",", usecols=3)
    print(f"{name:<10} {data.min():>20.6e} {data.max():>20.6e}")
