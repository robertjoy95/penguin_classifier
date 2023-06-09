from palmerpenguins import load_penguins
import pandas as pd

def get_penguins():
    penguins = pd.DataFrame(load_penguins())
    species = sorted(list(set(list(penguins['species']))))
    penguins = penguins.sample(frac=1).reset_index(drop=True)
    return penguins, species


if __name__ == '__main__':
    print(get_penguins())