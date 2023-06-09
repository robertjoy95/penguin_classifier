from get_penguins import get_penguins
import pandas as pd
import torch

def transform(data: pd.DataFrame):
    # transform the data into tensors
    torch_tensor = torch.tensor(data.values)
    print(torch_tensor)
    return torch_tensor

def translate_to_index(df, col_name, list_items):
    # translate a given list of string items into integers
    drop_rows = []
    for i, item in enumerate(df[col_name]):
        # translate to integers for the classifier
        if str(item).lower() == "nan":
            drop_rows.append(i)
            continue
        df.at[i, col_name] = list_items.index(item)
    return df.drop(drop_rows)


def get_data():
    penguins, species = get_penguins()
    islands = sorted(list(set(list(penguins['island']))))
    # translate string data to integer classifications
    penguins = translate_to_index(penguins, "species", species)
    penguins = translate_to_index(penguins, "island", islands)
    species_list = pd.DataFrame().assign(species=penguins['species'])
    drop_rows = []
    used_cols = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for col in penguins.columns:
        if col not in used_cols:
            penguins = penguins.drop(col, axis=1)
    for col_name in used_cols:
        # normalize data
        if col_name == 'island':
            continue
        max_val = max(list(penguins[col_name]))
        min_val = min(list(penguins[col_name]))
        for i, item in enumerate(penguins[col_name]):
            if str(item).lower() == "nan":
                drop_rows.append(i)
                continue 
            penguins.at[i, col_name] = (item - min_val)/(max_val - min_val)
    penguins = penguins.drop(drop_rows).reset_index()
    species_list = species_list.drop(drop_rows).reset_index()
    penguins = penguins.drop("index", axis=1)
    penguins_tensor = transform(penguins.astype(float)).type(torch.float)
    species_tensor = torch.tensor(species_list['species']).type(torch.float)

    # get a test/train split
    split_val = int(len(penguins["bill_length_mm"])*0.8)
    train_data = penguins_tensor[:split_val]
    test_data = penguins_tensor[split_val:]

    train_y = species_tensor[:split_val]
    test_y = species_tensor[split_val:]

    return train_data, test_data, train_y, test_y, species


if __name__ == '__main__':
    print(get_data())