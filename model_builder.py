import torch
from torch import nn


# 1. Construct a model class that subclasses nn.Module
class PenguinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=5, out_features=100) # takes in 4 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=100, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=10)
        self.layer_4 = nn.Linear(in_features=10, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))) # computation goes through layer_1 first then the output of layer_1 goes through layer_2 ect.
    
if __name__ == '__main__':
    from data_setup import get_data
    train_X, test_X, train_y, test_y, species = get_data()
    model = PenguinClassifier()
    model.train()
    y_logits = model(train_X)
    print(y_logits)
