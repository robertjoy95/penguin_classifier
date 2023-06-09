import torch
from model_builder import PenguinClassifier
from engine import train
from data_setup import get_data


train_X, test_X, train_y, test_y, species = get_data()
model = PenguinClassifier()
# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)
n_epochs = 300

results = train(model, 
    train_X,
    test_X,
    train_y,
    test_y, 
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=n_epochs,
    device='cpu')
