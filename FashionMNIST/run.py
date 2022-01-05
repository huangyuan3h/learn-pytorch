from FashionMNIST.NeuralNetwork import model
from FashionMNIST.data import train_dataloader, test_dataloader
from FashionMNIST.training import train, test, loss_fn, optimizer

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")