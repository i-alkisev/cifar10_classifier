import numpy as np
import torch
from data import dataset_cifar10_test, dataset_cifar10_train
from model import ConvNet


def eval_model(network, loss_fn, dataloader, device):
    """
    returns: Среднее значение функции потерь и точности по батчам
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = network(X)
            losses.append(loss_fn(preds, y).item())
            accuracies.append(
                torch.sum(preds.max(dim=1).indices == y).item() / y.shape[0]
            )
    return (np.mean(losses), np.mean(accuracies) * 100)


def training_loop(n_epochs, network, loss_fn, optimizer, dl_train, dl_test, device):
    """
    :param int n_epochs: Число итераций оптимизации
    :param torch.nn.Module network: Нейронная сеть
    :param Callable loss_fn: Функция потерь
    :param torch.nn.Optimizer optimizer: Оптимизатор
    :param torch.utils.data.DataLoader dl_train:
        Даталоадер для обучающей выборки
    :param torch.utils.data.DataLoader dl_test: Даталоадер для тестовой выборки
    :param torch.Device device: Устройство, на котором будут
        происходить вычисления
    :returns: Списки значений функции потерь и точности
        на обучающей и тестовой выборках
    """
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    for epoch in range(n_epochs):
        network.train()
        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_loss, train_accuracy = eval_model(network, loss_fn, dl_train, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_loss, test_accuracy = eval_model(network, loss_fn, dl_test, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(
                f"Epoch {epoch + 1}/{n_epochs}: "
                f"Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. "
                f"Accuracy, % (Train/Test): {train_accuracy:.2f}/"
                f"{test_accuracy:.2f}"
            )

    return train_losses, test_losses, train_accuracies, test_accuracies


def main():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"Available device: {device}")

    batch_size = 512
    dl_train = torch.utils.data.DataLoader(
        dataset_cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    dl_test = torch.utils.data.DataLoader(
        dataset_cifar10_test, batch_size=batch_size, num_workers=2
    )
    conv_network = ConvNet(use_batchnorm=True)
    conv_network.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(conv_network.parameters(), lr=2e-2)

    n_epochs = 1
    print(f"Training (n_epochs: {n_epochs})")
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=n_epochs,
        network=conv_network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dl_train=dl_train,
        dl_test=dl_test,
        device=device,
    )

    filename = "cifar10_cnn_classifier.pth"
    torch.save(conv_network.state_dict(), filename)
    print(f"Model saved in '{filename}'")


if __name__ == "__main__":
    main()
