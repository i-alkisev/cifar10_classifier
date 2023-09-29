import torch
import numpy as np


def eval_model(network, loss_fn, dataloader, device):
    '''
    returns: Среднее значение функции потерь и точности по батчам 
    '''
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = network(X)
            losses.append(loss_fn(preds, y).item())
            accuracies.append(torch.sum(preds.max(dim=1).indices == y).item() / y.shape[0])
    return (np.mean(losses), np.mean(accuracies) * 100)


def training_loop(n_epochs, network, loss_fn, optimizer, dl_train, dl_test, device):
    '''
    :param int n_epochs: Число итераций оптимизации
    :param torch.nn.Module network: Нейронная сеть
    :param Callable loss_fn: Функция потерь
    :param torch.nn.Optimizer optimizer: Оптимизатор
    :param torch.utils.data.DataLoader dl_train: Даталоадер для обучающей выборки
    :param torch.utils.data.DataLoader dl_test: Даталоадер для тестовой выборки
    :param torch.Device device: Устройство на котором будут происходить вычисления
    :returns: Списки значений функции потерь и точности на обучающей и тестовой выборках после каждой итерации
    '''
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
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

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            train_loss, train_accuracy = eval_model(network, loss_fn, dl_train, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_loss, test_accuracy = eval_model(network, loss_fn, dl_test, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f'Epoch {epoch + 1}/{n_epochs}: Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. Accuracy, % (Train/Test): {train_accuracy:.2f}/{test_accuracy:.2f}')

    return train_losses, test_losses, train_accuracies, test_accuracies
