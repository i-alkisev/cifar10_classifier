import torch
from model import ConvNet
from train import eval_model, training_loop
from data import dataset_cifar10_train, dataset_cifar10_test


if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    print(f'Available device: {device}')
    
    batch_size = 512
    dl_train = torch.utils.data.DataLoader(dataset_cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_test = torch.utils.data.DataLoader(dataset_cifar10_test, batch_size=batch_size, num_workers=2)
    conv_network = ConvNet(use_batchnorm=True)
    conv_network.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(conv_network.parameters(), lr=2e-2)

    n_epochs = 2
    print(f'Training (n_epochs: {n_epochs})')
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=n_epochs, network=conv_network, loss_fn=loss_fn, 
        optimizer=optimizer, dl_train=dl_train, dl_test=dl_test, device=device
    )

    print(f'Evaluating model')
    train_loss, train_accuracy = eval_model(conv_network, loss_fn, dl_train, device)
    test_loss, test_accuracy = eval_model(conv_network, loss_fn, dl_test, device)
    print(f'Finally: Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. Accuracy, % (Train/Test): {train_accuracy:.2f}/{test_accuracy:.2f}')

    filename = 'cifar10_cnn_classifier.pth'
    torch.save(conv_network.state_dict(), filename)
    print(f'Model saved in \'{filename}\'')
