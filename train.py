from __future__ import print_function
import os
import numpy as np
import h5py
import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from model import ILModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', '-data_file', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\gs_4images_nocov\\gs_4images_nocov.h5', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\he_policy\\il_policy_torch_relu', type=str)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to train on', default=50000, type=int)
parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=30, type=int)
args = parser.parse_args()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    counter = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long().squeeze())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        counter += 1

    train_loss /= counter

    print('Epoch: {} Train Loss: {:.6f}'.format(epoch, train_loss))

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.long().squeeze()).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            counter += 1

    test_loss /= counter

    print('Test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_dataset(dset_file, num_imgs, batch_size):

    # load h5 file
    dataset_dict = h5py.File(dset_file, 'r')

    # get dataset as numpy
    images_dataset = np.asarray(dataset_dict['images'], dtype=np.float32)
    labels_dataset = np.asarray(dataset_dict['labels'], dtype=np.int)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_dataset.shape[0])
    images_dataset = images_dataset[p]
    labels_dataset = labels_dataset[p]

    # NHWC => NCHW
    images_dataset = images_dataset.transpose(0,3,1,2)

    # trim data if asked to
    if images_dataset.shape[0] > num_imgs:
        images_dataset = images_dataset[:num_imgs,:,:,:]
        labels_dataset = labels_dataset[:num_imgs,:]

    # convert to torch tensors
    images_dataset = torch.tensor(images_dataset)
    labels_dataset = torch.tensor(labels_dataset)

    # make train and test datasets
    test_split = int(images_dataset.shape[0] * 0.1)
    train_dataset = TensorDataset(images_dataset[:-test_split,:,:,:],labels_dataset[:-test_split,:])
    test_dataset = TensorDataset(images_dataset[-test_split:,:,:,:],labels_dataset[-test_split:,:])

    return train_dataset, test_dataset

if __name__ == "__main__":

    # define gpu as device
    device = torch.device("cuda")

    # get dataset and make dataloaders
    train_dataset, test_dataset = load_dataset(args.data_file, args.num_imgs, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    # check if output folder exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # get model, loss and optimizer
    model = ILModel(num_actions=args.num_actions).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, criterion, test_loader)

        if epoch % 5 == 0:
            torch.save(model.state_dict(),os.path.join(args.output_dir, "model_{}.pt".format(epoch)))
