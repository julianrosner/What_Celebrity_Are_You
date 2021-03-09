# Python Module TrainUtil
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# returns dict where 'train' maps to a dataloader on the training set, 'test' is
# another on the test set, and 'classes' is a list of possible classes
def get_celeba_data(trainroot, testroot, image_size, batch_size, num_workers, num_classes):
    # train root is a directory of directory of images
    # where each inner directory represents a class
    trainset = dset.ImageFolder(root=trainroot,
                        transform=transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        #transforms.RandomHorizontalFlip(),
                        #transforms.RandomCrop(image_size, padding=2, padding_mode='edge'),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

    # create trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)

    testset = dset.ImageFolder(root=testroot,
                            transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    # create testloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
    return {'train': trainloader, 'test': testloader, 'classes': range(num_classes)}

# trains network according to provided hyperparameters
def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.05, verbose=1):
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    for epoch in range(epochs):
        print("epoch: " + str(epoch + 1))
        sum_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0], batch[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize 
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                if verbose:
                   print('[%d, %5d] loss: %.3f' %
                         (epoch + 1, i + 1, sum_loss / 100))
                   sum_loss = 0.0
    return losses

# computes accuracy of network
def accuracy(net, dataloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
      for batch in dataloader:
        images, labels = batch[0], batch[1]
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct/total