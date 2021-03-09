import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from  CelebrityConvNet import CelebrityConvNet
from TrainUtil import get_celeba_data
from TrainUtil import train
from TrainUtil import accuracy

if __name__ == '__main__':
    # 'SMALL', 'MEDIUM', or 'LARGE'
    DATA_SET_SIZE = 'SMALL'
    SHOW_BATCH = False
    SEED = False

    # misc. parameters
    num_workers = 4
    image_size = 64

    # set paths for saving information 
    fig_path = '../figs/' + DATA_SET_SIZE.lower() + '.jpg'
    nn_path = '../nns/' + DATA_SET_SIZE.lower() + '.nn'

    # seed rng if desired
    if SEED:
        manualSeed = 123
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    # decide which subset of CelebA to train on
    if DATA_SET_SIZE == 'SMALL':
        trainroot = '../data/small_known/train'
        testroot = '../data/small_known/test'
        num_classes = 100
        batch_size = 32
        print("Using small dataset...")
       
    elif DATA_SET_SIZE == 'LARGE':
        trainroot = '../data/large_known/train'
        testroot = '../data/large_known/test'
        num_classes = 10177
        batch_size = 256
        print("Using large dataset...")
       
    else:
        trainroot = '../data/medium_known/train'
        testroot = '../data/medium_known/test'
        num_classes = 1000
        batch_size = 128
        print("Using medium dataset...")
 
    data = get_celeba_data(trainroot, testroot, image_size, batch_size, num_workers, num_classes)

    # print representative data sample with labels
    if SHOW_BATCH:
        dataiter = iter(data['train'])
        images, labels = dataiter.next()

        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        
        # print labels
        print("" + ' '.join('%9s' % data['classes'][labels[j]] for j in range(batch_size)))
        # show images
        imshow(torchvision.utils.make_grid(images))
        flat = torch.flatten(images, 1)

    # train model
    cel_net = CelebrityConvNet(image_size, num_classes)
    conv_losses = train(cel_net, data['train'], epochs=20, lr=0.01, decay=0.05)
    conv_losses+= train(cel_net, data['train'], epochs=10, lr=0.002, decay=0.05)
    conv_losses+= train(cel_net, data['train'], epochs=6, lr=0.0002, decay=0.05)
    conv_losses+= train(cel_net, data['train'], epochs=6, lr=0.00002, decay=0.05)
    print(conv_losses)

    # compute accuracy
    print("Training accuracy: %f" % accuracy(cel_net, data['train']))
    print("Testing  accuracy: %f" % accuracy(cel_net, data['test']))

    # create informative plot
    plt.plot(conv_losses)
    plt.title("Loss for " + DATA_SET_SIZE.capitalize() + " Dataset")
    plt.xlabel('Batch')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(fig_path)
    plt.show()

    # save model to nn_path
    torch.save(cel_net.state_dict(), nn_path)