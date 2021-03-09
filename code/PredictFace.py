import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpim
from  CelebrityConvNet import CelebrityConvNet

if __name__ == '__main__':
    # path to folder where results will be saved
    fig_path = '../figs'

    # a folder containg a folder containg the images for which predictions are desired
    predict_folder_folder_path = '../data/predict'

    # folder containing the images for which predictions are desired
    predict_folder_path = '../data/predict/to_predict'

    # set parameters
    DATA_SET_SIZE = 'MEDIUM'
    image_size = 64

    # select proper saved nn and celebrity face folder paths given dataset choice
    if DATA_SET_SIZE == 'SMALL':
        num_classes = 100
        nn_path = '../nns/small.nn'
        celeb_face_path = '../data/small_known/train'
    elif DATA_SET_SIZE == 'LARGE':
        num_classes = 10177
        nn_path = '../nns/large.nn'
        celeb_face_path = '../data/large_known/train'
    else:
        num_classes = 1000
        nn_path = '../nns/medium.nn'
        celeb_face_path = '../data/medium_known/train'

    # obtain file names for pictures which are to be predicted on
    to_be_predicted_pics = [f for f in listdir(predict_folder_path) if isfile(join(predict_folder_path, f))]

    # load to-be-predicted-on face data
    dataset = dset.ImageFolder(root=predict_folder_folder_path,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(to_be_predicted_pics),
                                         shuffle=False, num_workers=1)

    # load network
    model = CelebrityConvNet(image_size, num_classes)
    model.load_state_dict(torch.load(nn_path))
    model.eval()

    # show to-be-predicted faces
    if True:
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        imshow(torchvision.utils.make_grid(images))

    # predict faces
    inputs, _ = iter(dataloader).next()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    celeb_ids = predicted.tolist()

    # display matches for each face
    for i in range(len(celeb_ids)):
        # the pic which was predicted on
        to_be_predicted_pic = to_be_predicted_pics[i]

        # extend celeb_id to 6 digits to match naming format
        celeb_id = str(celeb_ids[i])
        while len(celeb_id) < 6 :
            celeb_id = '0' + celeb_id
        print(celeb_id)
        
        # obtain list of photos of matching celebrity
        celeb_folder_path = celeb_face_path + '/' + celeb_id
        match_pictures = [f for f in listdir(celeb_folder_path) if isfile(join(celeb_folder_path, f))]

        f, axarr = plt.subplots(1, 4)

        # render face which was predicted on
        axarr[0].imshow(mpim.imread(predict_folder_path + '/' + to_be_predicted_pic))
        axarr[0].set_title('Original')
        axarr[0].axis('off')

        # render the first three pictures of the matching celebrity
        for j in range(3):
            predicted_img = mpim.imread(celeb_folder_path + '/' + match_pictures[j])
            axarr[j + 1].imshow(predicted_img)
            axarr[j + 1].axis('off')

        axarr[2].set_title('Best Matching Celebrity')

        # save and render
        plt.savefig(fig_path + '/' + DATA_SET_SIZE.lower() + '_' + to_be_predicted_pic)
        plt.show()