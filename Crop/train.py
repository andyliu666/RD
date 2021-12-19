import sys
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms, datasets as ds
import torchfcn
import torch
import torch.optim as optim
from dataset import myDataSet
from torch.autograd import Variable
import random
from unet import *
import time

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device for training, i.e. gpu or cpu
print(DEVICE)

_size = 256, 256
resize = transforms.Resize(_size, interpolation=0)

# set your transforms
train_transforms = transforms.Compose([
    transforms.Resize(_size, interpolation=0),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(_size, padding=10),  # needed after rotation (with original size)
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(_size, interpolation=0),
    transforms.ToTensor(),
])

path_images = "../data/bottom_view_png"
path_masks = "../data/bottom_view_label"

with open('../data/bottom_view.txt', 'r') as f:
    ids_train = [l.strip() for l in f]
with open('../data/bottom_view.txt', 'r') as f:
    ids_val = [l.strip() for l in f]

custom_dataset_train = myDataSet(ids_train, path_images, path_masks, test_transforms)
custom_dataset_val = myDataSet(ids_val, path_images, path_masks, test_transforms)
print(custom_dataset_train.__len__)
print("My custom training-dataset has {} elements".format(len(custom_dataset_train)))
print("My custom validation-dataset has {} elements".format(len(custom_dataset_val)))

BATCH_SIZE = 1
dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=0)
dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


dir = 'bottom_view'
if not os.path.exists('../model/' + dir):
        os.makedirs('../model/' + dir)

from metrics import iou_pytorch_eval, IoULoss, IoUBCELoss
from metrics import iou_pytorch_test, dice_pytorch_test, precision_pytorch_test, recall_pytorch_test, fbeta_pytorch_test, accuracy_pytorch_test

model = UNet(n_channels=3, n_classes=1)
#test_dir = '../model/att train/Epoch101.model'
#checkpoint = torch.load(test_dir)#####
#model.load_state_dict(checkpoint['net'])
model=model.to(DEVICE)
loss_criterion = IoUBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-8)


def img_to_rgb(img):
    colormap = [[0, 0, 0], [128, 0, 0]]
    cm = np.array(colormap).astype('uint8')
    img = cm[img]
    img = img.transpose(2, 0, 1)

    img = torch.from_numpy(img)
    img = transforms.ToPILImage()(img).convert('RGB')
    return img

'''
epochs = 101
for epoch in range(epochs):
    running_loss = 0
    # Train
    model.train()
    for i, (imgs, masks) in enumerate(dataloader_train):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        prediction = model(imgs)
        loss = loss_criterion(prediction, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}".format(epoch, epochs, i, len(dataloader_train),
                                                                         running_loss / (i + 1)), end="")

        state = {'net': model.state_dict()}
        if epoch % 5 == 0:
            torch.save(state, '../model/' + dir + '/Epoch{}.model'.format(epoch + 1))
        # running_iou += iou_pytorch_eval(prediction, masks)
        # print("\r Epoch: {} of {}, Iter.: {} of {}, IoU:  {:.6f}".format(epoch, epochs, i, len(dataloader_train), running_iou/(i+1)), end="")
'''
test_dir = '../model/bottom_view_depth/Epoch101.model'
checkpoint = torch.load(test_dir)  #####
model.load_state_dict(checkpoint['net'])
model.to(DEVICE)
model.eval()
val_loss = 0
val_iou = 0
total = 0
for i, (imgs, masks) in enumerate(dataloader_val):
    imgs, masks = imgs.cuda(), masks.cuda()
    #print(imgs.shape)
    start_time = time.time()
    prediction = model(imgs)
    total += time.time() - start_time
    print(time.time() - start_time)
    #print(total)
    #loss = loss_criterion(prediction, masks)
    #val_loss += loss.item()
    #print(" Val. Loss: {:.6f}".format(val_loss / (i + 1)))

    preds = prediction.squeeze().cpu().data.numpy()  ###从gpu取
    #print(preds.shape)
    imgs = imgs.cpu().data.numpy()

    gts = masks.squeeze().cpu().data.numpy().astype('uint8')

    preds = preds.round().astype('uint8')

    imgs = imgs.transpose(0, 2, 3, 1)
'''
    for i in range(preds.shape[0]):
        pred = img_to_rgb(preds[i])
        ig, ax = plt.subplots(1)

        mask = pred

        # np.nonzero会返回形如(array,array)的东西
        # 第一个是行的index，第二个是列的index
        # 例如 np.nonzero(np.array([[0,1],[2,0]])
        # 会返回 ( array([0,1]), array([1,0]) )
        coor = np.nonzero(mask)
        xmin = coor[0][0]
        xmax = coor[0][-1]
        coor[1].sort()  # 直接改变原数组，没有返回值
        ymin = coor[1][0]
        ymax = coor[1][-1]

        bottomleft = (ymin, xmin)

        width = ymax - ymin
        height = xmax - xmin
        total += time.time() - start_time
        print(xmin * 4.6875, xmax * 4.6875, ymin * 7.5, ymax * 7.5)

        # Display the image
        ax.imshow(imgs[i], cmap=plt.cm.gray)
        # Create a Rectangle patch
        rect = patches.Rectangle(bottomleft, width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()
        print(total)
        gt = img_to_rgb(gts[i])
        fig = plt.figure()
        ax1 = plt.subplot(131)
        ax1.imshow(imgs[i])
        ax1.set_title('Input')

        ax2 = plt.subplot(132)
        ax2.imshow(gt)
        ax2.set_title('Gt')

        ax3 = plt.subplot(133)
        ax3.imshow(pred)
        ax3.set_title('Pred')
        plt.show()
'''
