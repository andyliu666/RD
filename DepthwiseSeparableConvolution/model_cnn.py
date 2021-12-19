import torch.nn as nn
from DepthwiseSeparableConvolution import depthwise_separable_conv

class bottle_screener(nn.Module):
    def __init__(self, num_classes=3):
        super(bottle_screener, self).__init__()
        input_shape = () #input shape of the image 
        self.dsc0 = depthwise_separable_conv(nin, nout, kernel_size = 3, padding = 1, bias=False) #filters 32
        self.pool0 = nn.MaxPool2d((2,2)) #pool size(2,2)
        self.dsc1 = depthwise_separable_conv(nin, nout, kernel_size = 3, padding = 1, bias=False) #filters 64
        self.pool1 = nn.MaxPool2d((2,2))
        self.dsc2 = depthwise_separable_conv(nin, nout, kernel_size = 3, padding = 1, bias=False) #filters 128
        self.pool2 = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(512)
        self.relu = nn.ReLU
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(3)
        self.sigmoid = nn.Softmax()

#My old model in Tensorflow 
'''
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same',input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same',input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same',input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(len(folders)))
model.add(Activation('sigmoid'))
'''