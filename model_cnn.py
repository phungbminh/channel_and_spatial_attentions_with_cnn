
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (Dropout, Activation, ZeroPadding2D,Input, Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D)
from layers import stage, vgg_conv_block, select_attention


def resnet50(input_shape=(48, 48, 3), num_classes=1000, attention_type='CBAM'):
    input = Input(input_shape, name='input')
    layers = [3,4,6,3]
    print(input)
    net = ZeroPadding2D(padding=((3, 3), (3, 3)), name='Conv1_Pad')(input)
    net = Conv2D(64, 7, strides=2, name='Conv1')(net)
    net = BatchNormalization(axis=1, epsilon=1.001e-5, name='Conv1_BN')(net)
    net = Activation('relu', name='Conv1_relu')(net)

    net = ZeroPadding2D(padding=((1, 1), (1, 1)), name='MaxPool2D_1_Pad')(net)
    net = MaxPooling2D(3, strides=2, name='MaxPool2D_1')(net)

    # if attention_type is not None:
    #     net = select_attention(net, filter_num=64, attention_type=attention_type, layer_name='Conv1_block1_Attention_')
    # else:
    #     net = net
    # conv2_x, conv3_x, conv4_x, conv5_x
    filters = [64, 128, 256, 512]
    for i in range(len(filters)):
        print(f'Conv {i + 1}:')
        net = stage(input=net, filter_num=filters[i], num_block=layers[i],
                    stage_idx=i + 2, attention_type=attention_type)
    net = GlobalAveragePooling2D(name='avg_pool')(net)
    output = Dense(num_classes, activation='softmax', name='predictions')(net)
    model = Model(input, output)
    return model

def vgg16(input_shape=(48,48,1), num_classes=1000, attention_type='CBAM'):
    activation = 'relu'
    input = Input(input_shape, name='input')
    # Conv1
    net = Conv2D(filters=64, kernel_size=3, padding='same', activation=activation, name="Conv1.1")(input)
    net = Conv2D(filters=64, kernel_size=3, padding='same', activation=activation, name="Conv1.2")(net)
    net = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_1")(net)

    if attention_type is not None:
        net = select_attention(net, filter_num=64, attention_type=attention_type, layer_name='Conv1_block1_Attention_')
    else:
        net = net
    filters = [128, 256, 512, 512]
    for i in range(len(filters)):
        net = vgg_conv_block(input=net,
                             block_idx=(i + 2),
                             filter=filters[i],
                             attention_type=attention_type,
                             activation = activation)
    net = Flatten()(net)
    net = Dense(units=4096, activation=activation, name="Dense1")(net)
    net = Dropout(0.5)(net)
    net = Dense(units=4096, activation=activation, name="Dense2")(net)
    net = Dropout(0.5)(net)

    output = Dense(units=num_classes, activation='softmax', name="DenseFinal")(net)

    model = Model(input, output)
    return model


