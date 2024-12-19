from tensorflow.keras.layers import (MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D)
from layers import stage

# def build(input_shape,num_classes,layers,use_bottleneck=False,use_CBAM=False,use_CBAM_after_stage=False):
def build(input_shape, num_classes, layers, use_bottleneck=False, attention_type=None):
    '''A complete `stage` of ResNet
    '''
    input = Input(input_shape, name='input')
    # conv1
    net = Conv2D(filters=64,
                 kernel_size=7,
                 strides=2,
                 padding='same',
                 kernel_initializer='he_normal',
                 name='conv1_conv')(input)
    net = BatchNormalization(name='conv1_bn')(net)
    net = ReLU(name='conv1_relu')(net)
    net = MaxPooling2D(pool_size=3,
                       strides=2,
                       padding='same',
                       name='conv1_max_pool')(net)

    # conv2_x, conv3_x, conv4_x, conv5_x
    filters = [64, 128, 256, 512]
    for i in range(len(filters)):
        print(f'stage {i + 1}:')
        net = stage(input=net,
                    filter_num=filters[i],
                    num_block=layers[i],
                    use_downsample=i != 0,
                    use_bottleneck=use_bottleneck,
                    stage_idx=i + 2,
                    # use_CBAM=use_CBAM,
                    # use_CBAM_after_stage = use_CBAM_after_stage
                    attention_type=attention_type
                    )
        print("")
    net = GlobalAveragePooling2D(name='avg_pool')(net)
    output = Dense(num_classes, activation='softmax', name='predictions')(net)
    model = Model(input, output)

    return model

def resnet18(input_shape=(224,224,3),num_classes=1000, attention_type='CBAM'):
    return build(input_shape,num_classes,[2,2,2,2],use_bottleneck=False, attention_type=attention_type)

def resnet34(input_shape=(224,224,3),num_classes=1000, attention_type='CBAM'):
    return build(input_shape,num_classes,[3,4,6,3],use_bottleneck=False, attention_type=attention_type)

def resnet50(input_shape=(48,48,3),num_classes=1000, attention_type='CBAM'):
    return build(input_shape,num_classes,[3,4,6,3],use_bottleneck=True, attention_type=attention_type)

def resnet101(input_shape=(224,224,3),num_classes=1000, attention_type='CBAM'):
    return build(input_shape,num_classes,[3,4,23,3],use_bottleneck=True, attention_type=attention_type)

def resnet152(input_shape=(224,224,3),num_classes=1000, attention_type='CBAM'):
    return build(input_shape,num_classes,[3,8,36,3],use_bottleneck=True, attention_type=attention_type)


import math

from tensorflow.keras import layers
import tensorflow as tf
from keras import backend
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras import backend


def CBAM_block(input_layer, filter_num, reduction_ratio=32, kernel_size=7, name=None):
    """CBAM: Convolutional Block Attention Module Block
    Args:
      input_layer: input tensor
      filter_num: integer, number of neurons in the hidden layers
      reduction_ratio: integer, default 32, reduction ratio for the number of neurons in the hidden layers
      kernel_size: integer, default 7, kernel size of the spatial convolution excitation convolution
      name: string, block label
    Returns:
      Output A tensor for the CBAM attention block
    """
    axis = -1

    # CHANNEL ATTENTION
    avg_pool = layers.GlobalAveragePooling2D(name=name + "_Channel_AveragePooling")(input_layer)
    max_pool = layers.GlobalMaxPooling2D(name=name + "_Channel_MaxPooling")(input_layer)

    # Shared MLP
    dense1 = layers.Dense(filter_num // reduction_ratio, activation='relu', name=name + "_Channel_FC_1")
    dense2 = layers.Dense(filter_num, name=name + "_Channel_FC_2")

    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))

    channel = layers.add([avg_out, max_out])
    channel = layers.Activation('sigmoid', name=name + "_Channel_Sigmoid")(channel)
    channel = layers.Reshape((1, 1, filter_num), name=name + "_Channel_Reshape")(channel)

    channel_output = layers.multiply([input_layer, channel])

    # SPATIAL ATTENTION
    avg_pool2 = layers.Lambda(lambda x: tf.reduce_mean(x, axis=axis, keepdims=True))(input_layer)
    max_pool2 = layers.Lambda(lambda x: tf.reduce_max(x, axis=axis, keepdims=True))(input_layer)

    spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)

    # K = 7 achieves the highest accuracy
    spatial = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=name + "_Spatial_Conv2D")(spatial)
    spatial_out = layers.Activation('sigmoid', name=name + "_Spatial_Sigmoid")(spatial)

    CBAM_out = layers.multiply([channel_output, spatial_out])

    return CBAM_out


def vgg16(img_height=48,
                img_width=48,
                a_hidden='elu',  # Hidden activation
                a_output='softmax',  # Output activation
                attention="",
                num_classes=7
                ):
    """Function to output the VGG16 CNN Model
       Args:
          img_height: integer,default '48', input image height
          img_width: integer,default '48', input image width
          a_hidden: string,default 'relu', activation function used for hidden layerss
          a_output: string, default 'softmax', output activation function
          num_classes: integer, default 7,states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
    # Input
    input_img = Input(shape=(img_height, img_width, 1), name="img")

    # 1st Conv Block
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.1")(input_img)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.2")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_1")(x)

    # 2nd Conv Block
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.1")(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.2")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_2")(x)

    # 3rd Conv block
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.1")(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.2")(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_3")(x)

    # 4th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.1")(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.2")(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_4")(x)

    # 5th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.1")(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.2")(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_5")(x)

    if attention == "":
        x = x
    else:
        # if attention == "SEnet":
        #     attention_output = squeeze_excitation_block(x, 512, 16.0, name="Conv_Last_SNE_")
        # if attention == "ECANet":
        #     attention_output = ECA_Net_block(x, adaptive=True, name="Conv_Last_ECANet_")
        if attention == "CBAM":
            attention_output = CBAM_block(x, 512, reduction_ratio=16, kernel_size=7, name="Conv_Last_CBAM_")

        x = layers.Add(name='ConvLast_Add1')([attention_output, x])

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(units=4096, activation=a_hidden, name="Dense1")(x)
    x = Dense(units=4096, activation=a_hidden, name="Dense2")(x)

    output = Dense(units=num_classes, activation=a_output, name="DenseFinal")(x)
    if attention == "":
        model_name = "VGG16"
    else:
        model_name = "VGG16" + "_" + attention

    model = Model(inputs=input_img, outputs=output, name=model_name)

    return model