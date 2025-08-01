from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
from ..layers.layers import *
from tensorflow.keras.layers import (Dropout, Activation, ZeroPadding2D, Input, Conv2D, MaxPool2D, Flatten, Dense,
                                     MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, AveragePooling2D)



def gpu_check():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def VGG16(input_shape, num_classes, attention_type=""):
    """Function to output the VGG16 CNN Model
       Args:
          a_hidden: string,default 'relu', activation function used for hidden layerss
          a_output: string, default 'softmax', output activation function
          num_classes: integer, required, states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
    activation = 'elu'
    input = Input(input_shape, name='input')
    # Conv1
    print('Conv1_x2')
    net = Conv2D(filters=64, kernel_size=3, padding='same', activation=activation, name="Conv1.1")(input)
    net = Conv2D(filters=64, kernel_size=3, padding='same', activation=activation, name="Conv1.2")(net)
    net = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_1")(net)

    if attention_type is not None:
        attention_output = select_attention(net, filter_num=64, attention_type=attention_type, layer_name='Conv1_block1_Attention_')
        net = layers.Add(name='Conv1_Add_Att')([attention_output, net])
        model_name = "VGG16" + "_" + attention_type
    else:
        net = net
        model_name = "VGG16"

    # 2nd Conv Block
    print('Conv2_x2')
    net = vgg_conv(net, filters=[128, 128], block_num=2, activation=activation, attention_type=attention_type)
    # 3rd Conv Block
    print('Conv3_x3')
    net = vgg_conv(net, filters=[256, 256, 256], block_num=3, activation=activation, attention_type=attention_type)
    # 4th Conv Block
    print('Conv4_x3')
    net = vgg_conv(net, filters=[512, 512, 512], block_num=4, activation=activation, attention_type=attention_type)
    # 5th Conv Block
    print('Conv5_x3')
    net = vgg_conv(net, filters=[512, 512, 512], block_num=5, activation=activation, attention_type=attention_type)

    # Fully connected layers
    #net = Flatten()(net) ->
    net = GlobalAveragePooling2D(name="Final_AveragePooling")(net)
    #net = Dense(units=4096, activation=activation, name="Dense1")(net)
    net = Dense(units=2048, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(net)
    net = Dropout(rate=0.2)(net)
    #net = Dropout(0.2)(net)
    #net = Dense(units=4096, activation=activation, name="Dense2")(net)
    #net = Dropout(0.2)(net)

    output = Dense(units=num_classes, activation='softmax', name="DenseFinal")(net)
    model = Model(inputs=input, outputs=output, name=model_name)

    return model


def ResNet(num_classes, model_name="ResNet50", input_shape=(48, 48, 3), a_output='softmax', pooling='avg',
           attention=""):
    """Function that is able to return different ResNet V1 Models with adaptive input size support
       Args:
       	  model: string, default 'ResNet50', select which ResNet model to use ResNet50 , ResNet101 or ResNet152
          a_output: string, default 'softmax', output activation function
          pooling: string,default 'avg', pooling used for the final layer either 'avg' or 'max'
          attention: string, default '', select which Attention block to use SEnet , ECANet or CBAM
          num_classes: integer, required, states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
    # Input
    batch_axis = 1
    input = Input(input_shape, name='img')
    
    # Auto-detect input size for adaptive processing
    input_size = input_shape[0]
    
    if model_name == "ResNet50":
        num_blocks = [3, 4, 6, 3]
    elif model_name == "ResNet18":
        num_blocks = [2, 2, 2, 2]
    
    # Conv_1 - Adaptive for different input sizes
    print('Conv1 7x7 (adaptive)')
    if input_size <= 32:
        # For small inputs (32x32), use smaller stride to preserve spatial info
        print('  - Using stride=1 for small input size')
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='Conv1_Pad')(input)
        x = layers.Conv2D(64, 7, strides=1, name='Conv1')(x)  # stride=1 instead of 2
        x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name='Conv1_BN')(x)
        x = layers.Activation('relu', name='Conv1_relu')(x)
        # Skip MaxPooling for small inputs
    else:
        # For larger inputs (48x48+), use standard ResNet approach
        print('  - Using stride=2 for standard input size')
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='Conv1_Pad')(input)
        x = layers.Conv2D(64, 7, strides=2, name='Conv1')(x)
        x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name='Conv1_BN')(x)
        x = layers.Activation('relu', name='Conv1_relu')(x)
        
        # Add MaxPooling for larger inputs
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='MaxPool2D_1_Pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='MaxPool2D_1')(x)

    # Residual Stage
    print('Stage 1: Conv2_x{}'.format(num_blocks[0]))
    x = stage(x, [64, 64, 256], num_blocks[0], stride1=1, name='Conv2_x{}'.format(num_blocks[0]), attention=attention)
    print('Stage 2: Conv3_x{}'.format(num_blocks[1]))
    x = stage(x, [128, 128, 512], num_blocks[1], name='Conv3_x{}'.format(num_blocks[1]), attention=attention)
    print('Stage 3: Conv4_x{}'.format(num_blocks[2]))
    x = stage(x, [256, 256, 1024], num_blocks[2], name='Conv4_x{}'.format(num_blocks[2]), attention=attention)
    print('Stage 4: Conv5_x{}'.format(num_blocks[3]))
    x = stage(x, [512, 512, 2048], num_blocks[3], name='Conv5_x{}'.format(num_blocks[2]), attention=None)

    # Output
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='AvgPool2D_Final')(x)
    else:
        x = layers.GlobalMaxPooling2D(name='MaxPool2D_Final')(x)

    output = layers.Dense(num_classes, activation=a_output, name='DenseFinal')(x)

    if attention == "":
        model_name = model_name
    else:
        model_name = model_name + "_" + attention
    model = Model(inputs=input, outputs=output, name=model_name)
    return model


def ResNetCIFA(classes, input_shape, weight_decay=1e-4, attention=None):
    input = Input(shape=input_shape)
    x = input
    print('Conv1:')
    #x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(2, 2))
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))
    #x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)

    if attention is not None:
        model_name = 'ResNet18' + "_" + attention
    else:
        model_name = 'ResNet18'
    # # conv 2
    print('Stage 1:')
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, attention=attention, name='Block2.1')
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, attention=attention, name='Block2.2')
    # # conv 3
    print('Stage 2:')
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, attention=attention, name='Block3.1')
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, attention=attention, name='Block3.2')
    # # conv 4
    print('Stage 3:')
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, attention=attention,name='Block4.1')
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, attention=attention, name='Block4.2')
    # # conv 5
    print('Stage 4:')
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, attention=attention, name='Block5.1')
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, attention=attention, name='Block5.2')
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    #x = Flatten()(x)
    x = GlobalAveragePooling2D(name="Final_AveragePooling")(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x, name=model_name)
    return model