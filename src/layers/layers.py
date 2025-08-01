from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation,AveragePooling2D
from ..attention.attentions_module import bam_block, scse_block, cbam_block
from tensorflow.keras.layers import Conv2D, Add,MaxPool2D
from tensorflow.keras import layers
from keras.regularizers import l2
def select_attention(feature, filter_num, attention_type='CBAM', ratio=16, layer_name=None):
    if attention_type == 'CBAM':
        feature = cbam_block(feature, filter_num, reduction_ratio=ratio, name=layer_name + "_CBAM_")
        print('Using CBAM ne')
    elif attention_type == 'BAM':
        print('Using BAM ne')
        feature = bam_block(feature, filter_num, reduction_ratio=ratio, num_layers=1, dilation_val=4, name=layer_name + "_BAM_")
    elif attention_type == 'scSE':
        print('Using scSE')
        feature = scse_block(feature, filter_num, reduction_ratio=ratio, name=layer_name + "_scSE_")
    return feature

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, attention=""):
    """A residual block for ResNetV1
    Args:
      x: input tensor.
      filters: array, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
      name: string, block label.
      attention: string, select attention method
    Returns:
      Output tensor for the residual block.
    """
    batch_axis = 1
    filters1, filters2, filters3 = filters

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters3, 1, strides=stride, name=name + '_Shortcut_Conv')(x)
        shortcut = layers.BatchNormalization(
            axis=batch_axis, name=name + '_Shortcut_BN')(shortcut)
    else:
        shortcut = x

    #conv 1x1
    x = layers.Conv2D(filters1, 1, strides=stride, name=name + '_1_Conv')(x)
    x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name=name + '_1_BN')(x)
    x = layers.Activation('elu', name=name + '_1_elu')(x)
    # conv 3x3
    x = layers.Conv2D(filters2, kernel_size, padding='SAME', name=name + '_2_Conv')(x)
    x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name=name + '_2_BN')(x)
    x = layers.Activation('elu', name=name + '_2_elu')(x)
    # conv 1x1
    x = layers.Conv2D(filters3, 1, name=name + '_3_Conv')(x)
    x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name=name + '_3_BN')(x)

    if attention == "":
        x = x
    else:
        x = select_attention(x, filter_num=filters3, attention_type=attention, layer_name='Attention_{}'.format(name))
    x = layers.Add(name=name + '_Add')([shortcut, x])
    x = layers.Activation('elu', name=name + '_Output')(x)

    return x


def stage(x, filters, blocks, attention, stride1=2, name=None):
    """A group of stacked residual blocks for ResNetV1
    Args:
      x: tensor, input
      filters: array, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, group of blocks label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = residual_block(x, filters, stride=stride1, name=name + '_Block1', attention=attention)
    for i in range(2, blocks + 1):
        x = residual_block(x, filters, conv_shortcut=False, name=name + '_Block' + str(i), attention=attention)
    return x


def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1),name=""):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   name=name+'_Conv2D_BN'
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1), name=""):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides, name)
    layer = Activation('relu', name=name + '_Relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True, attention=None, name=""):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2,name=name + '_Conv2D_BN_DS')
        stride = 2
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              name=name + '_1_Conv2D_RElu'
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         name=name + '_2_Conv2D_BN'
                         )

    if attention is None:
        residual = residual
    else:
        residual = select_attention(residual, filter_num=filters, attention_type=attention, layer_name='{}_Attention_'.format( name))
    out = layers.add([residual_x, residual])

    out = Activation('relu')(out)
    return out



def vgg_conv_block(input, block_idx, filter, attention_type, activation='elu'):
    print('Conv: ' + str(block_idx) + ' filter: ' + str(filter))
    x = input
    for i in range(2):
        x = Conv2D(filters=filter, kernel_size=3, padding='same', activation=activation, name=f"Conv{block_idx}.{i + 1}")(x)
    if block_idx > 2:
        x = Conv2D(filters=filter, kernel_size=3, padding='same', activation=activation, name=f"Conv{block_idx}.3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=f"MaxPool2D_{block_idx}")(x)
    # if block_idx == 5:
    #     attention_output = select_attention(x, filter_num=filter, attention_type=attention_type, layer_name='Conv_Attention_')
    #     x = layers.Add(name='Conv_Last_Add' + str(block_idx))([attention_output, x])
    return x
def vgg_conv(input_tensor, filters, block_num, activation='elu', attention_type=None):
    """
    Creates a VGG convolutional block.

    Parameters:
    - input_tensor: The input tensor to the convolutional block.
    - filters: A list of integers representing the number of filters for each Conv layer in the block.
    - block_num: An integer representing the block number (used for naming layers).
    - activation: The activation function to use (default is 'relu').

    Returns:
    - The output tensor after applying the convolutional block.
    """

    x = input_tensor
    for i, f in enumerate(filters):
        x = Conv2D(filters=f, kernel_size=3, padding='same', activation=activation, name=f"Conv{block_num}.{i + 1}")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=f"MaxPool2D_{block_num}")(x)

    if attention_type is not None:
        attention_output = select_attention(x, filter_num=64, attention_type=attention_type, layer_name='Conv{}_Attention_'.format(block_num))
        x = layers.Add(name='Conv{}_Add_Att'.format(block_num))([attention_output, x])
    else:
        x = x
    return x