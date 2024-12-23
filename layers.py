from tensorflow.keras.layers import (BatchNormalization, ReLU)
from attentions_module import bam_block, scse_block, cbam_block
from tensorflow.keras.layers import Conv2D, Add,MaxPool2D
from tensorflow.keras import layers

# def cbam_block(cbam_feature, ratio=8):
def select_attention(feature, filter_num, attention_type='CBAM', ratio=16, layer_name=None):
    if attention_type == 'CBAM':
        feature = cbam_block(feature, filter_num, reduction_ratio=ratio, name=layer_name + "_CBAM_")
        # feature = spatial_attention(feature)
        print('Using CBAM ne')
    elif attention_type == 'BAM':
        print('Using BAM ne')
        feature = bam_block(feature, reduction_ratio=ratio, num_layers=1, dilation_val=4, name=layer_name + "_BAM_")

    elif attention_type == 'scSE':
        print('Using scSE')
        feature = scse_block(feature, reduction_ratio=ratio, name=layer_name + "_scSE_")
    return feature


# def bottleneck_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1, use_CBAM=False):
def bottleneck_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1, attention_type=None):
    '''BottleNeckBlock use stack of 3 layers: 1x1, 3x3 and 1x1 convolutions

    Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
    # conv1x1
    conv1 = Conv2D(filters=filter_num,
                   kernel_size=1,
                   strides=stride,
                   padding='valid',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(input)
    bn1 = BatchNormalization(name='conv{}_block{}_1_bn'.format(stage_idx, block_idx))(conv1)
    relu1 = ReLU(name='conv{}_block{}_1_relu'.format(stage_idx, block_idx))(bn1)
    # conv3x3
    conv2 = Conv2D(filters=filter_num,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(relu1)
    bn2 = BatchNormalization(name='conv{}_block{}_2_bn'.format(stage_idx, block_idx))(conv2)
    relu2 = ReLU(name='conv{}_block{}_2_relu'.format(stage_idx, block_idx))(bn2)
    # conv1x1
    conv3 = Conv2D(filters=filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='valid',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_3_conv'.format(stage_idx, block_idx))(relu2)

    bn3 = BatchNormalization(name='conv{}_block{}_3_bn'.format(stage_idx, block_idx))(conv3)
    if attention_type is not None:
        bn3 = select_attention(bn3, filter_num=filter_num, attention_type=attention_type, layer_name='Conv{}_block{}_Attention_'.format(stage_idx, block_idx))
    else:
        bn3 = bn3
    return bn3


# def resblock(input, filter_num, stride=1, use_bottleneck=False,stage_idx=-1, block_idx=-1, use_CBAM=False):
def residual_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1, attention_type=None, conv_shortcut=True):
    '''A complete `Residual Unit` of ResNet

    Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
    block_idx: index of current block in stage
    '''
    batch_axis = 1
    if conv_shortcut:
        shortcut = layers.Conv2D(filter_num, 1, strides=stride, name='conv{}_Shortcut_Conv'.format(stage_idx))(input)
        shortcut = layers.BatchNormalization(axis=batch_axis, name='conv{}_Shortcut_BN'.format(stage_idx))(shortcut)
    else:
        shortcut = input
    residual = bottleneck_block(input, filter_num, stride, stage_idx, block_idx, attention_type=attention_type)
    output = Add(name='conv{}_block{}_add'.format(stage_idx, block_idx))([shortcut, residual])
    output = ReLU(name='conv{}_block{}_relu'.format(stage_idx, block_idx))(output)
    return output

# def stage(input, filter_num, num_block, use_downsample=True, use_bottleneck=False,stage_idx=-1, use_CBAM=False, use_CBAM_after_stage=False):
def stage(input, filter_num, num_block, stage_idx=-1, attention_type=None, conv_shortcut=True):
    net = residual_block(input=input, filter_num=filter_num, stride=2 ,stage_idx=stage_idx, block_idx=1, attention_type=attention_type, conv_shortcut=True)
    for i in range(1, num_block):
        net = residual_block(input=net, filter_num=filter_num, stride=1,stage_idx=stage_idx, block_idx=i + 1, attention_type=attention_type, conv_shortcut=False)
    return net

def vgg_conv_block(input, block_idx, filter, attention_type, activation='elu'):
    print('Conv: ' + str(block_idx) + ' filter: ' + str(filter))
    x = input
    for i in range(2):
        x = Conv2D(filters=filter, kernel_size=3, padding='same', activation=activation, name=f"Conv{block_idx}.{i + 1}")(x)
    if block_idx > 2:
        x = Conv2D(filters=filter, kernel_size=3, padding='same', activation=activation, name=f"Conv{block_idx}.3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=f"MaxPool2D_{block_idx}")(x)
    if block_idx == 5:
        attention_output = select_attention(x, filter_num=filter, attention_type=attention_type, layer_name='Conv_Attention_')
        x = layers.Add(name='Conv_Last_Add' + str(block_idx))([attention_output, x])
    return x