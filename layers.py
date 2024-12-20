from tensorflow.keras.layers import (BatchNormalization, ReLU)
from attentions_module import channel_attention,spatial_attention, sc_conv, bam, scse, CBAM_block
from tensorflow.keras.layers import Conv2D, Add,MaxPool2D
from tensorflow.keras import layers

# def cbam_block(cbam_feature, ratio=8):
def attention_block(feature, attention_type='CBAM', ratio=8, sc_params=None):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    if attention_type == 'CBAM':
        feature = channel_attention(feature, ratio)
        feature = spatial_attention(feature)
        print('Using CBAM ne')

    elif attention_type == 'SCNet':
        pooling_r = 4
        print('Using SCNet ne')
        feature = sc_conv(feature, pooling_r=pooling_r)

    elif attention_type == 'BAM':
        print('Using BAM ne')
        feature = bam(feature, reduction_ratio=16, num_layers=1, dilation_conv_num=2, dilation_val=4)

    elif attention_type == 'scSE':
        print('Using scSE')
        feature = scse(feature, reduction_ratio=ratio)
    return feature

# def basic_block(input, filter_num, stride=1,stage_idx=-1, block_idx=-1, use_CBAM=False):
def basic_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1, attention_type=None):
    '''BasicBlock use stack of two 3x3 convolutions layers

    Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
    '''
    # conv3x3
    conv1 = Conv2D(filters=filter_num,
                   kernel_size=3,
                   strides=stride,
                   padding='same',
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

    # if use_CBAM == True:
    #     bn2 = cbam_block(cbam_feature = bn2)
    # return bn2
    if attention_type is not None:
        bn2 = attention_block(bn2, attention_type=attention_type)
    return bn2


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
    conv3 = Conv2D(filters=4 * filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='valid',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_3_conv'.format(stage_idx, block_idx))(relu2)

    # if use_CBAM == True:
    #     bn3 = cbam_block(cbam_feature = bn3)
    # return bn3
    layer_name = f'conv{stage_idx}_block{block_idx}_3_bn'
    print(layer_name)
    if attention_type is not None:
        bn_temp = BatchNormalization(name='conv{}_block{}_3_attention'.format(stage_idx, block_idx))(conv3)
        bn3 = attention_block(bn_temp, attention_type=attention_type)
    else:
        bn_temp = BatchNormalization(name='conv{}_block{}_3_bn'.format(stage_idx, block_idx))(conv3)
        bn3 = bn_temp
    return bn3, layer_name


# def resblock(input, filter_num, stride=1, use_bottleneck=False,stage_idx=-1, block_idx=-1, use_CBAM=False):
def resblock(input, filter_num, stride=1, use_bottleneck=False, stage_idx=-1, block_idx=-1, attention_type=None):
    '''A complete `Residual Unit` of ResNet

    Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
    block_idx: index of current block in stage
    '''
    # if use_bottleneck:
    #     residual = bottleneck_block(input, filter_num, stride,stage_idx, block_idx,use_CBAM)
    #     expansion=4
    # else:
    #     residual = basic_block(input, filter_num, stride,stage_idx, block_idx,use_CBAM)
    #     expansion=1
    if use_bottleneck:
        residual, _ = bottleneck_block(input, filter_num, stride, stage_idx, block_idx, attention_type=attention_type)
        expansion = 4
    else:
        residual = basic_block(input, filter_num, stride, stage_idx, block_idx, attention_type=attention_type)
        expansion = 1

    shortcut = input
    # use projection short cut when dimensions increase
    if stride > 1 or input.shape[3] != residual.shape[3]:
        # if stride > 1 or input.shape[-1] != residual.shape[-1]:
        shortcut = Conv2D(expansion * filter_num,
                          kernel_size=1,
                          strides=stride,
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv{}_block{}_projection-shortcut_conv'.format(stage_idx, block_idx))(input)
        shortcut = BatchNormalization(name='conv{}_block{}_projection-shortcut_bn'.format(stage_idx, block_idx))(
            shortcut)

    output = Add(name='conv{}_block{}_add'.format(stage_idx, block_idx))([residual, shortcut])

    return ReLU(name='conv{}_block{}_relu'.format(stage_idx, block_idx))(output)


# def stage(input, filter_num, num_block, use_downsample=True, use_bottleneck=False,stage_idx=-1, use_CBAM=False, use_CBAM_after_stage=False):
def stage(input, filter_num, num_block, use_downsample=True, use_bottleneck=False, stage_idx=-1, attention_type=None):
    ''' -- Stacking Residual Units on the same stage

    Args:
    filter_num: the number of filters in the convolution used during stage
    num_block: number of `Residual Unit` in a stage
    use_downsample: Down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
    '''
    # flag = 0;
    # if use_CBAM == False:
    #     flag = 0 # no use CBAM
    # elif use_CBAM == True and use_CBAM_after_stage == False:
    #     flag = 1 # use cbam after all blocks
    # elif use_CBAM == True and use_CBAM_after_stage == True:
    #     flag = 2 # use cbam only after stage
    # net = resblock(input = input, filter_num = filter_num, stride = 2 if use_downsample else 1, use_bottleneck = use_bottleneck, stage_idx = stage_idx, block_idx = 1,use_CBAM=True if flag == 1 else False)

    # for i in range(1, num_block):
    #     net = resblock(input = net, filter_num = filter_num,stride = 1,use_bottleneck = use_bottleneck,stage_idx = stage_idx, block_idx = i+1, use_CBAM=True if flag == 1 else False)

    # if flag == 2:
    #     net = cbam_block(cbam_feature = net)
    net = resblock(input=input, filter_num=filter_num, stride=2 if use_downsample else 1,
                   use_bottleneck=use_bottleneck, stage_idx=stage_idx, block_idx=1, attention_type=attention_type)

    for i in range(1, num_block):
        net = resblock(input=net, filter_num=filter_num, stride=1,
                       use_bottleneck=use_bottleneck, stage_idx=stage_idx, block_idx=i + 1,
                       attention_type=attention_type)

    return net

def vgg_conv_block(input, block_idx, filter, attention_type):
    print('Conv: '+  str(block_idx) + ' filter: ' + str(filter))
    x = input
    x = Conv2D(filters=filter, kernel_size=3, padding='same', activation='elu', name="Conv" + str(block_idx) + ".1")(x)
    x = Conv2D(filters=filter, kernel_size=3, padding='same', activation='elu', name="Conv" + str(block_idx) + ".2")(x)
    if block_idx > 2 :
        x = Conv2D(filters=filter, kernel_size=3, padding='same', activation='elu', name="Conv" + str(block_idx) +".3")(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_" + str(block_idx))(x)
    if block_idx <= 5 :
        if attention_type == "CBAM":
            print('Using CBAM ne')
            attention_output = channel_attention(x, ratio=16)
            attention_output = spatial_attention(attention_output)
            x = layers.Add(name='ConvLast_Add1')([attention_output, x])
        if attention_type == "BAM":
            print('Using BAM ne')
            attention_output = bam(x, reduction_ratio=16, num_layers=1, dilation_conv_num=2, dilation_val=4)
            x = layers.Add(name='ConvLast_Add1')([attention_output, x])
        if attention_type == 'scSE':
            print('Using scSE ne')
            x = scse(x, reduction_ratio=8)

    return x