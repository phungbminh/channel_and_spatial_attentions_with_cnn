from tensorflow.keras.layers import ( AveragePooling2D, GlobalMaxPooling2D, multiply, Permute, Concatenate, Lambda, BatchNormalization, ReLU, Layer)
from keras import backend as K
from tensorflow.keras import layers

#CBAM-BLOCK---------------------------------------------
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    mul = multiply([input_feature, cbam_feature])
    return mul


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    mul = multiply([input_feature, cbam_feature])

    return mul
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
#SCNET-BLOCK---------------------------------------------
class ResizeLayer(Layer):
    def call(self, inputs):
        source, target = inputs
        target_size = tf.shape(target)[1:3]  # Lấy kích thước không gian từ target tensor
        return tf.image.resize(source, size=target_size)
def sc_conv(input_tensor, pooling_r=4):
    """
    SCConv implementation as a function.

    Args:
        input_tensor: Input tensor.
        pooling_r: Reduction ratio for average pooling in the k2 branch.

    Returns:
        Output tensor after applying SCConv.
    """
    # Identity branch
    identity = input_tensor

    # k2 branch: Average pooling -> Conv2D -> BatchNorm -> Resize
    k2 = AveragePooling2D(pool_size=pooling_r, strides=pooling_r)(input_tensor)
    k2 = Conv2D(filters=input_tensor.shape[-1],  # Same number of filters as input channels
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False)(k2)
    k2 = BatchNormalization()(k2)
    #k2_resized = tf.image.resize(k2, size=tf.shape(identity)[1:3])  # Resize to match identity
    k2_resized = ResizeLayer()([k2, identity])

     # Combine identity and k2 with sigmoid
    attention = Add()([identity, k2_resized])
    attention = Activation('sigmoid')(attention)

    # k3 branch: Conv2D -> BatchNorm
    k3 = Conv2D(filters=input_tensor.shape[-1],  # Same number of filters as input channels
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False)(input_tensor)
    k3 = BatchNormalization()(k3)

    # Multiply k3 with attention
    out = Multiply()([k3, attention])

    # k4 branch: Conv2D -> BatchNorm
    out = Conv2D(filters=input_tensor.shape[-1],  # Same number of filters as input channels
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 use_bias=False)(out)
    out = BatchNormalization()(out)

    return out




#BAM-BLOCK---------------------------------------------
def channel_gate(x, reduction_ratio=16, num_layers=1):
    """
    Channel Gate for BAM.

    Args:
        x: Input tensor.
        reduction_ratio: Reduction ratio for the dense layers.
        num_layers: Number of intermediate dense layers.

    Returns:
        Channel attention tensor.
    """
    input_channels = x.shape[-1]
    avg_pool = GlobalAveragePooling2D()(x)  # Global Average Pooling
    for _ in range(num_layers):
        avg_pool = Dense(input_channels // reduction_ratio, use_bias=False)(avg_pool)
        avg_pool = BatchNormalization()(avg_pool)
        avg_pool = ReLU()(avg_pool)
    avg_pool = Dense(input_channels, use_bias=False)(avg_pool)
    avg_pool = Reshape((1, 1, input_channels))(avg_pool)  # Reshape to match input dimensions
    return avg_pool


def spatial_gate(x, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
    """
    Spatial Gate for BAM.

    Args:
        x: Input tensor.
        reduction_ratio: Reduction ratio for the first convolution.
        dilation_conv_num: Number of dilated convolutions.
        dilation_val: Dilation rate for the convolutions.

    Returns:
        Spatial attention tensor.
    """
    input_channels = x.shape[-1]
    s = Conv2D(input_channels // reduction_ratio, kernel_size=1, padding='same', use_bias=False)(x)
    s = BatchNormalization()(s)
    s = ReLU()(s)

    for _ in range(dilation_conv_num):
        s = Conv2D(input_channels // reduction_ratio, kernel_size=3, padding='same', dilation_rate=dilation_val,
                   use_bias=False)(s)
        s = BatchNormalization()(s)
        s = ReLU()(s)

    s = Conv2D(1, kernel_size=1, padding='same', use_bias=False)(s)
    return s


def bam(x, reduction_ratio=16, num_layers=1, dilation_conv_num=2, dilation_val=4):
    """
    Bottleneck Attention Module (BAM).

    Args:
        x: Input tensor.
        reduction_ratio: Reduction ratio for channel gate.
        num_layers: Number of intermediate dense layers in channel gate.
        dilation_conv_num: Number of dilated convolutions in spatial gate.
        dilation_val: Dilation rate for the convolutions in spatial gate.

    Returns:
        Output tensor after applying BAM.
    """
    # Channel Attention
    channel_att = channel_gate(x, reduction_ratio=reduction_ratio, num_layers=num_layers)
    # Spatial Attention
    spatial_att = spatial_gate(x, reduction_ratio=reduction_ratio, dilation_conv_num=dilation_conv_num,
                               dilation_val=dilation_val)
    # Combine Channel and Spatial Attention
    attention = Multiply()([channel_att, spatial_att])
    attention = Activation('sigmoid')(attention)
    # Apply attention to input
    out = Multiply()([x, 1 + attention])
    return out


import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Activation, Multiply, Reshape, Add


def channel_se(input_tensor, reduction_ratio=16):
    """
    Channel Squeeze-and-Excitation (cSE).

    Args:
        input_tensor: Input tensor of shape (batch_size, height, width, channels).
        reduction_ratio: Reduction ratio for the dense layers.

    Returns:
        Output tensor after applying channel attention.
    """
    channels = input_tensor.shape[-1]
    # Global Average Pooling
    squeeze = GlobalAveragePooling2D()(input_tensor)
    # Fully connected layers
    excitation = Dense(channels // reduction_ratio, activation='relu', use_bias=True)(squeeze)
    excitation = Dense(channels, activation='sigmoid', use_bias=True)(excitation)
    # Reshape and scale
    excitation = Reshape((1, 1, channels))(excitation)
    return Multiply()([input_tensor, excitation])

#SCSE-BLOCK---------------------------------------------
def spatial_se(input_tensor):
    """
    Spatial Squeeze-and-Excitation (sSE).

    Args:
        input_tensor: Input tensor of shape (batch_size, height, width, channels).

    Returns:
        Output tensor after applying spatial attention.
    """
    # Convolution to generate spatial attention map
    spatial_attention = Conv2D(1, kernel_size=1, activation='sigmoid', use_bias=True)(input_tensor)
    return Multiply()([input_tensor, spatial_attention])


def scse(input_tensor, reduction_ratio=16):
    """
    Channel and Spatial Squeeze-and-Excitation (scSE).

    Args:
        input_tensor: Input tensor of shape (batch_size, height, width, channels).
        reduction_ratio: Reduction ratio for the channel attention.

    Returns:
        Output tensor after applying scSE.
    """
    # Apply Channel SE
    cse = channel_se(input_tensor, reduction_ratio=reduction_ratio)
    # Apply Spatial SE
    sse = spatial_se(input_tensor)
    # Combine cSE and sSE using max
    return Add()([cse, sse])


