import tensorflow as tf
from tensorflow.keras.layers import (  GlobalAveragePooling2D, Dense, Conv2D, Activation, Multiply, Reshape, Add, AveragePooling2D, GlobalMaxPooling2D, multiply, Permute, Concatenate, Lambda, BatchNormalization, ReLU, Layer)
from keras import backend as K
from tensorflow.keras import layers

#CBAM-BLOCK---------------------------------------------
def cbam_block(input_layer, filter_num, reduction_ratio=16, kernel_size=7, name=None):
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
    input_channel = input_layer.shape[-1]
    num_squeeze = input_channel // reduction_ratio

    axis = -1

    # CHANNEL ATTENTION
    avg_pool = layers.GlobalAveragePooling2D(name=name + "_Channel_AveragePooling_{}".format(input_channel))(input_layer)
    max_pool = layers.GlobalMaxPooling2D(name=name + "_Channel_MaxPooling_{}".format(input_channel))(input_layer)

    # Shared MLP
    dense1 = layers.Dense(num_squeeze, activation='relu', name=name + "_Channel_FC_1_{}".format(input_channel))
    dense2 = layers.Dense(input_channel, name=name + "_Channel_FC_2_{}".format(input_channel))

    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))

    channel = layers.add([avg_out, max_out])
    channel = layers.Activation('sigmoid', name=name + "_Channel_Sigmoid_{}".format(input_channel))(channel)
    channel = layers.Reshape((1, 1, input_channel), name=name + "_Channel_Reshape_{}".format(input_channel))(channel)

    channel_output = layers.multiply([input_layer, channel])

    # SPATIAL ATTENTION
    # avg_pool2 = layers.Lambda(lambda x: tf.reduce_mean(x, axis=axis, keepdims=True))(input_layer)
    # max_pool2 = layers.Lambda(lambda x: tf.reduce_max(x, axis=axis, keepdims=True))(input_layer)
    #
    # spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)
    # #------
    avg_pool2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(input_layer)
    max_pool2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(input_layer)
    spatial = Concatenate(axis=3)([avg_pool2, max_pool2])

    spatial = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=name + "_Spatial_Conv2D_{}".format(input_channel))(spatial)
    spatial_out = layers.Activation('sigmoid', name=name + "_Spatial_Sigmoid_{}".format(input_channel))(spatial)

    #return layers.multiply([channel_output, spatial_out])
    multiply_layer = Multiply(name=name + 'Attention_CBAM_output_layer')([channel_output, spatial_out])
    return multiply_layer

#BAM-BLOCK---------------------------------------------
def bam_block(inputs, filter_num, reduction_ratio=16, num_layers=1, dilation_val=4, name=None):

    input_channel = inputs.shape[-1]
    num_squeeze = input_channel // reduction_ratio

    # Channel attention
    gap = GlobalAveragePooling2D()(inputs)
    channel = Dense(num_squeeze, activation=None,name=name + "_Channel_FC_1")(gap)
    channel = Dense(input_channel, activation=None,name=name + "_Channel_FC_2")(channel)
    channel = BatchNormalization()(channel)

    # Spatial attention
    spatial = Conv2D(num_squeeze, kernel_size=1, padding='same' , name=name + "_Spatial_Conv2D_1")(inputs)
    # spatial = Conv2D(num_squeeze, kernel_size=3, padding='same', dilation_rate=dilation_value)(spatial)
    # spatial = Conv2D(num_squeeze, kernel_size=3, padding='same', dilation_rate=dilation_value)(spatial)
    spatial = Conv2D(num_squeeze, kernel_size=3, padding='same',dilation_rate=4, name=name + "_Spatial_Conv2D_2")(spatial)
    spatial = Conv2D(num_squeeze, kernel_size=3, padding='same',dilation_rate=4, name=name + "_Spatial_Conv2D_3")(spatial)
    spatial = Conv2D(1, kernel_size=1, padding='same', name=name + "_Spatial_Conv2D_4")(spatial)
    spatial = BatchNormalization()(spatial)

    combined = Add(name=name + '_Combined_Layer')([channel, spatial])
    combined = Activation('sigmoid', name=name + "_Combined_Sigmoid")(combined)

    output = Add(name=name + 'Attention_BAM_output_layer')([inputs, inputs * combined])

    return output


def scse_block(inputs, filter_num, reduction_ratio=16, name=None):
    """Squeeze-and-Excitation Block with Spatial Attention."""
    #Squeeze-and-Excitation Layer
    # Global Average Pooling
    input_channels = inputs.shape[-1]
    channel_se = GlobalAveragePooling2D()(inputs)
    # Fully connected layers
    channel_se = Dense(input_channels // reduction_ratio, use_bias=False,name=name + "_Channel_Squeeze_FC_1")(channel_se)  # First FC layer
    channel_se = ReLU()(channel_se)  # ReLU activation
    channel_se = Dense(input_channels, use_bias=False,name=name + "_Channel_Squeeze_FC_2")(channel_se)  # Second FC layer
    channel_se = Activation('sigmoid',name=name + "_Channel_Squeeze_Activation")(channel_se)
    channel_se = Reshape((1, 1, input_channels))(channel_se)
    channel_se = Multiply()([inputs, channel_se])  # Element-wise multiplication

    # Spatial Squeeze and Excitation
    spatial_se = Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=False, name=name + "_Spatial_Squeeze_Conv2D_1")(inputs)
    spatial_se = Activation('sigmoid', name=name + "_Spatial_Squeeze_Activation")(spatial_se)
    spatial_se = Multiply()([inputs, spatial_se])  # Element-wise multiplication

    output = Add(name=name + 'Attention_SCSE_output_layer')([channel_se, spatial_se])
    return output



