import tensorflow as tf
from typing import Tuple

class ConvMixer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size:Tuple[int, int],
                 strides:Tuple[int, int],
                 activation="gelu",
                 data_format=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvMixer, self).__init__(**kwargs)
        
        self.filters                =   filters
        self.kernel_size            =   kernel_size
        self.activation_name        =   activation
        self.strides                =   strides
        self.data_format            =   data_format
        self.kernel_initializer     =   kernel_initializer
        self.bias_initializer       =   bias_initializer
        self.kernel_regularizer     =   kernel_regularizer
        self.bias_regularizer       =   bias_regularizer
        self.activity_regularizer   =   activity_regularizer
        self.kernel_constraint      =   kernel_constraint
        self.bias_constraint        =   bias_constraint

        self.depthwise = None
        self.activation1 = None
        self.add = None
        self.pointwise = None
        self.activation2 = None
    
    def build(self, input_shape):

        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size             =   self.kernel_size, 
            strides                 =   self.strides, 
            padding                 =   "same",
            data_format             =   self.data_format,
            kernel_initializer      =   self.kernel_initializer,
            bias_initializer        =   self.bias_initializer,
            kernel_regularizer      =   self.kernel_regularizer,
            bias_regularizer        =   self.bias_regularizer,
            activity_regularizer    =   self.activity_regularizer,
            kernel_constraint       =   self.kernel_constraint,
            bias_constraint         =   self.bias_constraint,
            name                    =   "depthwise_convmixer"
            )
        
        self.activation1 = tf.keras.layers.Activation(self.activation_name)

        self.add = tf.keras.layers.Add()

        self.pointwise = tf.keras.layers.Conv2D(
            filters                 =   self.filters, 
            kernel_size             =   1,
            data_format             =   self.data_format,
            kernel_initializer      =   self.kernel_initializer,
            bias_initializer        =   self.bias_initializer,
            kernel_regularizer      =   self.kernel_regularizer,
            bias_regularizer        =   self.bias_regularizer,
            activity_regularizer    =   self.activity_regularizer,
            kernel_constraint       =   self.kernel_constraint,
            bias_constraint         =   self.bias_constraint,
            name                    =   "pointwise_convmixer")
        
        self.activation2 = tf.keras.layers.Activation(self.activation_name)

    def call(self, x, training=None):
        x0 = x
        x = self.depthwise(x)
        x = self.activation1(x)
        x = self.add([x, x0])
        x = self.pointwise(x)
        x = self.activation2(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

    def get_config(self):
        config = {
            "filters"               :   self.filters,
            "kernel_size"           :   self.kernel_size,
            "strides"               :   self.strides,
            "data_format"           :   self.data_format,
            "activation_name"       :   self.activation,
            "kernel_initializer"    :   self.kernel_initializer,
            "bias_initializer"      :   self.bias_initializer,
            "kernel_regularizer"    :   self.kernel_regularizer,
            "bias_regularizer"      :   self.bias_regularizer,
            "activity_regularizer"  :   self.activity_regularizer,
            "kernel_constraint"     :   self.kernel_constraint,
            "bias_constraint"       :   self.bias_constraint,
        }
        base_config = super(ConvMixer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class Patcher(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 patch_size:Tuple[int, int],
                 activation="relu",
                 data_format=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Patcher, self).__init__(**kwargs)
        
        self.filters                =   filters
        self.patch_size            =   patch_size
        self.activation_name        =   activation
        self.data_format            =   data_format
        self.kernel_initializer     =   kernel_initializer
        self.bias_initializer       =   bias_initializer
        self.kernel_regularizer     =   kernel_regularizer
        self.bias_regularizer       =   bias_regularizer
        self.activity_regularizer   =   activity_regularizer
        self.kernel_constraint      =   kernel_constraint
        self.bias_constraint        =   bias_constraint

        self.conv = None
        self.activation = None
    
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters                 =   self.filters,
            kernel_size             =   self.patch_size,
            strides                 =   self.patch_size,
            data_format             =   self.data_format,
            kernel_initializer      =   self.kernel_initializer,
            bias_initializer        =   self.bias_initializer,
            kernel_regularizer      =   self.kernel_regularizer,
            bias_regularizer        =   self.bias_regularizer,
            activity_regularizer    =   self.activity_regularizer,
            kernel_constraint       =   self.kernel_constraint,
            bias_constraint         =   self.bias_constraint,
            name                    =   "patcher_conv"
        )
        
        self.activation = tf.keras.layers.Activation(self.activation_name)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

    def get_config(self):
        config = {
            "filters"               :   self.filters,
            "patch_size"           :   self.patch_size,
            "strides"               :   self.strides,
            "data_format"           :   self.data_format,
            "activation_name"       :   self.activation,
            "kernel_initializer"    :   self.kernel_initializer,
            "bias_initializer"      :   self.bias_initializer,
            "kernel_regularizer"    :   self.kernel_regularizer,
            "bias_regularizer"      :   self.bias_regularizer,
            "activity_regularizer"  :   self.activity_regularizer,
            "kernel_constraint"     :   self.kernel_constraint,
            "bias_constraint"       :   self.bias_constraint,
        }
        base_config = super(Patcher, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))