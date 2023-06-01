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
            "use_bias"              :   self.use_bias,
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
    

if __name__ == "__main__":
    def build_model(n_classes = 10, depth = 3):
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        t = inputs
        for i in range(depth):
            t = ConvMixer(256, (3, 3), (1, 1), "gelu")(t)
        final_conv = tf.keras.layers.Conv2D(n_classes, 
                                            kernel_size=1, 
                                            strides=1, 
                                            padding="same",
                                            activation="relu")(t)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(final_conv)
        outputs = tf.keras.layers.Dense(n_classes, "softmax")(outputs)
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_datasets():
        import tensorflow_datasets as tfds
        train_dataset = tfds.load("mnist", 
            split="train", 
            batch_size=32, 
            shuffle_files=True,
            as_supervised=True)
        val_dataset = tfds.load("mnist", 
            split="test", 
            batch_size=32, 
            shuffle_files=True,
            as_supervised=True)
        # test_dataset = tfds.load("mnist", split="test", batch_size=32, shuffle_files=True)
        return train_dataset, val_dataset
    
    model = build_model(
        n_classes=10,
        depth = 3)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
    
    train_dataset, val_dataset = get_datasets()

    model.fit(
        train_dataset,
        validation_data = val_dataset, 
        epochs = 10)