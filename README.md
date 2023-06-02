# convmixer
Conv-Mixer custom layer implementation in Tensorflow/Keras.
ConvMixer is introduced in ICLR 2022 submission ["Patches Are All You Need?"](https://openreview.net/forum?id=TVHS5Y4dNvM) by Asher Trockman and Zico Kolter.


## How to Use
Example code is provided in example.py file.

First layer should be patcher, so patches can be extracted
```python
inputs = tf.keras.layers.Input((w, h, c))
patches = Patcher(filters = 128, patch_size = (4, 4))(inputs)
```

Conv mixer layers are added sequentially with a certain "depth"
```python
t = patches
for i in range(depth):
    t = ConvMixer(256, (3, 3), (1, 1), "gelu")(t)
```

Then the features are ready to feed other layers. In this example i used Global Average Pooling and FC to perform classsification. 
```python
final_conv = tf.keras.layers.Conv2D(n_classes, 
                                    kernel_size=1, 
                                    strides=1, 
                                    padding="same",
                                    activation="relu")(t)
outputs = tf.keras.layers.GlobalAveragePooling2D()(final_conv)
outputs = tf.keras.layers.Dense(n_classes, "softmax")(outputs)
model = tf.keras.Model(inputs, outputs)
```