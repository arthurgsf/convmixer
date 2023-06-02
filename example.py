import tensorflow as tf
from convmixer import ConvMixer, Patcher

if __name__ == "__main__":
    def build_model(n_classes = 10, depth = 3):
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        patches = Patcher(128, (4, 4))(inputs)
        t = patches
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
        return train_dataset, val_dataset
    
    model = build_model(
        n_classes=10,
        depth = 3)
    model.summary()
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
    
    train_dataset, val_dataset = get_datasets()

    model.fit(
        train_dataset,
        validation_data = val_dataset, 
        epochs = 10)