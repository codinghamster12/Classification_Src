# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as tf_preprocessing

class AiDataAugmentation:
    # class or static variables

    def __init__(self):
        pass

    @staticmethod
    def get_data_augmentation():
        data_augmentation = tf.keras.Sequential([
            # tf_preprocessing.RandomFlip('horizontal'),
            # tf_preprocessing.RandomFlip('horizontal_and_vertical'),
            # RandomRotation(0.5) bedeutet von -180 bis +180 Grad
            # tf_preprocessing.RandomRotation(0.5),
            tf_preprocessing.RandomRotation(0.01),
            tf_preprocessing.RandomContrast(0.1),
            tf_preprocessing.RandomZoom(0.01),
            tf_preprocessing.RandomTranslation(0.01, 0.01)
        ])
        return data_augmentation
