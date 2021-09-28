# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from aidata import  AiData
import numpy as np

class AiModel:
    # class or static variables

    def __init__(self):
        self.model = None
        self.frozen = None # ob Modell layers gefrozen wurden (wird in aimodel gesetzt)
        self.base_name = None # Modellname Basis, nur wenn neues Modell (wird in aimodel gesetzt)
        self.loaded = None # ob Modell von extern (Datei) geladen wurde (wird in aimodel gesetzt)

    def load_model_from_disk(self, _model_path):
        self.model = tf.keras.models.load_model(_model_path)
        self.loaded = False            

    def load_model(self, ais):

        if ais.start_model_file is not None:
            self.load_model_from_disk(ais.start_model_file)
            self.loaded = True
            if not ais.freeze_model:
                for layer in self.model.layers:
                    layer.trainable = True
                self.frozen = False
            return
        self.loaded = False

        # base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
        #                                                include_top=False,
        #                                                weights='imagenet')

        base_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights='imagenet', input_shape = tuple(ais.input_shape))
        self.base_name = base_model.name
        preprocess_input = tf.keras.applications.resnet50.preprocess_input

        # base_model = tf.keras.applications.InceptionResNetV2 (
        #     include_top=False, weights='imagenet', input_shape = tuple(ais.input_shape))
        # self.base_name = base_model.name
        # preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

        # base_model = tf.keras.applications.ResNet101(
        #     include_top=False, weights='imagenet', input_shape = tuple(ais.input_shape))
        # self.base_name = base_model.name
        # preprocess_input = tf.keras.applications.resnet.preprocess_input

        # base_model = tf.keras.applications.NASNetLarge(
        #     include_top=False, weights='imagenet', input_shape = tuple(ais.input_shape))
        # self.base_name = base_model.name
        # preprocess_input = tf.keras.applications.nasnet.preprocess_input

        # base_model = tf.keras.applications.efficientnet.EfficientNetB0(
        #     include_top=False, weights='imagenet', input_shape = tuple(ais.input_shape))
        # self.base_name = base_model.name
        # preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        # from tensorflow.keras import models
        # from tensorflow.keras import layers
        # model = models.Sequential()

        # preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        # model.add(tf.keras.layers.Lambda(preprocess_input, name='preprocessing', input_shape=(224, 224, 3)))

        # data_augmentation = ais.get_data_augmentation()
        # model.add(tf.keras.layers.Lambda(data_augmentation, name='augmentation'))

        # model.add(base_model)
        # model.add(layers.GlobalMaxPooling2D(name="gap"))
        # # model.add(layers.Flatten(name="flatten"))
        # model.add(layers.Dropout(0.5, name="dropout_out"))
        # # model.add(layers.Dense(256, activation='relu', name="fc1"))
        # model.add(layers.Dense(ais.data.num_classes, activation='softmax', name="fc_out"))

        model_name = base_model.name
        model_name_suffix = '_full'

        self.frozen = False
        if ais.freeze_model:
            frozen_layers, trainable_layers = [], []
            model_name_suffix =  '_frozen'
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer.trainable = False
                    frozen_layers.append(layer.name)
                else:
                    if len(layer.trainable_weights) > 0:
                        # We list as 'trainable' only the layers with trainable parameters.
                        trainable_layers.append(layer.name)
            self.frozen = True
        model_name += model_name_suffix

        # # Logging the lists of frozen/trainable layers:
        # log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\n\033[94m', '\033[92m'
        # log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
        # log_end_format = '\033[0m'
        # ais.p('{2}Layers we froze:{4} {0} ({3}total = {1}{4}).'.format(
        #     frozen_layers, len(frozen_layers), log_begin_red, log_begin_bold, log_end_format))
        # ais.p('\n{2}Layers which will be fine-tuned:{4} {0} ({3}total = {1}{4}).'.format(
        #     trainable_layers, len(trainable_layers), log_begin_blue, log_begin_bold, log_end_format))

        inputs = tf.keras.Input(shape=ais.input_shape)

        data_augmentation = ais.get_data_augmentation()
        x = data_augmentation(inputs)
        x = preprocess_input(x)

        # x = preprocess_input(inputs)

        x = base_model(x)
        # x = base_model(x, training=False)
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        inter_layser_size = 1024
        x = Dense(inter_layser_size, activation='relu')(x)
        model_name += '_ils{}'.format(inter_layser_size)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = Dense(ais.data.num_classes, activation='softmax')(x)

        self.model = Model(inputs, predictions, name = model_name)

        # self.model = model

    def predict_image(self, ais, img_path):
        # Read the image and resize it
        from tensorflow.keras.preprocessing import image
        img = image.load_img(img_path, target_size=ais.image_size_2)
        # Convert it to a Numpy array with target shape.
        x = image.img_to_array(img)
        # Reshape
        x = x.reshape((1,) + x.shape)
        result = self.model.predict([x])
        result_pred = np.argmax(result, axis=1)
        return result_pred[0]

    def predict_dir(self, ais, dir):
        dataset = AiData.get_dataset_from_dir(dir, ais, False)
        predictions = self.model.predict(dataset)
        # image_batch, label_batch = dataset.as_numpy_iterator().next()
        # predictions = self.model.predict_on_batch(image_batch).flatten()
        ais.p('Predictions: {}'.format(predictions))
        # ais.p('Labels:\n', label_batch)
        
        class_predicted = np.argmax(predictions, axis=1)
        ais.p('Predictions: {}'.format(class_predicted))

        return


        labels = (dataset.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        ClassesTrue = dataset.classes
        return(ClassesPredict, ClassesTrue, generator.filenames, labels, predictFull)


        from matplotlib import pyplot as plt



        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(min(9, ais.batch_size)):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                # plt.title(ais.data.class_names[labels[i]])
                plt.title('p: {}'.format(predictions[i]))
                # plt.title('p{}'.format(predictions[i], labels[i])
                plt.axis('off')
        # fn = os.path.join(ais.dir_res, 'ImgSamples.png')
        # plt.savefig(fn)
        plt.show()
        # plt.close()

        # plt.figure(figsize=(10, 10))
        # for i in range(min(9, len(image_batch))):
        #     ax = plt.subplot(3, 3, i + 1)
        #     plt.imshow(image_batch[i].astype("uint8"))
        #     plt.title([predictions[i]])
        #     # plt.title(class_names[predictions[i]])
        #     plt.axis("off")





