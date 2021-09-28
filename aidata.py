# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
from diutil import DiUtil
import glob

class AiData:
    # class or static variables

    def __init__(self, ais):
        self._read_data(ais)

    @staticmethod
    def get_dataset_from_dir(dir, ais, shuffle):
        dataset = image_dataset_from_directory(dir,
                                               shuffle=shuffle,
                                               # color_mode='grayscale',
                                               batch_size=ais.batch_size,
                                               image_size=ais.image_size_2)
        return dataset
        
    @staticmethod
    def get_datagenerator_from_dir(dir, ais, shuffle,_use_data_augmentation = False):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator()
        if _use_data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=36,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1)
        generator = datagen.flow_from_directory(
                dir,
                shuffle = shuffle,
                target_size=ais.image_size_2,
                batch_size=ais.batch_size,
                class_mode='categorical')
        return generator
        
    def _read_data(self, ais):
        p = ais.p

        p('===== DATA =====')
        p('img_dirbase: {}'.format(ais.img_dirbase))
        p('img_subdir_train: {}'.format(ais.img_subdir_train))
        p('img_subdir_test: {}'.format(ais.img_subdir_test))
        
        self.train_dir = os.path.join(ais.img_dirbase, ais.img_subdir_train)
        self.test_dir = os.path.join(ais.img_dirbase, ais.img_subdir_test)

        # self.train_dataset = self.get_dataset_from_dir(self.train_dir, ais, shuffle=True)
        # self.test_dataset = self.get_dataset_from_dir(self.test_dir, ais, shuffle=False)

        self.train_generator = self.get_datagenerator_from_dir(self.train_dir, ais, shuffle=True, _use_data_augmentation=False)
        self.test_generator = self.get_datagenerator_from_dir(self.test_dir, ais, shuffle=False, _use_data_augmentation=False)

        if self.test_generator.class_indices != self.train_generator.class_indices:
            raise NameError('Error HL #61: class_indices differ in train and test!\nTrain: {}\nTest: {}'.format(self.train_generator.class_indices, self.test_generator.class_indices))

        self.class_names = [x for x in self.train_generator.class_indices]
        p('class_names: {}'.format(self.class_names))
        self.num_classes = len(self.class_names)
        p('num_classes: {}'.format(self.num_classes))

        import filecmp
        self.num_train_imgs = len(self.train_generator.filenames)
        self.num_test_imgs = len(self.test_generator.filenames)
        p('Train images #: {}'.format(self.num_train_imgs))
        p('Test images #: {}'.format(self.num_test_imgs)) 

        self.TrainClassesCount = {}
        self.TestClassesCount = {}
        for cdn in self.class_names:
            self.TrainClassesCount[cdn] = len(glob.glob(os.path.join(self.train_dir, cdn, ais.img_extension)))
            self.TestClassesCount[cdn] = len(glob.glob(os.path.join(self.test_dir, cdn, ais.img_extension)))
        p('TrainClassesCount: {}'.format(self.TrainClassesCount))
        p('TestClassesCount: {}'.format(self.TestClassesCount)) 

        p('Check for  equal images in test and training... SKIPPED ...')
        # p('Check for  equal images in test and training... may take a while...')
        # #erst mal MD5 -> deutliche Beschleuningung
        # fnsTrainMd5 = {}
        # for fn in fnsTrain:
        #     fnsTrainMd5[fn] = DiUtil.calc_md5(fn)
        # fnsTestMd5 = {}
        # for fn in fnsTest:
        #     fnsTestMd5[fn] = DiUtil.calc_md5(fn)
        # # check if some files are equal in test and training set
        # for k1, v1 in fnsTrainMd5.items():
        #     for k2, v2 in fnsTestMd5.items():
        #         if (v1 == v2):
        #             if (filecmp.cmp(k1, k2, shallow=False)):
        #                 raise NameError(
        #                     'Error HL: Identical files in test and training set, at least one:\nTraining: {}\nTest: {}'.format(k1, k2))
        # p('No equal images in test and training images - Ok!')
        # Use buffered prefetching to load images from disk without having I/O become blocking.
        # To learn more about this method see the data performance guide.
        # AUTOTUNE = tf.data.experimental.AUTOTUNE
        # self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        # self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

        p('==========')
