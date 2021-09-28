# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import platform
import os
import sys
from datetime import datetime
from diutil import DiUtil
from aidata import  AiData
from aimodel import  AiModel
from aireport import  AiReport


class AiSettings:
    # class or static variables

    def __init__(self, _main_file, _sub_dir = None):
        # _sub_dir vorhanden: keine neue analyse sondern z.B. Report

        self._dir_admistration(_main_file, _sub_dir)

        # create logger
        if _sub_dir is None:
            self.logger = DiUtil.get_logger('AI', [self.root_dir, self.dir_res])
        else:
            self.logger = DiUtil.get_logger('AI', [self.dir_res], _log_fn_name = 'Report.txt')
        self.p('==========================================')
        self.p('Platform (OS): {}'.format(platform.system()))
        self.p('Python Version: {}'.format(sys.version))
        self.p('root_dir: {}'.format(self.root_dir))
        self.p('dir_res: {}'.format(self.dir_res))

        self.copy_files()

        # Some hyper-parameters:
        # start_model_file: Datei von start-Modell
        self.freeze_model = False # ob layers von Basismodell freezed werden (in aimodel)
        self.start_model_file = None
        # self.start_model_file = os.path.join(self.root_dir, 'Res_2021-01-25_23-05-08', 'model_best_loss.h5')
        if _sub_dir is not None:
            # self.start_model_file = os.path.join(self.root_dir, _sub_dir, 'model_best_loss')
            self.start_model_file = os.path.join(self.root_dir, _sub_dir, 'model_best_loss.h5')
        self.learning_rate = 1.0e-04
        self.num_epochs  = 10#3000           # Max number of training epochs
        self.batch_size  = 16       # 64 VM Images per batch (reduce/increase according to the machine's capability)
        # self.image_size = 331 # NasNetLarge 
        self.image_size = 224 
        self.regression = False
        self.image_size_2 = (self.image_size, self.image_size)
        self.input_shape = [self.image_size, self.image_size, 3]        
        self.random_seed = 42            # Seed for some random operations, for reproducibility
        self.img_dirbase = 'd:\\Di\\Z_InfosTestTry\\Z_Programmier\\Python\\P37_TensorFlow2_Samples\\Classification\\data\\d01'
        self.img_subdir_train = 'train'
        self.img_subdir_test = 'test'
        self.img_extension = '*.jpg'

        self.p('start_model_file: {}'.format(self.start_model_file))
        self.p('freeze_model: {}'.format(self.freeze_model))
        self.p('learning_rate: {}'.format(self.learning_rate))
        self.p('num_epochs: {}'.format(self.num_epochs))
        self.p('batch_size: {}'.format(self.batch_size))
        self.p('image_size: {}'.format(self.image_size))
        self.p('regression: {}'.format(self.regression))
        self.p('image_size_2: {}'.format(self.image_size_2))
        self.p('input_shape: {}'.format(self.input_shape))
        self.p('random_seed: {}'.format(self.random_seed))

        # self.model_best_loss = os.path.join(self.dir_res, 'model_best_loss') # zum Speichern ueber Callback und verwenden in Report
        self.model_best_loss = os.path.join(self.dir_res, 'model_best_loss.h5') # zum Speichern ueber Callback und verwenden in Report
        self.model_best_acc = os.path.join(self.dir_res, 'model_best_acc.h5') # zum Speichern ueber Callback und verwenden in Report

        self.data = AiData(self)
        self.model = AiModel()
        self.report = AiReport(self)

    # Verzeichnisse...
    def _dir_admistration(self, _main_file, _sub_dir):
        # print(_main_file)
        self.root_dir = os.path.dirname(os.path.realpath(_main_file))
        if _sub_dir is not None:
            self.dir_res = os.path.join(self.root_dir, _sub_dir)
            return

        print(self.root_dir)
        if self.root_dir != '':
            os.chdir(self.root_dir) 
        basendir = 'Res_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.dir_res = os.path.join(self.root_dir, basendir)
        DiUtil.make_dir(self.dir_res)
        self.dir_tensorboard = os.path.join(self.root_dir, 'tb')
        import shutil
        if os.path.exists(self.dir_tensorboard): shutil.rmtree(self.dir_tensorboard)
        DiUtil.make_dir(self.dir_tensorboard)

    # logger print Hilfsfunktion
    def p(self, _str):
        self.logger.info(_str)  

    def copy_files(self):
        from shutil import copyfile
        import glob
        file_list = glob.glob(os.path.join(self.root_dir, '*.py'))
        dir_src_copy = os.path.join(self.dir_res, 'src')
        DiUtil.make_dir(dir_src_copy)
        for f in file_list:
            copyfile(f, os.path.join(dir_src_copy, os.path.basename(f)))

    # Holt Daten
    # def get_dataset(self):
    #     return self.data.train_dataset, self.data.test_dataset

    # Holt Daten als Generatoren
    def get_datagenerator(self):
        return self.data.train_generator, self.data.test_generator

    # Setzt callbacks
    def get_data_augmentation(self):
        from aidataaugmentation import  AiDataAugmentation
        return AiDataAugmentation.get_data_augmentation()

    # Setzt Optimizer
    def get_optimizer(self):
        import tensorflow.keras.optimizers as tf_optimizers

        # optimizer = tf_optimizers.SGD(learning_rate = self.learning_rate,
        #                                     momentum = 0.9, 
        #                                     nesterov = True)

        # optimizer=tf_optimizers.RMSprop(lr = self.learning_rate) 

        # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

        # decayed_lr = tf_optimizers.schedules.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
        # optimizer = tf_optimizers.Adam(lr = decayed_lr) #, epsilon=adam_epsilon)

        optimizer = tf_optimizers.Adam(lr = self.learning_rate)

        return optimizer

    # Erstell Modell
    def load_model(self):
        self.model.load_model(self)
        self.p('model.base_name: {}'.format(self.model.base_name))
        self.p('model.model.name: {}'.format(self.model.model.name))
        self.p('model.frozen: {}'.format(self.model.frozen))
        self.p('model.loaded: {}'.format(self.model.loaded))

    # Setzt callbacks
    def get_callbacks(self):
        from aicallbacks  import  AiCallbacks
        return AiCallbacks.get_callbacks(self)







