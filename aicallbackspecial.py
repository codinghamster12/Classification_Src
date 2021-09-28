
import os
import tensorflow as tf
import pandas as pd
import collections

class CallbackSpecial(tf.keras.callbacks.Callback):
    
    def __init__(self, _ais):
        super().__init__()

        self.ais = _ais
        self.metrics_dict = collections.OrderedDict([('loss', 'loss'), ('val_loss', 'val_loss'), ('acc', 'acc'), ('val_acc', 'val_acc')])
        self.dir_res_full_name = 'dir_res_full'

    def get_new_row(self, epoch, logs):
        new_row = {}
        new_row['dir_res_short'] = os.path.basename(self.ais.dir_res)
        new_row['model.base_name'] = self.ais.model.base_name
        new_row['model.model.name'] = self.ais.model.model.name
        new_row['model.frozen'] = self.ais.model.frozen
        new_row['model.loaded'] = self.ais.model.loaded
        new_row['learning_rate'] = self.ais.learning_rate
        if self.ais.start_model_file is None:
            new_row['start_model_file_short'] = 'None'
        else: 
            new_row['start_model_file_short'] = os.path.basename(os.path.dirname(self.ais.start_model_file))
        for metric_name in self.metrics_dict:
            value = logs[self.metrics_dict[metric_name]]
            new_row[metric_name] = value
        new_row['final_best_model_loss'] = None
        new_row['final_best_model_acc'] = None
        new_row['epoch'] = epoch
        new_row['num_epochs'] = self.ais.num_epochs
        new_row['batch_size'] = self.ais.batch_size
        new_row['img_dirbase_short'] = os.path.basename(self.ais.img_dirbase)
        new_row['image_size'] = self.ais.image_size
        new_row['regression'] = self.ais.regression
        new_row['image_size_2'] = str(self.ais.image_size_2)
        new_row['input_shape'] = str(self.ais.input_shape)
        new_row['random_seed'] = self.ais.random_seed
        new_row['img_subdir_train'] = self.ais.img_subdir_train
        new_row['img_subdir_test'] = self.ais.img_subdir_test
        new_row['img_extension'] = self.ais.img_extension
        new_row['start_model_file_full'] = self.ais.start_model_file
        new_row['img_dirbase_full'] = self.ais.img_dirbase
        new_row[self.dir_res_full_name] = self.ais.dir_res
        return new_row

    def _add_md5_entries(self, new_row):
        from diutil import DiUtil
        new_row['md5_Calc01'] = DiUtil.calc_md5('Calc01.py')
        new_row['md5_aicallbacks'] = DiUtil.calc_md5('aicallbacks.py')
        new_row['md5_aicallbackspecial'] = DiUtil.calc_md5('aicallbackspecial.py')
        new_row['md5_aidata'] = DiUtil.calc_md5('aidata.py')
        new_row['md5_aidataaugmentation'] = DiUtil.calc_md5('aidataaugmentation.py')
        new_row['md5_aimodel'] = DiUtil.calc_md5('aimodel.py')
        new_row['md5_aireport'] = DiUtil.calc_md5('aireport.py')
        new_row['md5_aisettings'] = DiUtil.calc_md5('aisettings.py')
        new_row['md5_classification_utils'] = DiUtil.calc_md5('classification_utils.py')
        new_row['md5_diutil'] = DiUtil.calc_md5('diutil.py')
        new_row['md5_FunctionsDI01'] = DiUtil.calc_md5('FunctionsDI01.py')
        new_row['md5_keras_custom_callbacks'] = DiUtil.calc_md5('keras_custom_callbacks.py')
        return new_row

    def on_train_begin(self, logs=None):
        print("Training: \033[92mstart\033[0m.")

    def on_train_end(self, logs=None):
        print("Training: \033[91mend\033[0m.")

    def on_epoch_end(self, epoch, logs={}):
        fnres = os.path.join(self.ais.dir_res, '../ressummary.txt')
        new_row = self.get_new_row(epoch, logs)
        if os.path.exists(fnres):
            df = pd.read_csv(fnres, sep='\t')
            if not self.ais.dir_res in list(df[self.dir_res_full_name]):
                new_row = self._add_md5_entries(new_row)
                df_new_row = pd.DataFrame(new_row, index = [0])
                df = df.append(df_new_row, ignore_index=True)
            else:
                for k, v in new_row.items():
                    df[k].loc[df[self.dir_res_full_name] == self.ais.dir_res] = v
        else:
            new_row = self._add_md5_entries(new_row)
            df = pd.DataFrame(new_row, index = [0])
        df.to_csv(fnres, sep='\t', index=False)