# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import os
import aicallbackspecial

class AiCallbacks:
    # class or static variables

    def __init__(self):
        pass

    @staticmethod
    def get_callbacks(ais):
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
        # from keras_custom_callbacks import SimpleLogCallback
        import collections

        callbacks = []

        # Callback to interrupt the training if the validation loss/metrics stops improving for some epochs:
        # cb = EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True)
        # cb = EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=False)
        cb = EarlyStopping(patience=32, monitor='val_loss', restore_best_weights=True)
        # cb = EarlyStopping(patience=64, monitor='val_loss', restore_best_weights=True)
        callbacks.append(cb)

        # Callback to log the graph, losses and metrics into TensorBoard:
        # cb = TensorBoard(log_dir=ais.dir_tensorboard, histogram_freq=0, write_graph=True)
        # cb = TensorBoard(log_dir=ais.dir_tensorboard, histogram_freq=0, write_graph=False)
        # callbacks.append(cb)
        # keras.callbacks.TensorBoard(
        #     log_dir="/full_path_to_your_logs",
        #     histogram_freq=0,  # How often to log histogram visualizations
        #     embeddings_freq=0,  # How often to log embedding visualizations
        #     update_freq="epoch",
        # )  # How often to write logs (default: once per epoch)

        # Callback to simply log metrics at the end of each epoch (saving space compared to verbose=1/2):
        # metrics_to_print = collections.OrderedDict([('loss', 'loss'), ('v-loss', 'val_loss'),
        #                                             ('acc', 'acc'), ('v-acc', 'val_acc')])
        # cb = SimpleLogCallback(metrics_to_print, num_epochs=ais.num_epochs, log_frequency=1),
        # callbacks.append(cb)

        # Callback to save the model (e.g., every 5 epochs), specifying the epoch and val-loss in the filename:
        # nach save_freq immer mal sichern...
        # save_freq = 5
        # cb = ModelCheckpoint(os.path.join(ais.dir_res, 'model_last_p{}.h5'.format(save_freq)), save_freq = save_freq)
        # callbacks.append(cb)

        # nach save_freq immer neu rausschreiben - anderer Dateiname
        # epochs_freq = 45
        # cb = ModelCheckpoint(os.path.join(ais.dir_res, 'model_e{epoch:03d}'), save_freq = (epochs_freq * ais.batch_size))
        # callbacks.append(cb)

        # bestes
        # cb = ModelCheckpoint(ais.model_best_loss, save_best_only = True)
        # callbacks.append(cb)
        cb = ModelCheckpoint(ais.model_best_loss, save_best_only = True)
        callbacks.append(cb)

        cb = ModelCheckpoint(ais.model_best_acc, save_best_only = True, monitor='val_acc', )
        callbacks.append(cb)

        # ModelCheckpoint(
        #     filepath, monitor='val_loss', verbose=0, save_best_only=False,
        #     save_weights_only=False, mode='auto', save_freq='epoch',
        #     options=None, **kwargs
        # )


        # Metriken in Dateien schreiben
        cb = CSVLogger(os.path.join(ais.root_dir, 'Metrics.txt'), separator='\t', append=True)
        callbacks.append(cb)
        cb = CSVLogger(os.path.join(ais.dir_res, 'Metrics.txt'), separator='\t', append=True)
        callbacks.append(cb)

        cb = aicallbackspecial.CallbackSpecial(ais)
        callbacks.append(cb)

        return callbacks

