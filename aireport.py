# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
import os
from aidata import  AiData
from aimodel import  AiModel
import aisettings

class AiReport:
    # class or static variables

    def __init__(self, ais):
        # Kann noch nichts machen, da bei MNist01 noch kein Modell vorliegt beim Anlegen des reports in AiSettings
        pass

    # def SaveToCsv(filenames, PredictedValue, TrueValue, fncsv):
    #     results=pd.DataFrame({"Filename":filenames,
    #                         "PredictedValue":PredictedValue,
    #                         "TrueValue":TrueValue})
    #     results.to_csv(fncsv, sep='\t', index=True)
    #     copyfile(fncsv, os.path.join(DirRes, fncsv))

    # def CalcResultsOfSet(pSet):
    #     dir = os.path.join(imgdirds, pSet) 
    #     ClassesPredict, ClassesTrue, filenames, labels, predictFull = PredictDir(model, dir, image_size, regression)
    #     predictValue = ClassesPredict
    #     if regression:
    #         # 1, 2, 3, 4
    #         predictValue = predictFull[:, 0]
    #     SaveToCsv(filenames, predictValue, ClassesTrue, "results{}.txt".format(pSet))
    #     ClassesPredict = predictValue.round().astype(int)
    #     # # 0, 1, 2, 3 ->  1, 2, 3, 4
    #     # ClassesTrue = 1 + ClassesTrue
    #     plftmp, acc = ClassificationReport(logger, ClassesTrue, ClassesPredict, pImagePaths=None, pTargetNames=None)


    def create_report(self, ais):
        
        p = ais.p

        p('===== REPORT =====')
        p('model_best_loss: {}'.format(ais.model_best_loss))
        p('(model_best_acc: {})'.format(ais.model_best_acc))

        # eigenes AiSettings
        self.model = AiModel()
        self.model.load_model_from_disk(ais.model_best_loss)
        p(self.model.model.summary())

        p('img_dirbase: {}'.format(ais.img_dirbase))
        p('img_subdir_test: {}'.format(ais.img_subdir_test))

        from aidata import AiData

        test_generator = AiData.get_datagenerator_from_dir(ais.data.test_dir, ais, shuffle=False)
        test_generator.reset()
        predictions = self.model.model.predict(test_generator)

        import numpy as np
        PredIndex = np.argmax(predictions, axis=1)
        # p('PredIndex: {}'.format(PredIndex))

        labels = (test_generator.class_indices)
        p('labels: {}'.format(labels))
        # p('labels: {}'.format(type(labels)))
        # p('labels: {}'.format(labels==labels))

        labels = dict((v,k) for k,v in labels.items())
        # p('labels: {}'.format(labels))
        NominalIndex = test_generator.classes
        # p('NominalIndex: {}'.format(NominalIndex))

        PredClass = [labels[k] for k in PredIndex]
        NominalClass = [labels[k] for k in NominalIndex]

        CorrIndex = np.equal(NominalIndex, PredIndex)

        preds = [item_pred[item_index] for item_pred, item_index in zip(predictions, PredIndex)]
        # preds0: Probability class 0, nur sinnvoll bei 2-Klassen-Problem
        preds0 = [item_pred[0] for item_pred in predictions]

        import pandas as pd
        filenames = test_generator.filenames
        results=pd.DataFrame({'Filename': filenames,
                            'Prob': preds,
                            'Prob0': preds0, # nur sinnvoll bei 2-Klassen-Problem
                            'PredIndex': PredIndex,
                            'PredClass': PredClass,
                            'NominalIndex': NominalIndex,
                            'NominalClass': NominalClass,
                            'CorrIndex': CorrIndex})
        results.to_csv(os.path.join(ais.dir_res, 'results.txt'), sep='\t', index=False)

        from FunctionsDI01 import ClassificationReport
        _, acc = ClassificationReport(ais.logger, NominalIndex, PredIndex, pImagePaths=None, pTargetNames=None)
        p('acc: {}'.format(acc))
        



# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch).flatten()

# # Apply a sigmoid since our model returns logits
# predictions = tf.nn.sigmoid(predictions)
# predictions = tf.where(predictions < 0.5, 0, 1)

# print('Predictions:\n', predictions.numpy())
# print('Labels:\n', label_batch)

# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(image_batch[i].astype("uint8"))
#   plt.title(class_names[predictions[i]])
#   plt.axis("off")

#         p("TRAIN")
#         CalcResultsOfSet('Train')

#         p("TEST")
#         CalcResultsOfSet('Test')



        
#         self.train_dir = os.path.join(ais.img_dirbase, ais.img_subdir_train)
#         self.test_dir = os.path.join(ais.img_dirbase, ais.img_subdir_test)

#         self.train_dataset = image_dataset_from_directory(self.train_dir,
#                                                     shuffle=True,
#                                                     batch_size=ais.batch_size,
#                                                     image_size=ais.image_size_2)

#         self.test_dataset = image_dataset_from_directory(self.test_dir,
#                                                         shuffle=True,
#                                                         batch_size=ais.batch_size,
#                                                         image_size=ais.image_size_2)
#         self.class_names = self.train_dataset.class_names
#         p('class_names: {}'.format(self.class_names))
#         self.num_classes = len(self.class_names)
#         p('num_classes: {}'.format(self.num_classes))

#         import glob
#         fnsTrain = glob.glob(os.path.join(self.train_dir, '**', ais.img_extension))
#         fnsTest = glob.glob(os.path.join(self.test_dir, '**', ais.img_extension)) 
#         self.num_train_imgs = len(fnsTrain)
#         self.num_test_imgs = len(fnsTest)
#         p('Train images #: {}'.format(self.num_train_imgs))
#         p('Test images #: {}'.format(self.num_test_imgs)) 

#         self.TrainClassesCount = {}
#         self.TestClassesCount = {}
#         for cdn in self.class_names:
#             self.TrainClassesCount[cdn] = len(glob.glob(os.path.join(self.train_dir, cdn, ais.img_extension)))
#             self.TestClassesCount[cdn] = len(glob.glob(os.path.join(self.test_dir, cdn, ais.img_extension)))
#         p('TrainClassesCount: {}'.format(self.TrainClassesCount))
#         p('TestClassesCount: {}'.format(self.TestClassesCount)) 

#         # Use buffered prefetching to load images from disk without having I/O become blocking.
#         # To learn more about this method see the data performance guide.
#         AUTOTUNE = tf.data.experimental.AUTOTUNE
#         self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
#         self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)



        # p('==========')
