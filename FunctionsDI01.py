
import sys
import os
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import hashlib

def getPathWithSystemPath(p, windowsroot="d:/", linuxroot='/data/home/hl/'):
    ps = ""
    if (platform.system() == 'Windows'):
        ps = os.path.join(windowsroot, p)
    if (platform.system() == 'Linux'):
        ps = os.path.join(linuxroot, p)
    return ps

def getPathWoSystemPath(p, windowsroot="d:/", linuxroot='/data/home/hl/'):
    ps = p
    if (platform.system() == 'Windows'):
        ps = ps.replace(windowsroot.lower(), "")
        ps = ps.replace(windowsroot.upper(), "")  
    if (platform.system() == 'Linux'):
        ps = ps.replace(linuxroot.lower(), "")
        ps = ps.replace(linuxroot.upper(), "")  
    return ps

def MakeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def CalcMD5(fn):
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    md5 = hashlib.md5()
    with open(fn, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def PlotHistory(DirRes, fnpart, logger, history, basename):
    import matplotlib
    matplotlib.use('Agg') # set the backend before importing pyplot
    import matplotlib.pyplot as plt

    name = basename
    val_name = "val_" + basename

    # Plot the accuracy and loss curves
    y = history.history[name]
    val_y = history.history[val_name]

    logger.info("{} Train Min/Max: {:.3} /  {:.3}".format(basename, min(y), max(y)))
    logger.info("{} Test/Valid. Min/Max: {:.3} /  {:.3}".format(basename, min(val_y), max(val_y)))

    epochs = range(len(y))

    plt.grid(True)
    plt.plot(epochs, y, 'b', label='Training ' + basename)
    plt.plot(epochs, val_y, 'r', label='Validation ' + basename)
    plt.title('Training and validation ' + basename)
    plt.legend()
    fn = "plt_{}_{}.png".format(basename, fnpart)
    fn = os.path.join(DirRes, fn)
    plt.savefig(fn)
    plt.close()
    logger.info("{} plot: {}".format(basename, fn))

    return (min(val_y), max(val_y))


# %%
def StringToLogger(logger, string):
    strsplit = None
    if isinstance(string, str):
        strsplit = string.split("\n")
    else:
        strsplit = str(string).split("\n")
    for i in strsplit:
        logger.info(i)    

# plftmp, acc = ClassificationReport(ClassesTrue, ClassesPredict, pImagePaths=None, pTargetNames=None)

# %%

def ClassificationReport(logger, pClassesTrue, pClassesPredict, pImagePaths=None, pTargetNames=None, lOverviewDifferences=False):
    StringToLogger(logger, classification_report(pClassesTrue, pClassesPredict, target_names=None))
    StringToLogger(logger, '    precision = TP / (TP + FP)  Anteil der korrekt klassifizierten dieser Klasse von allen dieser Klasse zugewiesenen Faellen')
    StringToLogger(logger, '    recall (Sensitivity) = TP / (TP + FN)  Anteil dieser Klasse, der  korrekt klassifiziert wurde')
    StringToLogger(logger, '    f1-score = 2*(Recall * Precision) / (Recall + Precision)  aehnlich accuracy, insbes. "uneven class distribution", unbalanced samples\n')

    StringToLogger(logger, pd.crosstab(pClassesTrue, pClassesPredict, rownames=['True'], colnames=['Predicted'], margins=True))
    acc = 100 * accuracy_score(pClassesTrue, pClassesPredict)
    logger.info("acc = {:.2f} % ({})".format(acc, acc))
    # welche sind falsch klassifiziert?
    CorrectClassification = pClassesTrue == pClassesPredict
    StringToLogger(logger, "Correct classified #: {} / {} = {:.1f} %".format(
                                    sum(CorrectClassification), 
                                    len(CorrectClassification),
                                    sum(CorrectClassification) / (0.01*len(CorrectClassification))))
    FalseClassification = pClassesTrue != pClassesPredict
    StringToLogger(logger, "False classified #: {} / {} = {:.1f} %".format(
                                    sum(FalseClassification), len(FalseClassification), 
                                    sum(FalseClassification) / (0.01*len(FalseClassification))))
    plftmp = None
    if pImagePaths is None:
        plftmp = [(p-l, p, l)
                  for is_good, p, l in zip(FalseClassification, pClassesTrue, pClassesPredict)]
    else:
        plftmp = [(p-l, p, l, f) for is_good, p, l,
                  f in zip(FalseClassification, pClassesTrue, pClassesPredict, pImagePaths)]
    if lOverviewDifferences:
        StringToLogger(logger, "Overview Differences")
        StringToLogger(logger, pd.Series([item[0] for item in plftmp]).value_counts())
    return (plftmp, acc)


# %%
def PredictDir(model, dir, image_size, regress, batch_size = 1):
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1./255)

    # Data Generator for Validation data
    class_mode='categorical'
    if regress:
        class_mode='input'
    generator = datagen.flow_from_directory(
            dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False)
    tmpfilenames = generator.filenames
    tmpSamplesN = len(tmpfilenames)
    predictFull = model.predict_generator(generator, tmpSamplesN)
    ClassesPredict = np.argmax(predictFull, axis=1)
    labels = (generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    ClassesTrue = generator.classes
    return(ClassesPredict, ClassesTrue, generator.filenames, labels, predictFull)

# %%
def yes_or_no_is_yes(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    return False

# %%
def Make14(score):
    score14 = np.full(len(score),None)
    score14[score>2.5] = 4
    score14[np.logical_and(score>1.5, score<=2.5)] = 3
    score14[np.logical_and(score>0.5, score<=1.5)] = 2
    score14[score<=0.5] = 1
    score14 = score14.astype(int)
    return (score14)
  

# %%
def MakeCallbacks(modelname, PROJECT_ROOT_DIR, DirRes, monitorname):

    from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

    # Callbacks
    # fpCP: filepathCheckPoint
    # CP: checkpoint
    # fp: filepathL
    fpCPLastEvery = modelname+"_Last.hdf5"
    fpCPLastEvery = os.path.join(DirRes, fpCPLastEvery)
    cpLastEvery = ModelCheckpoint(fpCPLastEvery, monitor=monitorname,
                                    verbose=1, save_best_only=False, mode='auto', period=1)
    fpCPEvery50 = modelname+"-{epoch:04d}.hdf5"
    fpCPEvery50 = os.path.join(DirRes, fpCPEvery50)
    cpEvery50 = ModelCheckpoint(fpCPEvery50, monitor=monitorname,
                                    verbose=1, save_best_only=False, mode='auto', period=50)
    fpCPLast10th = modelname+"_Last_10th.hdf5"
    fpCPLast10th = os.path.join(DirRes, fpCPLast10th)
    cpLast10th = ModelCheckpoint(fpCPLast10th, monitor=monitorname,
                                    verbose=1, save_best_only=False, mode='auto', period=10)
    fbCPBest = modelname+"_Best.hdf5"
    fbCPBest = os.path.join(
        PROJECT_ROOT_DIR, fbCPBest)
    cpBest = ModelCheckpoint(
        fbCPBest, monitor=monitorname, verbose=1, save_best_only=True, mode='auto', period=1)
    fpLoggerLocal = os.path.join(PROJECT_ROOT_DIR, modelname+"_Log.txt")
    csvloggerLocal = CSVLogger(fpLoggerLocal, separator='\t', append=True)
    fpoggerRes = os.path.join(DirRes, modelname+"_Log.txt")
    csvloggerRes = CSVLogger(fpoggerRes, separator='\t', append=True)
    # dirtensorboardlog=modelname
    # dirtensorboardlog=os.path.join("logs", dirtensorboardlog)
    # if not os.path.exists(dirtensorboardlog):
    #     os.makedirs(dirtensorboardlog)
    # tensorboard = TensorBoard(log_dir=dirtensorboardlog)

    return (cpLastEvery, cpEvery50, cpLast10th, cpBest, csvloggerLocal, csvloggerRes)