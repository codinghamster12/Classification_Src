# %%

import tensorflow as tf
from matplotlib import pyplot as plt
import math
import os

import aisettings
from diutil import DiUtil

import subprocess

print(__file__)

ais = aisettings.AiSettings(__file__)

    
# %%
# logger print Hilfsfunktion
p = ais.p

# %%

# train_dataset, test_dataset = ais.get_dataset()
train_generator, test_generator = ais.get_datagenerator()

import numpy as np
plt.figure(figsize=(10, 10))
images, labels = train_generator.next()
for i, (img, lab) in enumerate(zip(images, labels)):
    if i >= 9: break
    ax = plt.subplot(3, 3, i + 1)
    # plt.imshow(img.numpy().astype('uint8'))
    plt.imshow(img/255.0)
    result_pred = np.argmax(lab, axis=0)
    plt.title(ais.data.class_names[result_pred])
    plt.axis('off')
fn = os.path.join(ais.dir_res, 'ImgSamples.png')
plt.savefig(fn)
plt.close()

train_steps_per_epoch = math.ceil(ais.data.num_train_imgs  / ais.batch_size)
val_steps_per_epoch   = math.ceil(ais.data.num_test_imgs  / ais.batch_size)


# %%
data_augmentation = ais.get_data_augmentation()

import numpy as np
plt.figure(figsize=(10, 10))
images, labels = train_generator.next()
for i, (img, lab) in enumerate(zip(images, labels)):
    if i >= 9: break
    ax = plt.subplot(3, 3, i + 1)
    # plt.imshow(img.numpy().astype('uint8'))
    plt.imshow(img/255.0)
    tmp_class_idx = np.argmax(lab, axis=0)
    plt.title(ais.data.class_names[tmp_class_idx])
    plt.axis('off')
fn = os.path.join(ais.dir_res, 'ImgSamples.png')
plt.savefig(fn)
plt.close()

plt.figure(figsize=(10, 10))
img = images[0]
ax = plt.subplot(3, 3, 1)
plt.imshow(img/255.0)
tmp_class_idx = np.argmax(labels[0], axis=0)
plt.title('Original ({})'.format(ais.data.class_names[tmp_class_idx]))
plt.axis('off')
for i in range(min(8, ais.batch_size)):
    ax = plt.subplot(3, 3, i + 2)
    augmented_image = data_augmentation(tf.expand_dims(img, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.title('Augm. Data {}'.format(i))
    plt.axis('off')
fn = os.path.join(ais.dir_res, 'DataAugSample.png')
plt.savefig(fn)
plt.close()


# %%
ais.load_model()
model = ais.model.model
p(model.summary())


# %%

callbacks = ais.get_callbacks()

optimizer = ais.get_optimizer()

# Compile:
if ais.regression:
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
else:
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
# Try catch Idee: Dass es danach noch weiter gehen kann, z.B. Diagramm,....
history = None
try:
    history = model.fit(
        train_generator,  epochs=ais.num_epochs, steps_per_epoch=train_steps_per_epoch,
        validation_data=test_generator, validation_steps=val_steps_per_epoch,
        verbose=1, callbacks=callbacks)
except:
    print('Exception...')
    proceeding = input('Proceed? (HL) [n = No, other = Yes]: ')
    if proceeding == 'n' or proceeding == 'N':
        exit()

# p('\n')
# p('\nSaving: model.save(ais.model_best_loss +++ .h5)')
# model.save(ais.model_best_loss + '.h5')
# p('\nSaving: model.save(ais.model_best_loss)')
# model.save(ais.model_best_loss)


# %%

if history is not None:
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.grid(True)
    # plt.ylim([0,1.0])
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    y_train = history.history['loss']
    y_test = history.history['val_loss']
    # auf 2 begrenzen fuer Diagramm
    y_train = list(map(lambda x: x if x < 2 else 2, y_train))
    y_test = list(map(lambda x: x if x < 2 else 2, y_test))
    plt.plot(y_train, label='Training Loss')
    plt.plot(y_test, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    # fn = 'plt_{}_{}.png'.format(basename, fnpart)
    # fn = os.path.join(DirRes, fn)
    fn = os.path.join(ais.dir_res, 'ResAccLoss.png')
    plt.savefig(fn)
    # plt.show()
    plt.close()

# subprocess.Popen('\"C:\\Program Files\\IrfanView\\i_view64.exe\" {}'.format(fn))
p('\n')

# "Bestes" Modell einlesen
ais.model.model = tf.keras.models.load_model(ais.model_best_loss)
model = ais.model.model 
loss, acc = model.evaluate(test_generator,
                        batch_size=ais.batch_size,
                        verbose=1)
p('validation_dataset acc wird berechnet vom Modell mit minimalem loss!')
p('validation_dataset acc: {}\n'.format(acc))


# In ressummary schreiben - abgeschrieben von aicallbackspecial - besser irgendwie zusammenfassen, egal erst mal...
import pandas as pd
fnres = os.path.join(ais.dir_res, '../ressummary.txt')
if os.path.exists(fnres):
    df = pd.read_csv(fnres, sep='\t')
    dir_res_full_name = 'dir_res_full'
    if not ais.dir_res in list(df[dir_res_full_name]):
        pass
    else:
        df['final_best_model_loss'].loc[df[dir_res_full_name] == ais.dir_res] = loss
        df['final_best_model_acc'].loc[df[dir_res_full_name] == ais.dir_res] = acc
        df.to_csv(fnres, sep='\t', index=False)

_, acc = model.evaluate(train_generator,
                        batch_size=ais.batch_size,
                        verbose=1)
p('train_dataset acc: {}\n'.format(acc))
p('#####################################')

ais.report.create_report(ais)
