"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks

K.set_image_data_format('channels_last')
######################################Global Variables##################################
save_dir = "../output/"
routings=3
batch_size=100
debug=False
lr = 0.001
lr_decay = 0.9
epochs = 50
digit=5
lam_recon=0.392
########################################################################################
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels], for this project = [28,28,1] (Grayscale)
    :param n_class: number of classes, as of now = 10
    :param routings: number of routing iterations, by default = 3
    :return: A keras model for training and testing.
    """
    #print(input_shape)
    x = layers.Input(shape=input_shape)
    #print('x.shape = ' + str(x.shape))
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=2, strides=2, padding='same', activation='relu', name='conv1')(x)
    #print(conv1.shape)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primaryCapsule = PrimaryCap(conv1, dim_capsule=8, n_channels=2, kernel_size=22, strides=1, padding='same')
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primaryCapsule)
    out_caps = Length(name='capsnet')(digitcaps)
    classifier = layers.Dense(n_class, activation='softmax', name='capsnet_output')(out_caps)

    model = models.Model(inputs=x,outputs=classifier)
    return model
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
def train(model, data):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    # callbacks
    log = callbacks.CSVLogger(save_dir + '/AlphabetsLog.csv')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/AlphabetsTensorboard-logs',
                               batch_size=batch_size, histogram_freq=int(debug))
    checkpoint = callbacks.ModelCheckpoint(save_dir + 'Alphabets-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss],
                  loss_weights=[lam_recon],
                  metrics=['accuracy'])
    # Training without data augmentation:
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[log, tb, checkpoint, lr_decay])
    model.save_weights(save_dir + 'Alphabets_Final_Trained_Model.h5')
    print('Trained model saved to' + save_dir + '/Alphabets_Final_Trained_Model.h5')

    from utils import plot_log
    plot_log(save_dir + '/AlphabetsLog.csv', show=True)
    return model
def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + "/AlphabetsReal_and_recon.png")
    print('Reconstructed images are saved to %s/AlphabetsReal_and_recon.png' % save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(save_dir+ "/AlphabetsReal_and_recon.png"))
    plt.show()
def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + '/AlphabetsManipulate-%d.png' % digit)
    print('manipulated result saved to %s/AlphabetsManipulate-%d.png' % (save_dir, digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)
def load_mnist():
    """
    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
    EMNIST: an extension of MNIST to handwritten letters.
    Retrieved from http://arxiv.org/abs/1702.05373
    """
    import pandas as pd
    train = pd.read_csv('../input/emnist-letters-train-shuffled-10.csv', header=None)
    test  = pd.read_csv('../input/emnist-letters-test.csv', header=None)

    train_data = train.iloc[:, 1:]
    train_labels = train.iloc[:, 0]
    test_data = test.iloc[:, 1:]
    test_labels = test.iloc[:, 0]
    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)
    
    train_data = train_data.values
    train_labels = train_labels.values
    test_data = test_data.values
    test_labels = test_labels.values

    train_data = np.apply_along_axis(rotate, 1, train_data)/255
    test_data = np.apply_along_axis(rotate, 1, test_data)/255

    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    train_labels = train_labels.reshape(-1, 10)
    test_labels = test_labels.reshape(-1, 10)

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    return (train_data, train_labels), (test_data, test_labels)
if __name__ == "__main__":
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model = CapsNet(input_shape=x_train.shape[1:],
                    n_class=len(np.unique(np.argmax(y_train, 1))),
                    routings=routings)
    model.summary()

    # train or test
    #model.load_weights(args.weights)
    train(model=model, data=((x_train, y_train), (x_test, y_test)))
    #manipulate_latent(manipulate_model, (x_test, y_test), args)
    #test(model=eval_model, data=(x_test, y_test), args=args)
