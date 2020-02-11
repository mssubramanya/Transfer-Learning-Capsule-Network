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

Subramanya M.S
Transfer learning using Capsule neural networks. And their results.

Usage:
		python capsulenet.py
Result:
		validation accuracy :
		Choice 1: 
The original author who has authored the script (See Above), and me who has done a few modifications to it
have given the 'software' under GNU GENERAL PUBLIC LICENSE Use at your own peril.
See LICENSE file.
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length
from keras import callbacks
import utils
from keras.utils import to_categorical


K.set_image_data_format('channels_last')
K.set_learning_phase(1)
######################################Global Variables##################################
save_dir = "../output/"
routings=3
batch_size=100
debug=False
lr = 0.001
lr_decay = 0.9
epochs = 20
digit = 15
########################################################################################
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
def CapsNet_Architecture1(input_shape, n_class, routings,weights_to_load):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels], for this project = [28,28,2] (Grayscale)
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

    model_1 = models.Model(inputs=x,outputs=classifier)
    
    if weights_to_load is not None:  # init the model weights with provided one
        model_1.load_weights(weights_to_load)

    for layer in model_1.layers :
        layer.trainable = False

    # Adding our own layers at the end
    digitcaps_2 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='Xfer_learning_capsule_layer')(primaryCapsule)
    out_caps_2 = Length(name='capsnet_2')(digitcaps_2)
    # fully connected and 5-softmax layer
    classifier_2 = layers.Dense(n_class, activation='softmax')(out_caps_2)
    model_output = models.Model(inputs=model_1.input, outputs=classifier_2)
    return model_output
def CapsNet_Architecture2(input_shape, n_class, routings,weights_to_load, Testing5Layers=False):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels], for this project = [28,28,2] (Grayscale)
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
    # fully connected and 5-softmax layer
    classifier = layers.Dense(n_class, activation='softmax')(out_caps)
    model_1 = models.Model(inputs=x,outputs=classifier)

    for layer in model_1.layers :
        layer.trainable = False
    # Adding our own layers at the end
    digitcaps_2 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='Xfer_learning_capsule_layer')(digitcaps)
    out_caps_2 = Length(name='capsnet_2')(digitcaps_2)
    # fully connected and 5-softmax layer
    classifier_2 = layers.Dense(n_class, activation='softmax')(out_caps_2)
    model_output = models.Model(inputs=model_1.input, outputs=classifier_2)

    ### When we need to load the network, we need to be sure that we are loading the 
    ### newly created network, if we are testing on the 5 layer model.
    ### Else we load the regular CapsNet model, of 4 layers
    if Testing5Layers :
    	if weights_to_load is not None:
    		model_output.load_weights(weights_to_load)
    else:
    	if weights_to_load is not None:  # init the model weights with provided one
        	model_1.load_weights(weights_to_load)
    return model_output
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
def train(model, data, category):
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
    log = callbacks.CSVLogger(save_dir + category + '.csv')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/' + category + 'Tensorboard-logs',
                               batch_size=batch_size, histogram_freq=int(debug))
    checkpoint = callbacks.ModelCheckpoint(save_dir + 'Trans_' + category + 'Weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss],
                  loss_weights=[0.392],
                  metrics=['accuracy'])

    # Training without data augmentation:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=[x_test, y_test], 
              callbacks=[log, tb, checkpoint, lr_decay])
    model.save_weights(save_dir + category + '_Final_Trained_Model.h5')
    print('Trained model saved to' + save_dir + category + '_Final_Trained_Model.h5')

    from utils import plot_log
    plot_log(save_dir + category + '.csv', show=True)
    return model
def test(model, data, category):
    x_test, y_test = data
    y_pred = model.predict(x_test, batch_size=500)
    #y_pred = model.predict_classes(x_test)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(x_test[:500], 28, 28)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(save_dir+ "/" + category + "TransReal_and_recon.png")
    print('Reconstructed images are saved to' + save_dir + "/" + category + 'TransReal_and_recon.png')
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(save_dir+ "/" + category + "TransReal_and_recon.png"))
    plt.show()
if __name__ == "__main__":
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Please select amongst the below choices:')
    print('1.  Training the model on alphabets after loading digits weights.')
    print('2.  Training the model on digits after loading alphabets weights')
    print('3.  Testing using bymerge dataset of Digits after fully training the model on alphabets and digits.')
    print('4.  Testing of alphabets images using data from bymerge by loading alphabets and digits trained weights.')
    print('5.  Test on bymerge datset for which training is not done.')
    print('6.  Train Alphabets after loading Digits of first architecture onto the second architecture.')
    print('7.  Train Digits after loading Alphabets of first architecture onto the second architecture.')
    print('8.  Test Digits (Architecture 2)')
    print('9.  Test Alphabets (Architecture 2)')
    print('10. Exit')

    choice = input("Enter your choice: ")
    print('loading data ...')
    if (choice=='1'):
        print('Choice 1 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_bymerge_10_shuffled()
        # define model
        model = CapsNet_Architecture1(input_shape=x_train_1.shape[1:],
                        n_class=len(np.unique(np.argmax(y_train_1, 1))),
                        routings=routings,
                        weights_to_load='../output/Digits_Final_Trained_Model.h5')
        model.summary()
        train(model=model, data=((x_train_1, y_train_1), (x_test_1, y_test_1)), category='Alphabets_Digits')
        print('End of choice 1.')
    if(choice=='2'):
        print('Choice 2 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_digits_over_net()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))

        # define model
        model = CapsNet_Architecture1(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Alphabets_Final_Trained_Model.h5')
        model.summary()
        train(model=model, data=((x_train_1, y_train_1), (x_test_1, y_test_1)), category='Digits_Alphabets')
        print('End of choice 2.')
    if (choice=='3'):
        # load data
        print('Choice 3 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_digits_over_net()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))

        # define model
        model = CapsNet_Architecture1(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Digits_Alphabets_Final_Trained_Model.h5')
        model.summary()
        test(model=model, data=(x_test_1, y_test_1), category='Digits')
        print('End of choice 3.')
    if(choice=='4'):
        print('Choice 4 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_bymerge_10_shuffled()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))

        # define model
        model = CapsNet_Architecture1(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Digits_Alphabets_Final_Trained_Model.h5')
        model.summary()
        test(model=model, data=(x_test_1, y_test_1), category='Alphabets')
        print('End of choice 4.')
    if(choice=='5'):
        print('Choice 5 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_mnist_letters_10_20_shuffled()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))
        
        # define model
        model = CapsNet_Architecture1(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Digits_Alphabets_Final_Trained_Model.h5')
        model.summary()
        test(model=model, data=(x_test_1, y_test_1), category='Untrained')
        print('End of choice 5.')
    if(choice=='6'):
        print('Choice 6 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_bymerge_10_shuffled()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))
        
        # define model
        model = CapsNet_Architecture2(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Digits_Final_Trained_Model.h5',
                        Testing5Layers = False)
        model.summary()
        train(model=model, data=((x_train_1, y_train_1), (x_test_1, y_test_1)), category='Arch2_Alpha_On_Digits')
        print('End of choice 6.')
    if(choice=='7'):
        print('Choice 7 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_digits_over_net()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))
        
        # define model
        model = CapsNet_Architecture2(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Alphabets_Final_Trained_Model.h5',
                        Testing5Layers = False)
        model.summary()
        train(model=model, data=((x_train_1, y_train_1), (x_test_1, y_test_1)), category='Arch2_Digits_On_Alphabets')
        print('End of choice 7.')
    if (choice=='8'):
        # load data
        print('Choice 8 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_digits_over_net()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))

        # define model
        model = CapsNet_Architecture2(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Arch2_Digits_On_Alphabets_Final_Trained_Model.h5',
                        Testing5Layers = True)
        model.summary()
        test(model=model, data=(x_test_1, y_test_1), category='Digits')
        print('End of choice 8.')
    if (choice=='9'):
        # load data
        print('Choice 9 selected:')
        (x_train_1, y_train_1), (x_test_1, y_test_1) = utils.load_data_bymerge_10_shuffled()
        n_class_val = len(np.unique(np.argmax(y_train_1, 1)))

        # define model
        model = CapsNet_Architecture2(input_shape=x_train_1.shape[1:],
                        n_class=n_class_val,
                        routings=routings,
                        weights_to_load='../output/Arch2_Digits_On_Alphabets_Final_Trained_Model.h5',
                        Testing5Layers = True)
        model.summary()
        test(model=model, data=(x_test_1, y_test_1), category='Alphabets')
        print('End of choice 9.')