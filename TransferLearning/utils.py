"""
The original author who has authored the script has given the software as per MIT license,
and me who has done a few modifications to it have given the 'software' under GNU GENERAL PUBLIC LICENSE
Use at your own peril.

    capsulenet.py, which demonstrates transfer learning on capsule neural networks
    Copyright (C) 2020  Subramanya M.S @ 'ms.mssubramanya@gmail.com'

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
#slice off the first ten symbols for the work.
#This is required since digits EMNIST dataset has only 10 samples.
#Causes issues with the number of input and output connections, when we do transfer learning.
#NOTE: I don't have any other dataset for numbers like for ex: Hexadecimal, hence
#      i need to slice off the alphabets dataset, so that both the digits and the alphabets 
#      have the same number of test objects, as well as training objects.
import numpy as np
from numpy import savetxt
from matplotlib import pyplot as plt
import csv
import math
import pandas
from keras.utils import to_categorical

def load_mnist():
    """
    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
    EMNIST: an extension of MNIST to handwritten letters.
    Retrieved from http://arxiv.org/abs/1702.05373
    """
    import pandas as pd
    train = pd.read_csv('../input/emnist-bymerge-train-first-10-shuffled.csv', header=None)
    test = pd.read_csv('../input/emnist-bymerge-test-first-10-sorted.csv', header=None)
    
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
def load_data_digits_over_net():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)
def load_mnist_letters_10_20_shuffled():
    """
    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
    EMNIST: an extension of MNIST to handwritten letters.
    Retrieved from http://arxiv.org/abs/1702.05373
    """
    import pandas as pd
    train = pd.read_csv('../input/emnist-bymerge-train-remaining-shuffled-10-20.csv', header=None)
    test = pd.read_csv('../input/emnist-bymerge-test-remaining-sorted-10-20.csv', header=None)

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
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])

def load_data_bymerge_10_shuffled():
    # the data, shuffled and split between train and test sets
    """
    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
    EMNIST: an extension of MNIST to handwritten letters.
    Retrieved from http://arxiv.org/abs/1702.05373
    """
    import pandas as pd
    train = pd.read_csv('../input/emnist-letters-train-shuffled-10.csv', header=None)
    test = pd.read_csv('../input/emnist-letters-test.csv', header=None)
    
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

def load_and_split_datafiles(src_filename, split_at, dest_filename):
    table_one = pd.read_csv(filename)
    table_numpy_arr = numpy.sort(table_one.values, axis=0, kind='heapsort')
    table_one_first_part = table_numpy_arr[0:split_at,:]
    savetxt(fname=dest_filename, X=table_one_first_part, fmt='%d', delimiter=',')

def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

if __name__=="__main__":
    plot_log('../output/Digits_Alphabets.csv')
    plot_log('../output/AlphabetsLog.csv')
    plot_log('../output/Digitslog.csv')
    plot_log('../output/Arch2_Alpha_On_Digits.csv')
    plot_log('../output/Arch2_Digits_On_Alphabets.csv')