#
# from trainer.params_extractor import extract_parameters
# from trainer.tf_methods import *
#
import os.path

import csv
import collections
from trainer.tester.csv_methods import *
from io import BytesIO
from tensorflow.python.platform import gfile
from keras.preprocessing.image import *
import numpy as np
import keras
from keras.layers import Flatten,Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from data.bottleneck.BottleneckRepository import *
from data.bottleneck.helpers.tf_methods import *


import winsound
from data.bottleneck.helpers.params_extractor import Flags
FLAGS = Flags()
rng = np.random

# CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'
# CHECKPOINT_NAME = 'trainer/trained_model3/'
# CHECKPOINT_NAME = 'trainer/trained_model14-15F_lite/'
d = 'C:/Users/Emilia/Pycharm Projects/BoneAge/'
CHECKPOINT_NAME = d + 'trainer/trained_model_FM16/'

# FLAGS = None

# Parameters
learning_rate = 0.001
display_step = 50
display_all_step = 500
checkpoint_every_n_batches_step = 4000

batch_size = 64
bottleneck_tensor_size = 2048

cost_y = []

is_training = 1

def prepare_batches(_len, max_size):
    r = list(range(_len))
    l = r[0::max_size]
    l.append(_len)
    return l

def extract_genders(genders):
    return np.tile(np.reshape(genders, (-1, 1)), [1, 32])

max_batch_size = 64

no_inception_in_model = True


def add_jpeg_decoding(input_width = 299, input_height = 299, input_depth = 3, input_mean = 128, input_std = 128):
    """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std, name="Mul_1")
    return jpeg_data, mul_image


def scaleAge(value):
    "from real age <0;230> to <-1;1>"
    "(value * 2) / 230 - 1 "
    return float(value) / 115.0 - 1.0


def unscaleAgeL(valueList):
    if type(valueList) is not list:
        values = valueList
    else:
        values = np.squeeze(valueList)
    return [(float(value) + 1.0) * 115.0 for value in values]


def unscaleAgeT(value):
    return tf.scalar_mul(115.0, tf.add(value, 1.0))


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """

    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def gender_to_string(_i):
    if _i is 1:
        return "M"
    return "F"


def img_dir_get_name(_dir):
    return os.path.splitext(os.path.basename(_dir))[0]




if __name__ == '__main__':

    # to set in flags: image_dir_folder = 'FM_labeled_train_validate'
    # FLAGS = extract_parameters()

    FLAGS.create_bottlenecks = 1

    tf.logging.set_verbosity(tf.logging.INFO)

    winsound.Beep(440, 500)
    winsound.Beep(540, 100)
    winsound.Beep(440, 100)
    winsound.Beep(340, 1000)

    # Gather information about the model architecture we'll be using.
    # model_info = create_model_info(FLAGS.architecture)
    # if not model_info:
    #     tf.logging.error('Did not recognize architecture flag')
    #     sys.exit("finishing.")

    # Look at the folder structure, and create lists of all the images.
    image_lists = get_or_create_image_lists()

    tf.logging.info("elo. image lists created.")

    tf.logging.info("elo")
    winsound.Beep(440, 100)
    winsound.Beep(540, 100)
    winsound.Beep(640, 200)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        base_model = InceptionV3(include_top=False, input_shape=(500, 500, 3), weights='imagenet', pooling='max')

        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(500, 500)


        if FLAGS.create_bottlenecks:
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, "",
                              '', FLAGS.architecture, inception_model=base_model)



    winsound.Beep(600, 100)
    winsound.Beep(400, 300)
    winsound.Beep(300, 500)