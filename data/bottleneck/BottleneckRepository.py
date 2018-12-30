from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from datetime import datetime
from time import gmtime, strftime
import pickle

from shutil import copyfile

from data.bottleneck.helpers.params_extractor import Flags
FLAGS = Flags()

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'



def extract_genders(genders):
    return np.tile(np.reshape(genders, (-1, 1)), [1, 32])


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    architecture: The name of the model architecture.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        if not os.path.exists(os.path.join(FLAGS.saved_model_dir)):
            os.makedirs(os.path.join(FLAGS.saved_model_dir))
        copyfile(model_path, os.path.join(FLAGS.saved_model_dir, 'ssaved_model.pb'))
        #model_path = os.path.join(FLAGS.saved_model_dir, 'saved_model.pb')
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor

def read_model_graph_after_epoch(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.saved_model_dir, 'ssaved_model.pb')
        print('Model path extract: ', model_path)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        #     create_model_graph(model_info)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


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


def maybe_download_and_extract(data_url):
    """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.

  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s %d bytes.', filename,
                        statinfo.st_size)
        print('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of which set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    architecture: The name of the model architecture.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        print("ARGHH, tryin to create bottleneck file")
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    architecture: The name of the model architecture.

  Returns:
    Nothing.
  """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            if len(label_lists[category]) == 0:
                print("no images for " + category + 'for label ' + label_name)
                continue
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                if not category_list:
                    continue
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')

#
# def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
#                                   bottleneck_dir, image_dir, jpeg_data_tensor,
#                                   decoded_image_tensor, resized_input_tensor,
#                                   bottleneck_tensor, architecture):
#     """Retrieves bottleneck values for cached images.
#
#   If no distortions are being applied, this function can retrieve the cached
#   bottleneck values directly from disk for images. It picks a random set of
#   images from the specified category.
#
#   Args:
#     sess: Current TensorFlow Session.
#     image_lists: Dictionary of training images for each label.
#     how_many: If positive, a random sample of this size will be chosen.
#     If negative, all bottlenecks will be retrieved.
#     category: Name string of which set to pull from - training, testing, or
#     validation.
#     bottleneck_dir: Folder string holding cached files of bottleneck values.
#     image_dir: Root folder string of the subfolders containing the training
#     images.
#     jpeg_data_tensor: The layer to feed jpeg image data into.
#     decoded_image_tensor: The output of decoding and resizing the image.
#     resized_input_tensor: The input node of the recognition graph.
#     bottleneck_tensor: The bottleneck output layer of the CNN graph.
#     architecture: The name of the model architecture.
#
#   Returns:
#     List of bottleneck arrays, their corresponding ground truths, and the
#     relevant filenames.
#   """
#     class_count = len(image_lists.keys())
#     bottlenecks = []
#     ground_truths = []
#     filenames = []
#
#     # tf.logging.info(image_lists.keys())
#     # tf.logging.info(len(image_lists.keys()))
#     #
#     # tf.logging.info(list(image_lists.keys()))
#     # tf.logging.info(list(image_lists.keys())[1])
#     # image_lists[list(image_lists.keys())[1]]
#     # tf.logging.info(image_lists['101'])
#     # tf.logging.info(image_lists['101']['training'])
#     # tf.logging.info(len(image_lists['101']['training']))
#     # tf.logging.info(len(image_lists['102']['training']))
#
#     if how_many >= 0:
#         # Retrieve a random sample of bottlenecks.
#         unused_i = 0
#         while unused_i < how_many:
#             if len(image_lists.keys()) < 1:
#                 tf.logging.info("too little image list " + len(image_lists.keys()))
#                 return bottlenecks, ground_truths, filenames, -1
#             label_index = random.randrange(len(image_lists.keys()))
#             label_name = list(image_lists.keys())[label_index]
#             image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
#             if not image_lists[label_name][category]:
#                 continue
#             image_name = get_image_path(image_lists, label_name, image_index,
#                                         image_dir, category)
#             bottleneck = get_or_create_bottleneck(
#                 sess, image_lists, label_name, image_index, image_dir, category,
#                 bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
#                 resized_input_tensor, bottleneck_tensor, architecture)
#             bottlenecks.append(bottleneck)
#             ground_truths.append(label_index)
#             filenames.append(image_name)
#             unused_i += 1
#
#             # pop from list
#             mod_index = image_index % len(image_lists[label_name][category])
#             del image_lists[label_name][category][mod_index]
#             if len(image_lists[label_name][category]) is 0:
#                 del image_lists[label_name]
#                 # else:
#                 #   tf.logging.info("image removed from list in this epoch. label: "+ label_name+", "+ str(len(image_lists[label_name][category])) )
#
#     else:
#         # Retrieve all bottlenecks.
#         for label_index, label_name in enumerate(image_lists.keys()):
#             for image_index, image_name in enumerate(
#                     image_lists[label_name][category]):
#                 image_name = get_image_path(image_lists, label_name, image_index,
#                                             image_dir, category)
#                 bottleneck = get_or_create_bottleneck(
#                     sess, image_lists, label_name, image_index, image_dir, category,
#                     bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
#                     resized_input_tensor, bottleneck_tensor, architecture)
#                 bottlenecks.append(bottleneck)
#                 ground_truths.append(label_index)
#                 filenames.append(image_name)
#     return bottlenecks, ground_truths, filenames, image_lists
#

def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
    input_width: Horizontal size of expected input image to model.
    input_height: Vertical size of expected input image to model.
    input_depth: How many channels the expected input image should have.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          bottleneck_tensor_size, quantize_layer, is_training):
    """Adds a new softmax and fully-connected layer for training and eval.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    bottleneck_tensor_size: How many entries in the bottleneck vector.
    quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantized.
    is_training: Boolean, specifying whether the newly add layer is for training
        or eval.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
    # with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [None], name='GroundTruthInput')

    # # Organizing the following ops so they are easier to see in TensorBoard.
    # layer_name = 'final_retrain_ops'
    # with tf.name_scope(layer_name):
    #     with tf.name_scope('weights'):
    #         initial_value = tf.truncated_normal(
    #             [bottleneck_tensor_size, class_count], stddev=0.001)
    #         layer_weights = tf.Variable(initial_value, name='final_weights')
    #         variable_summaries(layer_weights)
    #
    #     with tf.name_scope('biases'):
    #         layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    #         variable_summaries(layer_biases)
    #
    #     with tf.name_scope('Wx_plus_b'):
    #         logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
    #         tf.summary.histogram('pre_activations', logits)

    # Organizing the following ops so they are easier to see in TensorBoard.
    #   layer_name = 'final_retrain_ops'
    # with tf.name_scope(layer_name):
    # with tf.name_scope('dense'):
    #   dense0 = tf.layers.dense(inputs=bottleneck_input, units=16, activation=tf.nn.sigmoid)

    with tf.name_scope('dense_1'):
        w2_initial = np.random.normal(size=(2048, 1000)).astype(np.float32)
        epsilon = 1e-3

        # w2_BN = tf.Variable(w2_initial)
        w2_BN = tf.get_variable("w2_BN", initializer= w2_initial)
        z2_BN = tf.matmul(bottleneck_input, w2_BN)
        batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])
        # scale2 = tf.Variable(tf.ones([1000]))
        # beta2 = tf.Variable(tf.zeros([1000]))
        scale2 = tf.get_variable("scale2", initializer=tf.ones([1000]))
        beta2 = tf.get_variable("beta2", initializer=tf.zeros([1000]))
        BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
        dense = tf.nn.relu(BN2, name="dense_1_activation_f")


    # dense = tf.layers.dense(inputs=bottleneck_input, units=1000, activation=tf.nn.relu, name="dense_1", bias_initializer=tf.constant_initializer(np.ones((1, 1000))))

    dense2 = tf.layers.dense(inputs=dense, units=1000, activation=tf.nn.relu, name="dense_2",
                             bias_initializer=tf.constant_initializer(np.ones((1, 1000))))
    dense3 = tf.layers.dense(
        inputs=dense2,
        units=1,
        activation=None,
        name="output",
        bias_initializer=tf.constant_initializer(1.0)
    )

    with tf.name_scope('dense_layers_summary'):
        # weights = tf.get_default_graph().get_tensor_by_name(
        #     os.path.split(dense.name)[0] + '/kernel:0')
        ####tf.summary.histogram('dense1_weights', w2_BN)
        # biases = tf.get_default_graph().get_tensor_by_name(
        #     os.path.split(dense.name)[0] + '/bias:0')
        # tf.summary.histogram('dense1_biases', biases)

        weights = tf.get_default_graph().get_tensor_by_name(
            os.path.split(dense2.name)[0] + '/kernel:0')
        tf.summary.histogram('dense2_weights', weights)
        biases = tf.get_default_graph().get_tensor_by_name(
            os.path.split(dense2.name)[0] + '/bias:0')
        tf.summary.histogram('dense2_biases', biases)

        weights = tf.get_default_graph().get_tensor_by_name(
            os.path.split(dense3.name)[0] + '/kernel:0')
        tf.summary.histogram('dense3_weights', weights)
        biases = tf.get_default_graph().get_tensor_by_name(
            os.path.split(dense3.name)[0] + '/bias:0')
        tf.summary.histogram('dense3_biases', biases)

    # print("xxxxxx " + dense3.name)
    # print(tf.get_default_graph().get_tensor_by_name(
    #     os.path.split(dense3.name)[0]))

    # with tf.name_scope('weights'):
    #     initial_value = tf.truncated_normal(
    #         [1000, 1], stddev=0.001)
    #     layer_weights = tf.Variable(initial_value, name='final_weights')
    #     variable_summaries(layer_weights)
    #
    # with tf.name_scope('biases'):
    #     layer_biases = tf.Variable(tf.zeros([1]), name='final_biases')
    #     variable_summaries(layer_biases)
    #
    # with tf.name_scope('Wx_plus_b'):
    #     logits = tf.matmul(dense2, layer_weights, name=final_tensor_name) + layer_biases
    #     tf.summary.histogram('pre_activations', logits)


    # # final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    final_tensor = tf.reshape(dense3, [-1])
    print("final tensor: ", final_tensor)
    print("ground truth input:", ground_truth_input)

    # The tf.contrib.quantize functions rewrite the graph in place for
    # quantization. The imported model graph has already been rewritten, so upon
    # calling these rewrites, only the newly added final layer will be
    # transformed.
    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.

    with tf.name_scope('MAE'):
        losses = tf.squared_difference(tf.cast(final_tensor, tf.float32),
                                       tf.cast(ground_truth_input, tf.float32))
        absolute_losses = tf.sqrt(losses, name="MAE")
        MAE = tf.reduce_mean(absolute_losses)
        if is_training:
            tf.summary.scalar('MAE_in_training', MAE)

    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name="ADAM_optimizer")
        train_step = optimizer.minimize(MAE, name="train_step")

        # print("ttrain", MSE, train_step)

    return (train_step, MAE, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
    # with tf.name_scope('accuracy'):
    #   with tf.name_scope('correct_prediction'):
    #     prediction = tf.argmax(result_tensor, 1)
    #     correct_prediction = tf.equal(prediction, ground_truth_tensor)
    #   with tf.name_scope('accuracy'):
    #     evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', evaluation_step)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.cast(result_tensor, tf.int64), ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy_evaluation_step', evaluation_step)
    return evaluation_step, result_tensor


def add_evaluation_step_MAE(result_tensor, ground_truth_tensor, scope_name="evaluation"):
    """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
    with tf.name_scope(scope_name):
        # with tf.name_scope('correct_prediction'):
        #   correct_prediction = tf.equal(tf.cast(result_tensor, tf.int64), ground_truth_tensor, name="correct_prediction")
        with tf.name_scope('accuracy'):
            # --real MAE
            losses = tf.squared_difference(tf.cast(result_tensor, tf.float32),
                                           tf.cast(ground_truth_tensor, tf.float32))
            absolute_losses = tf.sqrt(losses, name="MAE_real_eval")
            MAE = tf.reduce_mean(absolute_losses)

        tf.summary.scalar('MAE', MAE)
    return MAE, result_tensor



def build_eval_session(model_info, class_count):
    """Builds an restored eval session without train operations for exporting.

  Args:
    model_info: Model info dictionary from create_model_info()
    class_count: Number of classes

  Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
  """
    # If quantized, we need to create the correct eval graph for exporting.
    # eval_graph, bottleneck_tensor, _ = create_model_graph(model_info)
    eval_graph, bottleneck_tensor, _ = read_model_graph_after_epoch(model_info)

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS.final_tensor_name, bottleneck_tensor,
            model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
            False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                          ground_truth_input)
        # print("final prediction:", prediction)
        prediction = tf.cast(prediction, tf.int64)
        # print("eval for: ", ground_truth_input, final_tensor, ground_truth_input[0],final_tensor[0])

    return (eval_sess, bottleneck_input, ground_truth_input, evaluation_step,
            prediction)


def save_graph_to_file(graph, graph_file_name, model_info, class_count):
    """Saves an graph to file, creating a valid quantized one if necessary."""
    sess, _, _, _, _ = build_eval_session(model_info, class_count)
    # graph = sess.graph

    # graph_def = sess.graph.as_graph_def()
    # for node in graph_def.node:
    #     print("node: ",node)

    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), ["input/final_retrain_ops/Wx_plus_b/final_result"])
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), [FLAGS.final_tensor_name + ":0"])
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), ["output/BiasAdd"])
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["Reshape"])

    if not os.path.exists(graph_file_name):
        os.makedirs(graph_file_name)
    with gfile.FastGFile(graph_file_name + '/graph.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
    architecture = architecture.lower()
    is_quantized = False
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.5' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25',
  but found '%s' for architecture '%s'""", version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quant':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True

        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
        model_name = 'mobilenet_v1_' + version_string + '_' + size_string
        if is_quantized:
            model_name += '_quant'
        data_url += model_name + '.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        resized_input_tensor_name = 'input:0'
        model_file_name = model_name + '_frozen.pb'

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
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
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def export_model(model_info, class_count, saved_model_dir):
    """Exports model for serving.

  Args:
    model_info: The modelinfo for the current model.
    class_count: The number of classes.
    saved_model_dir: Directory in which to save exported model and variables.
  """
    # The SavedModel should hold the eval graph.
    sess, _, _, _, _ = build_eval_session(model_info, class_count)
    graph = sess.graph
    with graph.as_default():
        input_tensor = model_info['resized_input_tensor_name']
        in_image = sess.graph.get_tensor_by_name(input_tensor)
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        # out_classes = sess.graph.get_tensor_by_name('input/final_retrain_ops/Wx_plus_b/final_result:0')
        # out_classes = sess.graph.get_tensor_by_name('/' + FLAGS.final_tensor_name + ':0')
        # out_classes = sess.graph.get_tensor_by_name('output/BiasAdd:0')
        out_classes = sess.graph.get_tensor_by_name('Reshape:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    prepare_file_system()
    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1

    tf.logging.info("elo")

    # Look at the folder structure, and create lists of all the images.
    image_lists = get_or_create_image_lists()

    tf.logging.info("elo. image lists created.")

    # tf.logging.info(image_lists)

    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    # Set up the pre-trained graph.
    maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = (
        create_model_graph(model_info))


    tf.logging.info("elo. graph prepared")

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, MAE, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS.final_tensor_name, bottleneck_tensor,
            model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
            True)

    tf.logging.info("elo. added new layer")

    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])


        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        if FLAGS.create_bottlenecks:
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, FLAGS.architecture)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)
        evaluation_MAE, _ = add_evaluation_step_MAE(final_tensor, ground_truth_input)

        tf.logging.info("elo. evaluations added")

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.logging.info("elo. session prepared")

        tf.logging.info("elo. starting training")

        training_data = 0
        for k in list(image_lists.keys()):
            if image_lists[k]['training']:
                training_data += len(image_lists[k]['training'])
        tf.logging.info("all training data: %d" % (training_data))

        steps_in_epoch = int(training_data / FLAGS.train_batch_size) - 1

        # Run the training for as many cycles as requested on the command line.
        # epoch = 0
        i = 0
        for j in range(FLAGS.how_many_epochs):
            epoch_image_lists = []
            # epoch_image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
            #                              FLAGS.validation_percentage)
            with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
                epoch_image_lists = pickle.load(input)
                tf.logging.info("loaded")

            tf.logging.info("image list:")
            tf.logging.info(len(list(epoch_image_lists.keys())))

            tf.logging.info("starting epoch %d" % j)

            for s in range(steps_in_epoch):
                if len(epoch_image_lists.keys()) is 0:
                    tf.logging.info("too many steps")
                    continue
                tf.logging.info("epoch %d, batch %d, overal iteration %d" % (j, s, i))
                #
                # train_bottlenecks = all_train_bottlenecks[
                #                   j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]
                # train_ground_truth = all_train_ground_truth[
                #                   j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]


                (train_bottlenecks,
                 train_ground_truth, _, epoch_image_lists) = get_random_cached_bottlenecks(
                    sess, epoch_image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.architecture)
                if epoch_image_lists is -1:
                    break

                # training_data_left = 0
                # for k in list(epoch_image_lists.keys()):
                #     if epoch_image_lists[k]['training']:
                #         training_data_left += len(epoch_image_lists[k]['training'])
                # tf.logging.info("epoch %d, training data left: %d" %(j,training_data_left))

                # if training_data_left < FLAGS.train_batch_size:
                #     epoch_finished = 1



                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                train_writer.add_summary(train_summary, i)

                # # Every so often, print out how well the graph is training.
                # is_last_step = (i + 1 == FLAGS.how_many_epochs)
                # if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                #   train_accuracy, MAE_value = sess.run(
                #       [evaluation_step, MAE],
                #       feed_dict={bottleneck_input: train_bottlenecks,
                #                  ground_truth_input: train_ground_truth})
                #   tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                #                   (datetime.now(), i, train_accuracy * 100))
                #   tf.logging.info('%s: Step %d: Train MAE = %f' %
                #                   (datetime.now(), i, MAE_value))
                #   # TODO(suharshs): Make this use an eval graph, to avoid quantization
                #   # moving averages being updated by the validation set, though in
                #   # practice this makes a negligable difference.
                #   validation_bottlenecks, validation_ground_truth, _, _ = (
                #       get_random_cached_bottlenecks(
                #           sess, epoch_image_lists, FLAGS.validation_batch_size, 'validation',
                #           FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                #           decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                #           FLAGS.architecture))
                #   # Run a validation step and capture training summaries for TensorBoard
                #   # with the `merged` op.
                #   validation_summary, validation_accuracy = sess.run(
                #       [merged, evaluation_step],
                #       feed_dict={bottleneck_input: validation_bottlenecks,
                #                  ground_truth_input: validation_ground_truth})
                #   validation_writer.add_summary(validation_summary, i)
                #   tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                #                   (datetime.now(), i, validation_accuracy * 100,
                #                    len(validation_bottlenecks)))
                #
                #   # validation_summary2, validation_accuracy2 = sess.run(
                #   #       [merged, evaluation_MAE],
                #   #       feed_dict={bottleneck_input: validation_bottlenecks,
                #   #                  ground_truth_input: validation_ground_truth})
                #   # validation_writer.add_summary(validation_summary2, i)
                #   # tf.logging.info('%s: Step %d: Validation accuracy2 = %.1f%% (N=%d)' %
                #   #                   (datetime.now(), i, validation_accuracy2 * 100,
                #   #                    len(validation_bottlenecks)))
                #
                #   validation_summary2, validation_accuracy2 = sess.run(
                #       [evaluation_step, MAE],
                #       feed_dict={bottleneck_input: validation_bottlenecks,
                #                  ground_truth_input: validation_ground_truth})
                #
                #   validation_summary2_sum = tf.Summary(value=[tf.Summary.Value(tag="validation_summary2_mae", simple_value=validation_summary2)])
                #   validation_writer.add_summary(validation_summary2_sum, i)
                #   tf.logging.info('%s: Step %d: Validation MAE = %.1f' %
                #                   (datetime.now(), i,  validation_accuracy2))

                i += 1

                # with tf.name_scope("epoch"):
                #     with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
                #         all_image_lists = pickle.load(input)
                #         tf.logging.info("loaded all")
                #     # tf.logging(len(image_lists.keys() ))
                #     epoch_summary(MAE, all_image_lists, evaluation_step, bottleneck_input, ground_truth_input, i, sess,
                #                  train_writer, validation_writer, j, jpeg_data_tensor,
                #       decoded_image_tensor, resized_image_tensor, bottleneck_tensor)



                # # Store intermediate results
                # intermediate_frequency = FLAGS.intermediate_store_frequency
                #
                # if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                #     and i > 0):
                #   # If we want to do an intermediate save, save a checkpoint of the train
                #   # graph, to restore into the eval graph.
                #   train_saver.save(sess, CHECKPOINT_NAME)
                #   intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                #                             'intermediate_' + str(i) + '.pb')
                #   tf.logging.info('Save intermediate result to : ' +
                #                   intermediate_file_name)
                #   save_graph_to_file(graph, intermediate_file_name, model_info,
                #                      class_count)

            train_saver.save(sess, CHECKPOINT_NAME)

            # We've completed all our training, so run a final test evaluation on
            # some new images we haven't used before.
            # run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
            #                decoded_image_tensor, resized_image_tensor,
            #                bottleneck_tensor)

            # Write out the trained graph and labels with the weights stored as
            # constants.
            save_graph_to_file(graph, FLAGS.output_graph, model_info, class_count)
            # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            #     f.write('\n'.join(image_lists.keys()) + '\n')

            #export_model(model_info, class_count, FLAGS.saved_model_dir)
        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, CHECKPOINT_NAME)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        # run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
        #                decoded_image_tensor, resized_image_tensor,
        #                bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as
        # constants.
        save_graph_to_file(graph, FLAGS.output_graph_fin, model_info, class_count)
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        export_model(model_info, class_count, FLAGS.saved_model_dir)


def get_or_create_image_lists():

    if FLAGS.create_bottlenecks:
        print("ARGHH tryin to create image_list_dividion.pkl")
        image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                         FLAGS.validation_percentage)
        if not os.path.exists(FLAGS.bottleneck_dir):
            os.makedirs(FLAGS.bottleneck_dir)
        with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'wb') as output:
            pickle.dump(image_lists, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
            image_lists = pickle.load(input)
    return image_lists


def epoch_summary(MAE, image_lists, evaluation_step, bottleneck_input, ground_truth_input, j, sess,
                  train_writer, validation_writer, epoch, jpeg_data_tensor,
                  decoded_image_tensor, resized_image_tensor, bottleneck_tensor):
    sum_tra = 0
    acc_tra = 0

    epoch_finished = 0
    # epoch_image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
    #                                FLAGS.validation_percentage)
    with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
        epoch_image_lists = pickle.load(input)
    i = 0
    tf.logging.info("starting epoch total summary " + str(len(epoch_image_lists.keys())))

    training_data = 0
    for k in list(epoch_image_lists.keys()):
        if epoch_image_lists[k]['training']:
            training_data += len(epoch_image_lists[k]['training'])
    steps_in_epoch = int(training_data / FLAGS.train_batch_size) - 1

    tf.logging.info("steps in epoch %d, %d" % (training_data, steps_in_epoch))

    for s in range(steps_in_epoch):
        (train_bottlenecks,
         train_ground_truth, _, epoch_image_lists) = get_random_cached_bottlenecks(
            sess, epoch_image_lists, FLAGS.train_batch_size, 'training',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture)

        # training_data_left = 0
        # for k in list(epoch_image_lists.keys()):
        #     if epoch_image_lists[k]['training']:
        #         training_data_left += len(epoch_image_lists[k]['training'])
        # if training_data_left < FLAGS.train_batch_size:
        #     epoch_finished = 1

        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        all_tra_sum, all_tra_acc = sess.run(
            [evaluation_step, MAE],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        sum_tra += all_tra_sum
        acc_tra += all_tra_acc
        i += 1
    tf.logging.info("sum tra " + str(sum_tra))
    sum_tra /= i
    acc_tra /= i

    # for j in range(steps_in_epoch):
    #     train_bottlenecks = all_train_bottlenecks[
    #                         j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]
    #     train_ground_truth = all_train_ground_truth[
    #                          j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]
    #     all_tra_sum, all_tra_acc = sess.run(
    #         [evaluation_step, MAE],
    #         feed_dict={bottleneck_input: train_bottlenecks,
    #                    ground_truth_input: train_ground_truth})
    #     sum_tra += all_tra_sum / steps_in_epoch
    #     acc_tra += all_tra_acc / steps_in_epoch

    summary_sum_tra = tf.Summary(value=[tf.Summary.Value(tag="sum_tra", simple_value=sum_tra)])
    summary_acc_tra = tf.Summary(value=[tf.Summary.Value(tag="acc_tra", simple_value=acc_tra)])
    train_writer.add_summary(summary_sum_tra, epoch)
    train_writer.add_summary(summary_acc_tra, epoch)

    sum_val = 0
    acc_val = 0
    epoch_finished = 0
    # epoch_image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
    #                                FLAGS.validation_percentage)
    with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
        epoch_image_lists = pickle.load(input)
    i = 0

    validation_data = 0
    for k in list(epoch_image_lists.keys()):
        if epoch_image_lists[k]['validation']:
            validation_data += len(epoch_image_lists[k]['validation'])
    steps_in_epoch_v = int(validation_data / FLAGS.train_batch_size) - 1

    for s in range(steps_in_epoch_v):
        (vali_bottlenecks,
         vali_ground_truth, _, epoch_image_lists) = get_random_cached_bottlenecks(
            sess, epoch_image_lists, FLAGS.train_batch_size, 'validation',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture)


if __name__ == '__main__':
    name = '12'
    image_dir_folder = 'three_classes'
    epochs = 5
    create_bottlenecks = 0

    params = (name, image_dir_folder, epochs, create_bottlenecks)

    # FLAGS, unparsed = extract_parameters(argparse.ArgumentParser(), params)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]])
