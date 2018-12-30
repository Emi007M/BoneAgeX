from data.IData import Data



# from data.bottleneck.BottleneckRepository import *
from data.bottleneck.helpers.tf_methods import *

from data.bottleneck.helpers.params_extractor import Flags
FLAGS = Flags()


# gives square images from folders (analogously to bottlenecks but jpeg 500x500) not passed through inception
# bottlenecks are not bottlenecks but images
class DataImage(Data):
    def __init__(self, x=None, y=None):
        super().__init__(x, y)

        self.sess = None
        self.graph = None

        self.image_lists = get_or_create_image_lists(FLAGS.image_dir)
        tf.logging.info("image lists created.")

        class_count = len(self.image_lists.keys())

        self.__validate_image_list(class_count)

        self.reload_epoch_image_lists()

    def reload_epoch_image_lists(self):
        with open(FLAGS.image_dir + '/image_list_division.pkl', 'rb') as input:
            self.epoch_image_lists = pickle.load(input)
            tf.logging.info("loaded")

    def reload_image_lists(self):
        with open(FLAGS.image_dir + '/image_list_division.pkl', 'rb') as input:
            self.image_lists = pickle.load(input)
            tf.logging.info("loaded")



    def count_steps_in_epoch(self, image_lists = None, type='training', batch_size=FLAGS.train_batch_size):
        if image_lists is None:
            image_lists = self.image_lists
        training_data = self.count_data(image_lists, type)

        steps_in_epoch = int(training_data / batch_size) - 1
        return steps_in_epoch

    def count_data(self, image_lists = None, type='training'):
        if image_lists is None:
            image_lists = self.image_lists

        training_data = 0
        for k in list(image_lists.keys()):
            if image_lists[k][type]:
                training_data += len(image_lists[k][type])
        tf.logging.info("all training data: %d" % (training_data))

        return training_data

    def __validate_image_list(self, class_count):
        if class_count == 0:
            tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
            return -1
        if class_count == 1:
            tf.logging.error('Only one valid folder of images found at ' +
                             FLAGS.image_dir +
                             ' - multiple classes are needed for classification.')
            return -1


    # # to be used with first half of graph
    # def cache_bottlenecks_into_subgroups(self):
    #     if FLAGS.create_bottlenecks:
    #         cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
    #                           FLAGS.bottleneck_dir, jpeg_data_tensor,
    #                           decoded_image_tensor, resized_image_tensor,
    #                           bottleneck_tensor, FLAGS.architecture)


    def init_sess(self, sess, graph):
        self.sess = sess
        self.graph = graph

    def get_all_data(self, amount): # not all but validation
        # raise Exception("wont take all data")
        if self.sess is None:
            return None
        return self.__get_data(-1, self.image_lists, 'validation')

    def get_random_data(self, batch_size, type='training'):
        if self.sess is None:
            return None
        return self.__get_data(batch_size, self.epoch_image_lists, type)

    def __get_data(self, amount, image_lists, dataset='training'):
        #dataset 'training'/'validation'
        #amount ?, to get all -> amount :=-1
        #FLAGS.train_batch_size = amount

        sess = self.sess
        jpeg_data_tensor = None
        decoded_image_tensor = None
        resized_image_tensor = None
        bottleneck_tensor = None

        bottleneck_rnd_test = BottlenecksRandomizer(dataset, image_lists)
        (train_bottlenecks,
         train_ground_truth, _, _, train_genders) = get_random_cached_bottlenecks(
            sess, bottleneck_rnd_test, amount,
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture, bottlenecks_not_images=False)
        if train_genders is -1:
            return None
            # self.get_all_data(-1)
            # (train_bottlenecks,
            #  train_ground_truth, _, _, train_genders) = get_random_cached_bottlenecks(
            #     sess, bottleneck_rnd_test, amount,
            #     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            #     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            #     FLAGS.architecture)

        # self.bottlenecks = get_random_cached_bottlenecks
        # (train_bottlenecks,
        #  train_ground_truth, _, image_lists) = self.bottlenecks(
        #     sess, image_lists, FLAGS.train_batch_size, dataset,
        #     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
        #     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
        #     FLAGS.architecture)
        if image_lists is -1:
            return None

        #train_bottlenecks = np.reshape(train_bottlenecks, (amount, 1))
        if len(train_ground_truth) > 0:
            train_ground_truth = np.reshape(train_ground_truth, (len(train_ground_truth), 1))

        return Data(train_bottlenecks, train_ground_truth, train_genders)

        # # x_data = np.arange(amount * 0.1, step=.1)
        # x_data = np.random.normal(0, 100, (amount, 3))
        #
        # y = []
        # for x in x_data:
        #     y.append(x[0])
        #
        # y = np.transpose(y)
        #
        # print(y)
        #
        # # y_data = 0.6 * x_data + 20
        # # y_data = y + 20 * np.sin(y / 10)
        # y_data = 0.07 * 1 / y + 15
        #
        # x_data = np.reshape(x_data, (amount, 3))
        # y_data = np.reshape(y_data, (amount, 1))
        #
        # return Data(x_data, y_data)
