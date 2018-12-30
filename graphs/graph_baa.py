from graphs.IGraph import Graph
import tensorflow as tf
import numpy as np
from data.bottleneck.helpers.tf_methods import *

class GraphBAA(Graph):
    def __init__(self, x=None, y=None, W=None, b=None, y_pred=None, loss=None, opt_operation=None, batch_size=None):
        super().__init__(x, y, W, b, y_pred, loss, opt_operation, batch_size)


    # def get_inception(self):
    #     # Set up the pre-trained graph.
    #     maybe_download_and_extract(model_info['data_url'])
    #     graph, bottleneck_tensor, resized_image_tensor = (
    #         create_model_graph(model_info))
    #
    #     tf.logging.info("elo. graph prepared")
    #
    #     # Add the new layer that we'll be training.
    #     with graph.as_default():
    #         (train_step, MAE, bottleneck_input,
    #          ground_truth_input, final_tensor) = add_final_retrain_ops(
    #             class_count, FLAGS.final_tensor_name, bottleneck_tensor,
    #             model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
    #             True)

        # tf.logging.info("elo. added new layer")

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

    def get_graph(self, graph=None):
        gender_size = 32
        #gender_size = 1
        learning_rate=0.8
        dense_size = 1000
        activation_function = tf.nn.relu

        bottleneck_tensor_size = self.input_tensor_size
        output_tensor_size = self.output_tensor_size

        # with graph.as_default():  - bylo na cale do konca
        batch_size = tf.placeholder(tf.float32, name="batch_size") # to feed as bottleneck_tensor_size

        gender_input = tf.placeholder(tf.float32, [None, gender_size], 'GenderInput')
        # --> gender_input = tf.placeholder_with_default(1.0, [None, gender_size], 'GenderInput')

        # bottleneck_input = tf.placeholder_with_default(bottleneck_tensor,
        #                                                [None, bottleneck_tensor_size],
        #                                                'BottleneckInput')
        bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], 'BottleneckInput')
        # bottleneck_input = tf.placeholder(tf.float32,  'BottleneckInput')

        ground_truth_input = tf.placeholder(tf.float32, [None, output_tensor_size], 'GroundTruthInput')
        # ground_truth_input = tf.placeholder(tf.float32,  'GroundTruthInput')

        # g = tf.transpose(gender_input)
        # print(tf.shape(g)[0:1])
        # #gender_i = tf.tile(g, [32, 1])

        # gender_i = tf.tile(g, tf.reshape(32, [1]))
        # gender_i = tf.tile(g, tf.shape(g)[0:1])
        # print(tf.shape(gender_i))
        # g = tf.transpose(gender_i)
        # print(tf.shape(gender_input))

        dense_gender = tf.layers.dense(inputs=gender_input, units=gender_size,
                                       activation=activation_function, name="dense_gender",
                                       bias_initializer=tf.constant_initializer(
                                           np.full_like((1, gender_size), float(0.1), dtype=np.float32)))  # changed from ones


        # # scale age
        # ground_truth_input = scaleAge(ground_truth_input)

        # with tf.name_scope('dense_1'):
        #     w2_initial = np.random.normal(size=(bottleneck_tensor_size, 1000)).astype(np.float32)
        #     epsilon = 1e-3
        #
        #     w2_BN = tf.get_variable("w2_BN", initializer=w2_initial)
        #
        #     #todo
        #     # input będzie tf.concat([dense_gender, bottleneck_input], 0)
        #     z2_BN = tf.matmul(bottleneck_input, w2_BN)
        #     batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])
        #     scale2 = tf.get_variable("scale2", initializer=tf.ones([1000]))
        #     beta2 = tf.get_variable("beta2", initializer=tf.zeros([1000]))
        #     BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
        #     dense1 = tf.nn.relu(BN2, name="dense_1_activation_f")

        merged_input = tf.concat([dense_gender, bottleneck_input], 1)

        # z merged input na bottleneck input
        dense1 = tf.layers.dense(inputs=merged_input, units=dense_size, activation=activation_function, name="dense_1",
                                 bias_initializer=tf.constant_initializer(
                                     np.full_like((0.8, dense_size), 0.1, dtype=np.float32)))

        # tf.nn.batch_norm_with_global_normalization
        ###########
        #
        # with tf.name_scope('dense_1'):
        #     in_training_mode = tf.placeholder_with_default(False, shape=None, name='InTrainingMode')
        #
        #     # byl input > bottleneck_input <, zmienione na concat
        #     merged_input = tf.concat([dense_gender, bottleneck_input], 1)
        #     batch_normed = tf.layers.batch_normalization(
        #         inputs=merged_input,
        #         axis=-1,
        #         momentum=0.999,
        #         epsilon=1e-3,
        #         center=True,
        #         scale=True,
        #         training=in_training_mode
        #     )
        #
        #
        #     # hidden = tf.layers.dense(inputs=bottleneck_input, units=1000, activation=tf.nn.relu,
        #     #                 bias_initializer=tf.constant_initializer(np.ones((1, 1000))))
        #     # # hidden = tf.keras.layers.Dense(n_units,
        #     # #                                activation=None)(X)  # no activation function, yet
        #     # batch_normed = tf.keras.layers.BatchNormalization()(hidden, training=in_training_mode)
        #     # # dense1 = tf.keras.activations \
        #     #     .relu(batch_normed)  # ReLu is typically done after batch normalization
        #     dense1 = tf.nn.relu(batch_normed, name="dense_1_activation_f")
        #
        #     tf.summary.histogram('dense_1_batch_normed', batch_normed)
        #     tf.summary.histogram('after_dense_1_relu', dense1)
        #
        # #############

        dense2 = tf.layers.dense(inputs=dense1, units=dense_size, activation=activation_function, name="dense_2",
                                 bias_initializer=tf.constant_initializer(
                                     np.full_like((1, dense_size), 0.1, dtype=np.float32)))  # changed from ones np.zeros((1, 1000))

        final_tensor = tf.layers.dense(
            inputs=dense2,
            units=1,
            activation=None,
            name="output",
            bias_initializer=tf.constant_initializer(float(0.2))  # changed from one,
        )


        # # bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], 'BottleneckInput')
        # # x = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # nn = tf.layers.dense(bottleneck_input, 1)
        # nn2 = tf.layers.dense(nn, 5)
        # encoded = tf.layers.dense(nn2, 2)
        # nn3 = tf.layers.dense(encoded, 5)
        # final_tensor = tf.layers.dense(nn3, 1, name="output")
        # # final_tensor = tf.reshape(nn, [-1], name="output")

        # final_tensor = tf.reshape(dense3, [-1], name="output")

        #
        # tf.summary.histogram('after_dense_g', dense_gender)
        # dg_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense_gender.name)[0] + '/kernel:0')
        # tf.summary.histogram('dense_g_weights', dg_weights)
        # dg_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense_gender.name)[0] + '/bias:0')
        # tf.summary.histogram('dense_g_biases', dg_biases)
        #
        # tf.summary.histogram('after_dense_1', dense1)
        # d1_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense1.name)[0] + '/kernel:0')
        # tf.summary.histogram('dense_1_weights', d1_weights)
        # d1_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense1.name)[0] + '/bias:0')
        # tf.summary.histogram('dense_1_biases', d1_biases)
        #
        # tf.summary.histogram('after_dense_2', dense2)
        # d2_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/kernel:0')
        # tf.summary.histogram('dense_2_weights', d2_weights)
        # d2_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/bias:0')
        # tf.summary.histogram('dense_2_biases', d2_biases)
        # # d2_activations = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/activation:0')
        # # tf.summary.histogram('dense_2_activations', d2_activations
        #
        # tf.summary.histogram('after_dense_3', dense3)
        # d3_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/kernel:0')
        # tf.summary.histogram('dense_3_weights', d3_weights)
        # d3_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/bias:0')
        # tf.summary.histogram('dense_3_bias', d3_biases)
        # # d3_activations = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/activation:0')
        # # tf.summary.histogram('dense_3_activations', d3_activations)



        with tf.name_scope('MAE'):
            MAE_scaled = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(final_tensor, tf.float32),
                                                            tf.cast(ground_truth_input, tf.float32))), name="MAE_scaled")



            #MAE_scaled = tf.reduce_sum(tf.divide((ground_truth_input - final_tensor) ** 2, batch_size), name="MAE")

            # unscaled mae to show
            MAE = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(unscaleAgeT(final_tensor), tf.float32),
                                                    tf.cast(unscaleAgeT(ground_truth_input), tf.float32))),
                                  name="MAE")
            # tf.summary.scalar('MAE', MAE)

        with tf.name_scope('train'):
            # lr_decay_step = tf.Variable(0.0, trainable=False)
            # lr_decay = tf.train.exponential_decay(learning_rate, lr_decay_step, 150, 0.99)
            #
            # optimizer = tf.train.AdamOptimizer(lr_decay, name="ADAM_optimizer")
            optimizer = tf.train.AdamOptimizer(name="ADAM_optimizer")
            train_step = optimizer.minimize(MAE_scaled, name="train_step")

            #tf.summary.scalar('lr_decay', lr_decay)
        # return bottleneck_input, ground_truth_input, unscaleAgeT(
        #     final_tensor), MAE, train_step, gender_input, lr_decay_step

        final_tensor_unscaled = unscaleAgeT(final_tensor)
        ground_truth_input_unscaled = unscaleAgeT(ground_truth_input)

        # return Graph(bottleneck_input, ground_truth_input, None, None, final_tensor_unscaled, MAE, train_step,
        #              batch_size, gender_input)  # ew unscaleAgeT(final_tensor) i MAE nie scaled

        return Graph(bottleneck_input, ground_truth_input, ground_truth_input_unscaled, MAE_scaled, final_tensor, MAE, train_step,
                 batch_size, gender_input)  # ew unscaleAgeT(final_tensor) i MAE nie scaled

    @staticmethod
    def get_graph_ops():
        return Graph("BottleneckInput:0", "GroundTruthInput:0",
                     None, None,
                     "output/BiasAdd:0",
                     "MAE/MAE:0",
                     "train/train_step", "batch_size:0", "GenderInput:0")
