from graphs.IGraph import Graph
import tensorflow as tf
import numpy as np
from data.bottleneck.helpers.tf_methods import *
from keras.layers.normalization import BatchNormalization
from keras import *
from keras.layers import Input, Dense, Concatenate
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau

class GraphBAAkeras(Graph):
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
        # learning_rate=0.8
        dense_size = 1000
        activation_function = tf.nn.relu

        bottleneck_tensor_size = self.input_tensor_size
        output_tensor_size = self.output_tensor_size


        # gender_input = tf.placeholder(tf.float32, [None, gender_size], 'GenderInput')
        gender_input = Input(shape=[None, gender_size], name='GenderInput')
        # bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], 'BottleneckInput')
        bottleneck_input = Input(shape=[None, bottleneck_tensor_size], name='BottleneckInput')
        # ground_truth_input = tf.placeholder(tf.float32, [None, output_tensor_size], 'GroundTruthInput')
        # ground_truth_input = Input(shape=[None, output_tensor_size], name='GroundTruthInput')

        # in_training_mode = tf.placeholder(tf.bool, name="in_training")
        # learning_rate = tf.placeholder(tf.float32, name="lr")  # to feed as bottleneck_tensor_size


        bottleneck_input_bn_k = BatchNormalization()(bottleneck_input)

        # ground_truth_input_k = Input(shape=[None, output_tensor_size], name='GroundTruthInput')
        dense_gender_k = Dense(gender_size, activation=activation_function, name='dense_gender')(gender_input)
        merged_input_k = Concatenate()([dense_gender_k, bottleneck_input_bn_k])

        # dense_1_k = Dense(dense_size, activation=None, use_bias=False, name='dense_1')(merged_input_k)
        # dense_1_bn_k = BatchNormalization(dense_1_k, training=in_training_mode)
        # dense_1_af_k = Activation(activation_function)(dense_1_bn_k)
        #
        # dense_2_k = Dense(dense_size, activation=None, use_bias=False, name='dense_2')(dense_1_af_k)
        # dense_2_bn_k = BatchNormalization(dense_2_k, training=in_training_mode)
        # dense_2_af_k = Activation(activation_function)(dense_2_bn_k)

        dense_1_k = Dense(dense_size, activation=activation_function, use_bias=False, name='dense_1')(merged_input_k)
        dense_1_bn_k = BatchNormalization()(dense_1_k)

        dense_2_k = Dense(dense_size, activation=activation_function, use_bias=False, name='dense_2')(dense_1_bn_k)
        dense_2_bn_k = BatchNormalization()(dense_2_k)

        # BN used after relu because: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        # todo https://arxiv.org/abs/1502.03167


        # dense2_k = Dense(dense_size, activation=activation_function, name='dense_2')(dense1_k)

        # final_tensor_k = Dense(1, activation=None, name='output')(dense2_k)
        final_tensor_k = Dense(1, activation=None, use_bias=False, name='output')(dense_2_bn_k)
        # final_tensor_bn_k = BatchNormalization(final_tensor_k, training=in_training_mode)


        # ----

        # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # loss = losses.mean_absolute_error(ground_truth_input, final_tensor_k)
        # loss = losses.mean_absolute_error(unscaleAgeT(ground_truth_input), unscaleAgeT(final_tensor_k))
        # metric = metrics.mean_absolute_error(unscaleAgeT(ground_truth_input), unscaleAgeT(final_tensor_k))
        # metric = metrics.mean_absolute_error(ground_truth_input, final_tensor_k)


        model = Model(inputs=[gender_input, bottleneck_input], outputs=final_tensor_k)
        model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss='mae',
                      metrics=['mae'])


        # todo to use outside?
        # reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
        # #                                    epsilon=0.0001, cooldown=5, min_lr=0.0001)
        #
        # factor = 0.8
        # patience = 10
        # epsilon = 0.0001
        # cooldown = 5
        # min_lr = 0.0001
        #
        # lr = 0.001
        # last_loss = 10000
        # cooldown_cnt = 0
        # stagnation = 0
        # # //firstly wait cooldown, then check if not smaller through patience, then diminish lr
        #
        #
        # res_loss = model.train_on_batch({'GenderInput': x, 'BottleneckInput': x2, 'in_training': True, 'lr':lr},
        #     {'GroundTruthInput': y})  # starts training
        # print("loss")
        # print(res_loss)
        # print("lr")
        # print(lr)
        #
        # cooldown_cnt += 1
        # if cooldown_cnt >= cooldown:
        #     if res_loss >= last_loss - epsilon: # and res_loss <= last_loss + epsilon:
        #         stagnation += 1
        #         if stagnation >= patience:
        #             cooldown_cnt = 0
        #             stagnation = 0
        #             if lr > min_lr:
        #                 lr *= factor
        #                 lr = max(lr, min_lr)
        #     else:
        #         stagnation = 0
        #



        return Graph(model)  # ew unscaleAgeT(final_tensor) i MAE nie scaled

    @staticmethod
    def get_graph_ops(model):

        model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss='mae',
                      metrics=['mae'])

        return Graph(model)
