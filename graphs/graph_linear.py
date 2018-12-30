from graphs.IGraph import Graph
import tensorflow as tf
import numpy as np


class GraphLinear(Graph):
    def __init__(self, x=None, y=None, W=None, b=None, y_pred=None, loss=None, opt_operation=None, batch_size=None):
        super().__init__(x, y, W, b, y_pred, loss, opt_operation, batch_size)

    def get_graph():
        batch_size = tf.placeholder(tf.float32, name="ba_batch_size")

        x = tf.placeholder(tf.float32, name="ba_input")
        y = tf.placeholder(tf.float32, name="ba_expected_output")

        with tf.variable_scope("linear-regression"):
            W = tf.get_variable("ba_weights", (1, 1), initializer=tf.random_normal_initializer())
            b = tf.get_variable("ba_bias", (1,), initializer=tf.constant_initializer(5.0))

            y_pred = tf.add(tf.matmul(x, W), b, name="ba_output")
            loss = tf.reduce_sum(tf.divide((y - y_pred) ** 2, batch_size), name="ba_loss")

        opt = tf.train.AdamOptimizer()
        opt_operation = opt.minimize(loss, name="ba_opt_operation")

        return Graph(x, y, W, b, y_pred, loss, opt_operation, batch_size)

    @staticmethod
    def get_graph_ops():
        return Graph("ba_input:0", "ba_expected_output:0",
                     "linear-regression/ba_weights:0", "linear-regression/ba_bias:0",
                     "linear-regression/ba_output:0", "linear-regression/ba_loss:0",
                     "ba_opt_operation", "ba_batch_size:0")