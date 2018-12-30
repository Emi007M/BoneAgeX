import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from optparse import OptionParser

from data.DataService import DataService, DataType
from graphs.GraphService import GraphService, GraphType
from graphs.IGraph import Graph

from data.bottleneck.BottleneckRepository import extract_genders

from data.bottleneck.helpers.tf_methods import unscaleAgeT, unscaleAge
from keras.models import model_from_json
from keras import backend as K

from data.bottleneck.helpers.params_extractor import Flags
FLAGS = Flags()

# from termcolor import *
# import colorama
# colorama.init()
from utils.logger import Logger

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"]="0"

# test of training with assessment, saving model, reassessing with loaded model.

class ModelEval:
    def __init__(self, input, output, expected_output, loss):
        self.input = input
        self.output = output
        self.expected_output = expected_output
        self.loss = loss

    def to_string(self, lines = 5):
        ret = ""
        for output, expected_output in zip(self.output[:lines], self.expected_output[:lines]):
            ret += str(output) + " - " + str(expected_output) + "\n"
        ret += "loss: " + str(self.loss)
        return ret

class Session:
    def __init__(self, sess, graph, data ):
        self.sess = sess
        # K.set_session(sess)
        self.graph = graph
        self.data = data

        self.log = Logger()

        # last vals from training
        self.loss_val = None
        self.W_val = None
        self.b_val = None

    def train_graph(self, epochs, batch_size):

        # decreasing learning rate ---
        factor = 0.8
        patience = 10
        epsilon = 0.0001
        cooldown = 5
        min_lr = 0.0001

        lr = 0.001
        last_loss = 10000
        cooldown_cnt = 0
        stagnation = 0
        # if self.graph.optimizer:
        self.graph.x.optimizer.lr = 0.001
        # ----


        epoch_steps = data_service.get_data_struct().count_steps_in_epoch(type='training', batch_size=batch_size)

        for e in range(epochs):
            self.log.print_warning("starting epoch " + str(e))
            data_service.get_data_struct().reload_epoch_image_lists()
            data_service.get_data_struct().reload_image_lists()



            for i in range(epoch_steps):

                # get random data to train from data_struct
                self.step_data = data_service.get_data_struct().get_random_data(batch_size)


                if len(self.step_data.x) > 0:
                    [self.loss_val] = self.__train_model(self.step_data)

                    # decreasing learning rate ---
                    cooldown_cnt += 1
                    if cooldown_cnt >= cooldown:
                        if self.loss_val >= last_loss - epsilon:  # and res_loss <= last_loss + epsilon:
                            stagnation += 1
                            if stagnation >= patience:
                                cooldown_cnt = 0
                                stagnation = 0
                                if lr > min_lr:
                                    lr *= factor
                                    lr = max(lr, min_lr)
                                    self.graph.x.optimizer.lr = lr
                                    print("new lr: "+str(lr))
                        else:
                            stagnation = 0
                    last_loss = self.loss_val
                    # --------


                    # logs
                    if i % 100 == 0:
                        self.log.print_train(i+1, epoch_steps, self.loss_val)
                        # self.log.print_eval(self.eval_model(self.step_data)) # eval only batch data after training
                        # self.log.print_eval(self.eval_model(self.data)) # eval whole dataset

                    if i % 1000 == 0:
                        self.log.print_info("Saving model xx")
                        self.save_model(self.graph.x, MODEL_DIR + ""+str(i)+"/", CHECKPOINT_NAME)

            self.log.print_info("Saving model")
            self.save_model(self.graph.x, MODEL_DIR + "ff/f"+str(e)+"/", CHECKPOINT_NAME)

            self.log.print_info("Evaluations after epoch "+str(e))

            # self.log.print_eval(self.eval_model(self.data))
            self.evaluate_data_graph(batch_size, 'validation')
            self.evaluate_data_graph(batch_size, 'training')

        print("--final evals of last train step:")
        # self.log.print_train_epoch(epochs, self.loss_val)
        self.log.print_eval(self.eval_model(self.data))

        self.print_plot()

    def evaluate_data_graph(self, batch_size, type='training'):

        self.log.print("starting evaluation of %s dataset " % type)
        data_service.get_data_struct().reload_epoch_image_lists()
        data_service.get_data_struct().reload_image_lists()

        epoch_steps = data_service.get_data_struct().count_steps_in_epoch(type=type, batch_size=batch_size)

        whole_MAE = 0

        for i in range(epoch_steps):

            # get random data  from data_struct
            self.step_data = data_service.get_data_struct().get_random_data(batch_size, type)

            if len(self.step_data.x) > 0:
                step_eval_model = self.eval_model(self.step_data, False)
                whole_MAE += step_eval_model.loss / epoch_steps

            print("\r%3d%% out of %5d (last step MAE: %10.8f)" %  ((i*100/epoch_steps), epoch_steps, step_eval_model.loss), end='')

        self.log.print("\r %10s MAE: %10.8f" % (type, whole_MAE), self.log.Styles.HEADER)



    def save_model(self, model, dir, filename):
        #create dir if not exists
        if not os.path.exists(dir):
            os.makedirs(dir)
        # serialize model to JSON
        model_json = model.to_json()
        with open(dir + filename + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(dir + filename + ".h5")

        self.log.print("model saved to: " + str(dir))

    def print_plot(self):
        px = []
        if (isinstance(self.data.x, str) and self.data.x.shape and self.data.x.shape[1] > self.data.y.shape[1]) or not isinstance(self.data.x, str):
            for x in self.data.x:
                px.append(x[0])
        else:
            px = self.data.x

        eval_result = self.eval_model(self.data)
        loss = []
        # eval_result.output = unscaleAge(eval_result.output)
        for i in range(len(eval_result.expected_output)):
            loss.append(eval_result.expected_output[i] - eval_result.output[i])

        plt.scatter(loss, eval_result.output, s=0.1, c='C1')

        p0 = []
        for x in self.data.x:
            p0.append(0)
        plt.scatter(p0, unscaleAge(self.data.y), s=0.5, c='C0')

        plt.show()

    def __train_model(self, data):
        x_batch, y_batch, gender_batch = data.x, data.y, data.gender
        graph = self.graph

        gender_batch, x_batch, y_batch = self.__keras_reshape(gender_batch, x_batch, y_batch)

        # print(np.array(gender_batch).ndim)
        # print(x_batch.ndim)
        # print(np.array(y_batch).ndim)

        res_loss = graph.x.train_on_batch(x={'GenderInput': gender_batch, 'JpegInput': x_batch},
                                          y={'output': y_batch})[0]  # starts training

        return [res_loss]

    def __keras_reshape(self, gender_batch, x_batch, y_batch):

        genders = extract_genders(gender_batch)
        genders = np.reshape(genders, (genders.shape[0], genders.shape[1]))
        x_batch = np.array(x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], 500, 500, 3))
        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1, y_batch.shape[1]))
        y_batch = np.array(unscaleAge(y_batch))
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))
        return genders, x_batch, y_batch

    def eval_model(self, data, get_output=True) -> ModelEval:
        input, expected_output, gender_batch = data.x, data.y, data.gender
        graph = self.graph

        gender_batch, input, expected_output = self.__keras_reshape(gender_batch, input, expected_output)

        loss = graph.x.test_on_batch(
            {'GenderInput': gender_batch, 'JpegInput': input},
            {'output': expected_output})[0]  # starts training

        output = [[0]]
        if get_output:
            output = graph.x.predict_on_batch(
                {'GenderInput': gender_batch, 'JpegInput': input})

        return ModelEval(input, np.squeeze(output), np.squeeze(expected_output), loss)

    def use_model(self, input, gender_input=None):
        graph = self.graph

        gender_input, input, _ = self.__keras_reshape(gender_input, input, [[0]])

        # output = self.sess.run(graph.y_pred, feed_dict={graph.x: input,
        #                                               graph.gender_x: extract_genders(gender_input),
        #                                               graph.batch_size: self.get_batch_size(input)})

        output = graph.x.predict_on_batch(
            {'GenderInput': gender_input, 'JpegInput': input})

        return np.squeeze(output)

    def get_batch_size(self, x):
        # print(x)
        # print(len(x))
        # print(len(x[0]))
        # return x.size
        return len(x[0])

def main_model(data, graph_struct, create_new=False, train=True, save=True, evaluate=False, use=False):
    tf.reset_default_graph()

    # -- start session
    with tf.Session() as sess:
        # K.set_session(sess)

        if create_new:
            graphModel = graph_struct.get_graph()

        # -- init graph
        sess.run(tf.global_variables_initializer())

        if not create_new:
            model = load_model(MODEL_DIR+MODEL_TO_LOAD, CHECKPOINT_NAME)
            graphModel = graph_struct.get_graph_ops(model)


        if data is None:
            data_service.get_data_struct().init_sess(sess, graph_struct)
            data = data_service.get_data_struct().get_all_data(None)

        s = Session(sess, graphModel, data)

        # -- run ops on graph
        if train:
            s.log.print_info("Training started (batch size: %d, GPUs: %d, max epochs: %d)" %
                             (FLAGS.train_batch_size, FLAGS.gpus, FLAGS.how_many_epochs))
            s.train_graph(FLAGS.how_many_epochs, FLAGS.train_batch_size)
            #data_service.get_data_struct().reload_epoch_image_lists() # do after every epoch

        if save:
            s.log.print_info("Saving model")
            # train_saver = tf.train.Saver()
            # train_saver.save(sess, MODEL_DIR + CHECKPOINT_NAME)
            # s.log.print("model saved to: " + str(MODEL_DIR + CHECKPOINT_NAME))
            s.save_model(s.graph.x, MODEL_DIR, CHECKPOINT_NAME)

        if evaluate:
            # s.log.print_info("Evaluation for images from validation dataset")
            # s.log.print_eval(s.eval_model(s.data))
            s.log.print_info("Evaluation for both datasets")
            s.evaluate_data_graph(FLAGS.validation_batch_size, 'validation')
            s.evaluate_data_graph(FLAGS.validation_batch_size, 'training')

            s.print_plot()

        if use:
            s.log.print_info("Using graph")
            # print(np.transpose(data.x[0]))
            # print(s.use_model(np.transpose(data.x[:10]), np.transpose(data.gender[:10])))
            print("for expected values:")
            print(np.squeeze(unscaleAge(data.y[:10])))
            print("model calculated:")
            print(s.use_model(data.x[:10], data.gender[:10]))


def load_model(dir, filename):
    # load json and create model
    json_file = open(dir + filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print("Loading model " + dir + filename + ".h5")
    loaded_model.load_weights(dir + filename + ".h5")
    print("Model successfully loaded from disk")

    return loaded_model


def handle_command_line_args():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-b", "--tra_batch", dest="tra_batch", help="size of a training batch", type="int", default=FLAGS.train_batch_size)
    parser.add_option("-v", "--val_batch", dest="val_batch", help="size of a validation batch", type="int", default=FLAGS.validation_batch_size)
    parser.add_option("-e", "--epochs", dest="epochs", help="amount of epochs", type="int", default=FLAGS.how_many_epochs)
    parser.add_option("-g", "--gpus", dest="gpus", help="amount of available GPUs", type="int", default=FLAGS.gpus)
    (options, args) = parser.parse_args()
    FLAGS.train_batch_size = options.tra_batch
    FLAGS.validation_batch_size = options.val_batch
    FLAGS.how_many_epochs = options.epochs
    FLAGS.gpus = options.gpus


if __name__ == "__main__":

    MODEL_DIR = "trained_models/i/"
    MODEL_TO_LOAD = "55000/"
    CHECKPOINT_NAME = "model"
    n_samples = 1000 # applicable only if data struct allows to generate a specified amount of data
    batch_size = 4
    # iterations = 1000

    FLAGS.validation_batch_size = batch_size
    FLAGS.train_batch_size = batch_size
    FLAGS.how_many_epochs = 20
    FLAGS.gpus = 1

    handle_command_line_args()

    data_service = DataService(DataType.Jpeg)
    data = data_service.get_data_struct().get_all_data(n_samples) # for data_bottleneck will be now None

    graph_service = GraphService(GraphType.BAAkerasJpeg)

    if data:
        input_tensor_size = data.x.shape[1]
        output_tensor_size = data.y.shape[1]
    else:
        input_tensor_size = 2048#(500, 500, 3) #2048
        output_tensor_size = 1


    graph_struct = graph_service.get_graph_struct(input_tensor_size, output_tensor_size)

    # create new model, train it with given data, than save to file
    # main_model(data, create_new=True, train=True, save=True, evaluate=False, use=False)

    # retrieve model from file, evaluate and use on given data
    # main_model(data, create_new=False, train=False, save=False, evaluate=True, use=True)
    # main_model(data, create_new=False, train=False, save=False, evaluate=True, use=False)
    # main_model(data, graph_struct, create_new=False, train=False, save=False, evaluate=True, use=True)
    main_model(data, graph_struct, create_new=True, train=True, save=False, evaluate=False, use=True)
    # main_model(data, graph_struct, create_new=False, train=True, save=True, evaluate=False, use=True)
    # main_model(data, graph_struct, create_new=False, train=False, save=False, evaluate=True, use=False)

# trainingData - to preapare train and validation sets of images in 2 folders with saved gender in file names
# bottleneck_maker (from train3) - can be used to produce bottlenecks from previously prepared divided images
# test_new = (from test) to asses either single image or set of images with gender taken from csv
# to profile: in terminal python -m cprovilev main2.py