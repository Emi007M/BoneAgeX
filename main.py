#!/mnt/storage/longterm/emiliam/TF_virt_env/bin/python

from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from data.DataService import DataService, DataType
from data.bottleneck.BottleneckRepository import extract_genders
from data.bottleneck.helpers.tf_methods import unscaleAge
from graphs.GraphService import GraphService, GraphType
from utils.params_extractor import Flags

import pickle

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
    def __init__(self, sess, graph, data, writers):
        self.sess = sess
        # K.set_session(sess)
        self.graph = graph
        self.data = data

        self.log = Logger()

        # last vals from training
        self.loss_val = None
        self.W_val = None
        self.b_val = None

        self.writers = writers

    def train_graph(self, epochs, batch_size, lr_drop):

        # dodać is_inception_trainable / bottlenecks_instead_of_images

        if lr_drop is 0:
            lr_snapshot_drop = False
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
        else:
            lr_snapshot_drop = True
            # snapshot cyclic cosine annealing
            lr0 = 0.1
            //S = lr_drop #amount of snapshot cycles
            lr = lr0

        # ----


        epoch_steps = data_service.get_data_struct().count_steps_in_epoch(type='training', batch_size=batch_size)

        if lr_snapshot_drop:
            S = epoch_steps*epochs/lr_drop

        for e in range(epochs):
            self.log.print_warning("starting epoch " + str(e))
            data_service.get_data_struct().reload_epoch_image_lists()
            data_service.get_data_struct().reload_image_lists()

            # # handling learning rate
            # if lr_snapshot_drop:
            #     lr = self.snapshot_lr(S, e*epoch_steps+i, lr, lr0)
            # else:
            #     last_loss = self.decrease_learning_rate(cooldown, cooldown_cnt, epsilon, factor, last_loss, lr, min_lr,
            #                             patience, stagnation)

            summary = tf.Summary(value=[tf.Summary.Value(tag="learning_rate", simple_value=lr, ), ])
            self.writers['learning_rate'].add_summary(summary, e)


            for i in range(epoch_steps):

                # handling learning rate
                if lr_snapshot_drop:
                    lr = self.snapshot_lr(S, e * epoch_steps + i, lr0)
                else:
                    last_loss = self.decrease_learning_rate(cooldown, cooldown_cnt, epsilon, factor, last_loss, lr,
                                                            min_lr,
                                                            patience, stagnation)

                # get random data to train from data_struct
                self.step_data = data_service.get_data_struct().get_random_data(batch_size)


                if len(self.step_data.x) > 0:
                    [self.loss_val] = self.__train_model(self.step_data)

                    # logs
                    if i % 100 == 0:
                        self.log.print_train(i+1, epoch_steps, self.loss_val)
                        # self.log.print_eval(self.eval_model(self.step_data)) # eval only batch data after training
                        # self.log.print_eval(self.eval_model(self.data)) # eval whole dataset

                        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=self.loss_val, ), ])
                        self.writers['train_batch'].add_summary(summary, e * epoch_steps + i)


                    # if i % 10000 == 0 and i is not 0:
                    #     self.log.print_info("Saving model ")
                    #     self.save_model(self.graph.x, MODEL_DIR + ""+str(i)+"/", CHECKPOINT_NAME)

            if not lr_snapshot_drop or ((e * epoch_steps + i + 1) % S is 0):
                self.log.print_info("Saving model")
                self.save_model(self.graph.x, MODEL_DIR + ""+str(e)+"/", CHECKPOINT_NAME)

            self.log.print_info("Evaluations after epoch "+str(e))

            # self.log.print_eval(self.eval_model(self.data))
            v_MAE = self.evaluate_data_graph(batch_size, 'validation',
                                             save_eval_to_files=True, eval_files=MODEL_DIR + ""+str(e)+"/")
            summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=v_MAE, ), ])
            self.writers['validation_set'].add_summary(summary, e)

            t_MAE = self.evaluate_data_graph(batch_size, 'training_na')
            summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=t_MAE, ), ])
            self.writers['training_set'].add_summary(summary, e)

        # print("--final evals of last train step:")
        # # self.log.print_train_epoch(epochs, self.loss_val)
        # self.log.print_eval(self.eval_model(self.data))
        #
        # self.print_plot()

    def snapshot_lr(self, S, t, lr0=0.1):
        """
        :param S: length of single cycle in iterations
        :param t: overal iteration
        :param lr0: initial learning rate
        :return: updated learning rate
        """
        lr = lr0 / 2 * (np.cos((np.pi * np.mod(t, S)) / (S)) + 1)
        self.graph.x.optimizer.lr = lr
        return lr

    def decrease_learning_rate(self, cooldown, cooldown_cnt, epsilon, factor, last_loss, lr, min_lr, patience,
                               stagnation):
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
                        print("new lr: " + str(lr))
            else:
                stagnation = 0
        last_loss = self.loss_val
        return self.loss_val



    def evaluate_data_graph(self, batch_size, type='training', save_eval_to_files=False, eval_files=None):

        self.log.print("starting evaluation of %s dataset " % type)
        data_service.get_data_struct().reload_epoch_image_lists()
        data_service.get_data_struct().reload_image_lists()

        epoch_steps = data_service.get_data_struct().count_steps_in_epoch(type=type, batch_size=batch_size)

        evals_dict = {}

        whole_MAE = 0

        for i in range(epoch_steps):

            # get random data  from data_struct
            self.step_data = data_service.get_data_struct().get_random_data(batch_size, type)

            if len(self.step_data.x) > 0:
                step_eval_model = self.eval_model(self.step_data, get_output=save_eval_to_files)
                whole_MAE += step_eval_model.loss / epoch_steps

                if save_eval_to_files:
                    for s in range(len(self.step_data.x)):
                        evals_dict[self.step_data.filenames[s]] = {"gt": unscaleAge(self.step_data.y[s])[0], "output": unscaleAge([step_eval_model.output[s]])[0]}


            print("\r%3d%% out of %5d (last step MAE: %10.8f)" %  ((i*100/epoch_steps), epoch_steps, step_eval_model.loss), end='')

        self.log.print("\r %10s MAE: %10.8f" % (type, whole_MAE), self.log.Styles.HEADER)

        if save_eval_to_files:
            save_evals(eval_files, "evals_"+"training", evals_dict)

        return whole_MAE


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

def main_model(data, graph_struct, create_new=False, train=True, save=True, evaluate=False, use=False, lr_drop=0):
    tf.reset_default_graph()

    # -- start session
    with tf.Session() as sess:
        # K.set_session(sess)

        if create_new:
            graphModel = graph_struct.get_graph()


        # init tensorboard
        train_writer = tf.summary.FileWriter(TB_DIR + FLAGS.summaries_dir + '/train_batch',
                                             sess.graph)

        validation_set_writer = tf.summary.FileWriter(
            TB_DIR + FLAGS.summaries_dir + '/validation')

        train_set_writer = tf.summary.FileWriter(
            TB_DIR + FLAGS.summaries_dir + '/train')

        lr_writer = tf.summary.FileWriter(
            TB_DIR + FLAGS.summaries_dir + '/learning_rate')

        writers = {
            'train_batch': train_writer,
            'validation_set': validation_set_writer,
            'training_set': train_set_writer,
            'learning_rate': lr_writer
        }

        merged = tf.summary.merge_all()

        # -- init graph
        sess.run(tf.global_variables_initializer())

        if not create_new:
            model = load_model(MODEL_DIR_TO_LOAD, CHECKPOINT_NAME)
            graphModel = graph_struct.get_graph_ops(model)


        if data is None:
            data_service.get_data_struct().init_sess(sess, graph_struct)
            data = data_service.get_data_struct().get_all_data(None)

        s = Session(sess, graphModel, data, writers)

        # -- run ops on graph
        if train:
            s.log.print_info("Training started (batch size: %d, GPUs: %d, max epochs: %d)" %
                             (FLAGS.train_batch_size, FLAGS.gpus, FLAGS.how_many_epochs))
            s.train_graph(FLAGS.how_many_epochs, FLAGS.train_batch_size, lr_drop)
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
            s.evaluate_data_graph(FLAGS.validation_batch_size, 'training_na')

            # s.print_plot()

        if use:
            s.log.print_info("Using graph")
            # print(np.transpose(data.x[0]))
            # print(s.use_model(np.transpose(data.x[:10]), np.transpose(data.gender[:10])))
            print("for expected values:")
            print(np.squeeze(unscaleAge(data.y[:10])))
            print("model calculated:")
            evals = s.use_model(data.x[:10], data.gender[:10])
            print(evals)
            return evals


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

def save_evals(file_path, file_name, item):
    # create dir if not exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path+file_name, 'wb') as fp:
        pickle.dump(item, fp)
    with open(file_path+file_name+'.txt', 'w') as f:
        if type(item) is dict:
            for i in item:
                f.write("%s\t%s\t%s\n" % (i, item[i]["gt"], item[i]["output"]))
        else:
            for i in item:
                f.write("%s\n" % i)

def read_evals(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)

def handle_command_line_args():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-b", "--tra_batch", dest="tra_batch", help="size of a training batch", type="int", default=FLAGS.train_batch_size)
    parser.add_option("-v", "--val_batch", dest="val_batch", help="size of a validation batch", type="int", default=FLAGS.validation_batch_size)
    parser.add_option("-e", "--epochs", dest="epochs", help="amount of epochs", type="int", default=FLAGS.how_many_epochs)
    parser.add_option("-g", "--gpus", dest="gpus", help="amount of available GPUs", type="int", default=FLAGS.gpus)
    parser.add_option("-i", "--img_dir", dest="img_dir", help="input images dir", type="string", default=FLAGS.image_dir)

    parser.add_option("-m", "--model_save_dir", dest="model_save_dir", help="dir for saving model", type="string", default="trained_models/")
    parser.add_option("-l", "--model_load_dir", dest="model_load_dir", help="trained model dir", type="string", default="trained_models/")
    parser.add_option("-x", "--tb_dir", dest="tb_dir", help="dir for tensorboard logs", type="string", default="tensorboard_logs/")


    parser.add_option("-n", "--new_model", dest="new_model", help="use for creating new model", action='store_true')
    parser.add_option("-t", "--train", dest="train", help="use to train model", action='store_true')
    parser.add_option("-s", "--save_model", dest="save_model", help="save model after training", action='store_true')
    parser.add_option("-o", "--evaluate", dest="evaluate", help="use to evaluate model (to be used with importing model)", action='store_true')
    parser.add_option("-z", "--use", dest="use", help="use to test reading input and model", action='store_true')


    parser.add_option("-r", "--lr_drop", dest="lr_drop", help="if 0 lr drops steadily, if larger - drops in cyclic cosine annealing manner for snapshot", type="int", default=5)

    (options, args) = parser.parse_args()
    FLAGS.train_batch_size = options.tra_batch
    FLAGS.validation_batch_size = options.val_batch
    FLAGS.how_many_epochs = options.epochs
    FLAGS.gpus = options.gpus
    FLAGS.image_dir = options.img_dir

    MODEL_DIR = options.model_save_dir
    MODEL_DIR_TO_LOAD = options.model_load_dir
    TB_DIR = options.tb_dir

    N = options.new_model
    T = options.train
    S = options.save_model
    O = options.evaluate
    Z = options.use
    LR_DROP = options.lr_drop

    return MODEL_DIR, MODEL_DIR_TO_LOAD, TB_DIR, N, T, S, O, Z, LR_DROP


if __name__ == "__main__":

    MODEL_DIR = "trained_models/"
    MODEL_DIR_TO_LOAD = MODEL_DIR+"55000/"
    TB_DIR = "tensorboard_logs/"


    CHECKPOINT_NAME = "model"
    n_samples = 1000 # applicable only if data struct allows to generate a specified amount of data
    batch_size = 16
    # iterations = 1000

    FLAGS.validation_batch_size = batch_size
    FLAGS.train_batch_size = batch_size
    FLAGS.how_many_epochs = 20
    FLAGS.gpus = 1

    N = False
    T = False
    S = False
    O = False
    Z = True

    # FLAGS.create_bottlenecks = True

    MODEL_DIR, MODEL_DIR_TO_LOAD, TB_DIR, N, T, S, O, Z, LR_DROP = handle_command_line_args()


    # #options manually overriding args
    # N = True
    # T = True
    # MODEL_DIR = "trained_models/saving_test_sn_epochs/"
    # FLAGS.image_dir = "C:/Users/Emilia/Pycharm Projects/BoneAge/training_dataset/imgs_sm"
    # #FLAGS.image_dir = "C:/Users/Emilia/Pycharm Projects/BoneAge/training_dataset/FM_labeled_train_validate"
    # LR_DROP = 5


    if not os.path.exists(TB_DIR):
        os.makedirs(TB_DIR)

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
    main_model(data, graph_struct, create_new=N, train=T, save=S, evaluate=O, use=Z, lr_drop=LR_DROP)
    # main_model(data, graph_struct, create_new=False, train=True, save=True, evaluate=False, use=True)
    # main_model(data, graph_struct, create_new=False, train=False, save=False, evaluate=True, use=False)

    # x = []
    # x.append([130.93224, 128.6378, 129.80742, 129.99046, 129.19624, 129.46765, 128.52744, 134.57977, 135.84741, 128.91914])
    # x.append([118.54085, 117.13107, 118.11336, 118.27703,  117.18795,  117.035736, 118.32126,  117.72592,  117.335464, 118.73112])
    # x.append([126.884766, 125.765366, 125.21182, 125.356064, 125.67647, 125.27998, 126.10977, 125.253265, 125.13985, 130.3324])
    #
    # np.mean(x, 0)

    ##if using multiple models
    # d = "trained_models/i/"
    #
    # models_to_use = [d+'8000/', d+'19000/', d+'30000/']
    # evaluations = []
    #
    # g = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # maes = []
    # for model in models_to_use:
    #     MODEL_DIR_TO_LOAD = model
    #     evaluation = main_model(data, graph_struct, create_new=False, train=False, save=False, evaluate=False, use=True)
    #     save_evals(model + "evals.txt", evaluation.tolist())
    #     evaluations.append(evaluation.tolist())
    #
    #     maes.append(np.mean(np.subtract(evaluation.tolist(),g)))
    #
    # print(evaluations)
    # mean_evaluations = np.mean(evaluations, 0)
    # print("mean evals:")
    # print(mean_evaluations)
    #
    # print("maes:")
    # print(maes)
    # print("mean mae:")
    # print(np.mean(np.subtract(mean_evaluations,g)))




# trainingData - to prepare train and validation sets of images in 2 folders with saved gender in file names
# bottleneck_maker (from train3) - can be used to produce bottlenecks from previously prepared divided images
# test_new = (from test) to asses either single image or set of images with gender taken from csv
# to profile: in terminal python -m cprovilev main2.py
