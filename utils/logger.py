from termcolor import *
import colorama
colorama.init()

class Logger(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)

        return cls._instance

    class Styles:
        HEADER = '\033[95m'     # pink
        OKBLUE = '\033[94m'     # blue
        OKGREEN = '\033[92m'    # green
        WARNING = '\033[93m'    # yellow
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    # def print_train(self, step, loss_val, W_val, b_val):
    #     print("step " + str(step) + ": \t" + str(loss_val) + " \t" + str(W_val) + " \t" + str(b_val))
    def print_train(self, step, epoch_steps, loss_val):
        self.__print("step " + str(step) + " \tof " + str(epoch_steps) +": \t " + str(loss_val))

    def print_eval(self, eval_model):
        self.__print("evaluation: \n" + eval_model.to_string())
        # self.__print(("validation MAE: " + str(eval_model.loss)), self.Styles.HEADER)

    def print_train_epoch(self, epoch, loss_val):
        self.__print("epoch " + str(epoch) + ": \t" + str(loss_val))

    def print(self, text, type=""):
        self.__print(text, type)

    def print_warning(self, text):
        self.__print(text, self.Styles.WARNING)
    def print_error(self, text):
        self.__print(text, self.Styles.FAIL)
    def print_info(self, text):
        self.__print('\n' + text, self.Styles.OKBLUE)
    def print_success(self, text):
        self.__print(text, self.Styles.OKGREEN)

    def __print(self, text, type=""):
        print(type + text + self.Styles.ENDC)

