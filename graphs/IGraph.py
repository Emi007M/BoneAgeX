
class Graph:

    def __init__(self, x=None, y=None, W=None, b=None, y_pred=None, loss=None, opt_operation=None, batch_size=None, gender_x=None):
        self.x = x
        self.y = y
        self.W = W
        self.b = b
        self.y_pred = y_pred
        self.loss = loss
        self.opt_operation = opt_operation
        self.batch_size = batch_size
        self.gender_x = gender_x

    def set_tensor_lengths(self, input_tensor_size, output_tensor_size):
        self.input_tensor_size = input_tensor_size
        self.output_tensor_size = output_tensor_size

    @staticmethod
    def get_graph():
        raise NotImplementedError("calling abstract get_graph on Graph object")

    @staticmethod
    def get_graph_ops(model = None):
        raise NotImplementedError("calling abstract get_graph_ops on Graph object")
