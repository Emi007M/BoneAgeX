
class Data:

    def __init__(self, x=None, y=None, gender=None, filenames=None):
        self.x = x
        self.y = y
        self.gender = gender
        self.filenames = filenames

    def init_sess(self, sess, graph):
        self.sess = sess
        self.graph = graph

    def get_all_data(self, mount):
        raise NotImplementedError("calling abstract get_all_data on Data object")

    def get_random_data(self, amount, type=None):
        raise NotImplementedError("calling abstract get_random_data on Data object")

    def reload_epoch_image_lists(self):
        pass

    def count_steps_in_epoch(self, image_lists=None, type=None, batch_size=None):
        pass
