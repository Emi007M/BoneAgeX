from datetime import datetime
from time import gmtime, strftime

class Flags(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Flags, cls).__new__(cls, *args, **kwargs)

        name = '12'
        #image_dir_folder = 'three_classes'
        #image_dir_folder = 'FM_labeled_train_validate'
        image_dir_folder = 'imgs_sm'
        epochs = 16
        create_bottlenecks = 0
        batch_size = 16 # ?

        d = 'C:/Users/Emilia/Pycharm Projects/BoneAge/'

        # image_dir,  summaries_dir, output_graph ?,

        cls._instance.image_dir =            d + 'training_dataset/' + image_dir_folder  # Path to folders of labeled images
        cls._instance.create_bottlenecks =   create_bottlenecks  # Path to folders of labeled images
        cls._instance.bottleneck_dir =       d + 'bottleneck/' + image_dir_folder  # Path to cache bottleneck layer values as files
        cls._instance.output_graph =         d + 'model/output_graph/' + name + '/output_graph' + '-' + strftime("%m-%d %H.%M.%S", gmtime()) # Where to save the trained graph
        cls._instance.output_graph_fin =     'model'  # Where to save the trained graph
        cls._instance.intermediate_output_graphs_dir = d + 'model/intermediate_graph/' + name + '/'  # Where to save the intermediate graphs
        cls._instance.intermediate_store_frequency = 0  # How many steps to store intermediate graph. If "0" then will not store
        cls._instance.output_labels =        d + 'model/output_labels.txt'  # Where to save the trained graph's labels
        # cls._instance.summaries_dir = d + 'model/models/retrain_logs/' + name + '-' + strftime("%Y-%m-%d %H.%M.%S",
        #                                                                                        gmtime())  # Where to save summary logs for TensorBoard
        cls._instance.summaries_dir = strftime("%Y-%m-%d %H.%M.%S", gmtime())  # Where to save summary logs for TensorBoard
        cls._instance.how_many_epochs =      epochs  # How many training steps to run before ending
        cls._instance.learning_rate =        0.001  # How large a learning rate to use when training
        cls._instance.testing_percentage =   0  # What percentage of images to use as a test set
        cls._instance.validation_percentage = 15  # What percentage of images to use as a validation set
        cls._instance.eval_step_interval =   50  # How often to evaluate the training results
        cls._instance.train_batch_size =     batch_size  # How many images to train on at a time
        cls._instance.validation_batch_size = batch_size  # -..-
        cls._instance.model_dir =            d + 'model/imagenet' # Path to classify_image_graph_def.pb, imagenet_synset_to_human_label_map.txt, and imagenet_2012_challenge_label_map_proto.pbtxt
        cls._instance.final_tensor_name =    'output_layer'  # The name of the output classification layer in the retrained graph
        cls._instance.architecture =         'inception_v3'  # .
        cls._instance.saved_model_dir =      d + 'model/saved_models/' + name + '-' + strftime("%Y-%m-%d %H.%M.%S", gmtime()) + '/' # Where to save the exported graph
        cls._instance.gpus = 1 # how many gpus available

        return cls._instance
