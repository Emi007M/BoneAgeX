import os.path
import numpy as np
import sys, os
from data.IData import Data
from data.bottleneck.helpers.tf_methods import get_image_file_from_path
from graphs.GraphService import GraphService, GraphType
from main import main_model
from contextlib import contextmanager
import csv
from os import walk

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_images_as_data_for_evaluation(rtg_file_dir, gender): #0=female, 1=male

    filename = os.path.basename(rtg_file_dir)
    image_data = get_image_file_from_path(rtg_file_dir)

    return Data([image_data], [0], [gender], [filename])



# def print_as_table(models_to_use, data, evaluations):
#     mean_evaluations = np.mean(evaluations, 0)
#
#     header = cell("image") + cell("gender")
#     for model_dir in models_to_use:
#         header += cell(model_dir)
#     header += cell("mean_eval")
#     print(header)
#     for i in range(len(data.x)):
#         row = cell(data.filenames[i]) + cell(num_to_gender(data.gender[i]))
#         for x in range(len(evaluations)):
#             row += cell(int(round(evaluations[x][i])))
#         row += "\t" + cell(int(round(mean_evaluations[i])))
#         print(row)


models_to_use = [
    "W:/Python Projects/BoneAgeX/ssh-final_models/trained_models/fm9/33399/",
    "W:/Python Projects/BoneAgeX/ssh-final_models/trained_models/fm9/82831/",
    "W:/Python Projects/BoneAgeX/ssh-final_models/trained_models/fm9/116231/"
]
CHECKPOINT_NAME = "model"


# rtg_file_dir = "W:/Python Projects/BoneAgeX/eval_gui_data/rtg.jpg"
# gender = 0
# model_no = 0
def evalBoneAge(rtg_file_dir, gender, model_no):

    data = get_images_as_data_for_evaluation(rtg_file_dir, gender)

    graph_service = GraphService(GraphType.BAAkerasJpeg)
    graph_struct = graph_service.get_graph_struct(500 * 500 * 3, 1)

    # g = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # maes = []
    with suppress_stdout():
        model_dir = models_to_use[model_no]
        evaluation = main_model(data, graph_struct, model_dir=model_dir,
                                create_new=False, train=False, save=False,
                                evaluate=False, use=True)

        return np.round(float(evaluation),2)