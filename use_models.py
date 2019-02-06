import os.path
import numpy as np
import sys, os
from data.IData import Data
from data.bottleneck.helpers.tf_methods import get_image_file_from_path
from graphs.GraphService import GraphService, GraphType
from main import main_model
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_images_as_data_for_evaluation(dir, image_list=[], gender_list=[]):
    jpegs = []
    ground_truths = []
    filenames = []
    genders = []

    if len(image_list) is 0:
        for (dirpath, dirnames, filenames) in os.walk(dir):
            image_list.extend(filenames)
            break

    # Retrieve all images.
    for image_name in image_list:
        image_data = get_image_file_from_path(dir + "/" + image_name)
        jpegs.append(image_data)
        ground_truths.append(0)
        filenames.append(image_name)

        genders.append(1 if os.path.basename(image_name)[0] is 'M' else 0)

    return Data(jpegs, ground_truths, genders, filenames)

def cell(input):
    return "\t\'" + str(input) + "\';"

def print_as_table(models_to_use, data, evaluations):
    mean_evaluations = np.mean(evaluations, 0)

    header = cell("image") + cell("gender")
    for model_dir in models_to_use:
        header += cell(model_dir)
    header += cell("mean_eval")
    print(header)
    for i in range(len(data.x)):
        row = cell(data.filenames[i]) + cell(data.gender[i])
        for x in range(len(evaluations)):
            row += cell(evaluations[x][i])
        row += "\t" + cell(mean_evaluations[i])
        print(row)


dir = "C:/Users/Emilia/Pycharm Projects/BoneAge/training_dataset/imgs_sm/validate_not_augmented/108"
imgs = ["M_42_6839368.jpeg", "M_58_5642036.jpeg", "M_11_9161997.jpeg", "M_30_2021571.jpeg"]

# right now images have to be 500x500 and with gender in filename
data = get_images_as_data_for_evaluation(dir, imgs)

graph_service = GraphService(GraphType.BAAkerasJpeg)
graph_struct = graph_service.get_graph_struct(500 * 500 * 3, 1)

models_to_use = [
    # "M:/Desktop/ssh/trained_models/fm3/c136815/",
    # "M:/Desktop/ssh/trained_models/fm3/c273631/",
    "M:/Desktop/ssh/fm9/trained_models/33399/",
    "M:/Desktop/ssh/fm9/trained_models/82831/",
    "M:/Desktop/ssh/fm9/trained_models/116231/",
    "M:/Desktop/ssh/fm9/trained_models/177687/",
]
CHECKPOINT_NAME = "model"
evaluations = []

# g = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
# maes = []
with suppress_stdout():
    for model_dir in models_to_use:
        evaluation = main_model(data, graph_struct, model_dir=model_dir,
                                create_new=False, train=False, save=False,
                                evaluate=False, use=True)

        # save_evals(model_dir, "evals.txt", evaluation.tolist())
        evaluations.append(evaluation.tolist())
        # maes.append(np.mean(np.subtract(evaluation.tolist(), g)))

# print(evaluations)
# mean_evaluations = np.mean(evaluations, 0)
# print("mean evals:")
# print(mean_evaluations)
#
# print("maes:")
# print(maes)
# print("mean mae:")
# print(np.mean(np.subtract(mean_evaluations, g)))
#
# print("-------")


print_as_table(models_to_use, data, evaluations)
