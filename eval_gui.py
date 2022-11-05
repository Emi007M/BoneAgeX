from tkinter import *
from eval_gui_data.use_models import evalBoneAge
from threading import *
from eval_gui_data.gui_widgets.top_frame import TopFrame
from eval_gui_data.gui_widgets.rtg_image_choice_frame import RtgChoiceFrame
from eval_gui_data.gui_widgets.gender_choice_frame import GenderChoiceFrame
from eval_gui_data.gui_widgets.center_frame import CenterFrame
from eval_gui_data.gui_widgets.eval_frame import EvalFrame
from eval_gui_data.gui_widgets.bottom_frame import BottomFrame

# Defining i/o data
gender_val = None # 0 - Female, 1 = Male
rtg_val = None # 500x500 rgb jpg image
eval_rtg_file = "W:/Python Projects/BoneAgeX/eval_gui_data/rtg.jpg"
rtg_file_name = None

evaluations = [None, None, None]

# Defining colors and fonts
color_bg_dark = "#192229"
color_bg_lighter_dark = "#293239"
color_text_primary = "white"
color_text_secondary = "#ababab"
font_title_header = 'Calibri 16 bold'
font_title_subheader = 'Calibri 10'
font_section_header = 'Calibri 13 bold'
font_eval_doctor = 'Calibri 10 bold'
font_eval_y = 'Calibri 16'
font_eval_m = 'Calibri 11'
font_eval_a = 'Calibri 11'

# Creating a window
window = Tk()
window.title("Ocena Wieku Kostnego")
window.geometry("800x600")
window.config(bg=color_bg_dark)

# Defining gui imgs
img_dir = "W:/Python Projects/BoneAgeX/eval_gui_data/gui_imgs/"

# Defining layout

window.grid_columnconfigure(0, weight=4)
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=4)
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(2, weight=1)

top_frame = Frame(window, width=800, height=20, padx=10, pady=5, bg=color_bg_dark)
top_frame.grid(row=0, column=0, columnspan=3, sticky="ew")

preview_frame = Frame(window, width=500, height=500, bg=color_bg_dark)
preview_frame.grid(row=1,column=0)
preview_frame.grid_columnconfigure(0, weight=1)

center_frame = Frame(window, width=50, height=500, bg=color_bg_dark)
center_frame.grid(row=1, column=1, sticky="ns")
center_frame.grid_rowconfigure(0, weight=3)

eval_frame = Frame(window, width=200, height=500, bg=color_bg_dark)
eval_frame.grid(row=1, column=2)
eval_frame.grid_columnconfigure(0, weight=1)

bottom_frame = Frame(window, width=800, height=20, padx=10, pady=5, bg=color_bg_dark)
bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew")


# Defining ui elements

# - top frame
top_frame_widget = TopFrame(top_frame, color_bg_dark, color_text_primary, font_title_header, color_text_secondary, font_title_subheader)

# - preview frame

def getInputImageCallback(img):
    img.save(eval_rtg_file)
    changeRtgVal(img)

def updateInputFileNameCallback(file_name):
    global rtg_file_name
    rtg_file_name = file_name

rtg_choice_frame_widget = RtgChoiceFrame(
    preview_frame, img_dir, color_bg_dark, color_text_primary, font_section_header, color_text_secondary, font_title_subheader)
rtg_choice_frame_widget.registerCallbacks(getInputImageCallback, updateInputFileNameCallback)

def updateGenderValCallback(val):
    global gender_val
    gender_val = val
    updateEvalButtonState()
    updateLogs()

gender_choice_frame_widget = GenderChoiceFrame(preview_frame, img_dir, color_bg_dark, color_text_primary, font_section_header)
gender_choice_frame_widget.registerCallbacks(updateGenderValCallback)

# - center frame

def launchEvaluationCallback():
    clearEvals()
    updateLogs()

    t1 = Thread(target=launchEvaluation)
    t1.start()

center_frame_widget = CenterFrame(center_frame, img_dir, color_bg_dark)
center_frame_widget.registerCallbacks(launchEvaluationCallback)

# - eval frame
eval_frame_widget = EvalFrame(
    eval_frame, img_dir, color_bg_dark, color_text_primary, font_section_header, font_eval_doctor, color_bg_lighter_dark, font_eval_y, font_eval_m)

# - bottom frame
bottom_frame_widget = BottomFrame(bottom_frame, color_bg_dark, color_text_secondary, font_title_subheader)

# Defining auxiliary methods

def clearEvals():
    global evaluations, doctor_labels, eval_labels
    evaluations = [None, None, None]

    updateEvals()

def changeRtgVal(val):
    global rtg_val
    rtg_val = val
    updateEvalButtonState()
    updateLogs()

def updateEvalButtonState():
    center_frame_widget.updateEvalButtonState(gender_val, rtg_val)

def launchEvaluation():
    global evaluations
    gender = gender_val
    rtg_file = eval_rtg_file

    for model_no in [0, 1, 2]:

        launchPartialEvals(model_no)

        evaluation = evalBoneAge(rtg_file, gender, model_no=model_no)
        evaluations[model_no] = evaluation

        updateEvals()
        updateLogs()

def launchPartialEvals(model_no):
    eval_frame_widget.launchPartialEvals(model_no)

def updateEvals():
    eval_frame_widget.updateEvals(evaluations, gender_val)

def updateLogs():
    bottom_frame_widget.updateLogs(rtg_file_name, gender_val, evaluations)


# Window initialization

updateEvalButtonState()
clearEvals()
updateLogs()

window.mainloop()
