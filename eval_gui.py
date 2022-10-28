# Importing Tkinter library
# in the environment
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from eval_gui_data.crop_image import resize_and_crop
from eval_gui_data.use_models import evalBoneAge
import os
import numpy as np
import time
from threading import *

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
color_text_sedondary = "#ababab"
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
#window.eval('tk::PlaceWindow . center')
#window.grid_columnconfigure(0,weight=1)

# Defining gui imgs
gui_dir = "W:/Python Projects/BoneAgeX/eval_gui_data/gui_imgs/"

img_search_button_leave = PhotoImage(file=gui_dir + "search_button.png")
img_search_button_hover = PhotoImage(file=gui_dir + "search_button_hover.png")
img_search_button = img_search_button_leave

img_hand_placeholder = PhotoImage(file=gui_dir + "hand.png")
img_rtg = img_hand_placeholder

img_gender_switch_0 = PhotoImage(file=gui_dir + "gender_0.png")
img_gender_switch_0_hover = PhotoImage(file=gui_dir + "gender_0_hover.png")
img_gender_switch_F = PhotoImage(file=gui_dir + "gender_F.png")
img_gender_switch_M = PhotoImage(file=gui_dir + "gender_M.png")
img_gender_button = img_gender_switch_0

img_center_line_n = PhotoImage(file=gui_dir + "center_line_n.png")
img_center_line_s = PhotoImage(file=gui_dir + "center_line_s.png")
img_eval_button_leave = PhotoImage(file=gui_dir + "eval_button.png")
img_eval_button_hover = PhotoImage(file=gui_dir + "eval_button_hover.png")
img_eval_button = img_eval_button_leave

img_lek_0 = [PhotoImage(file=gui_dir + "lekA_0.png"), PhotoImage(file=gui_dir + "lekA.png")]
img_lek_1 = [PhotoImage(file=gui_dir + "lekB_0.png"), PhotoImage(file=gui_dir + "lekB.png")]
img_lek_2 = [PhotoImage(file=gui_dir + "lekC_0.png"), PhotoImage(file=gui_dir + "lekC.png")]
doc_imgs = [img_lek_0, img_lek_1, img_lek_2]
#
# img_lekB_0 = PhotoImage(file=gui_dir + "lekB_0.png")
# img_lekB = PhotoImage(file=gui_dir + "lekB.png")
# img_lekC_0 = PhotoImage(file=gui_dir + "lekC_0.png")
# img_lekC = PhotoImage(file=gui_dir + "lekC.png")

img_eval_ladder = PhotoImage(file=gui_dir + "ladder.png")

img_eval_placeholder = PhotoImage(file=gui_dir + "gender_icon_placeholder.png")
img_eval_F1 = PhotoImage(file=gui_dir + "F1.png")
img_eval_F2 = PhotoImage(file=gui_dir + "F2.png")
img_eval_F3 = PhotoImage(file=gui_dir + "F3.png")
img_eval_M1 = PhotoImage(file=gui_dir + "M1.png")
img_eval_M2 = PhotoImage(file=gui_dir + "M2.png")
img_eval_M3 = PhotoImage(file=gui_dir + "M3.png")
eval_imgs = [[img_eval_F1, img_eval_F2, img_eval_F3], [img_eval_M1, img_eval_M2, img_eval_M3]]


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

center_frame = Frame(window, width=50, height=500, bg=color_bg_dark)
center_frame.grid(row=1, column=1, sticky="ns")
center_frame.grid_rowconfigure(0, weight=3)
center_frame.grid_rowconfigure(2, weight=3)

eval_frame = Frame(window, width=200, height=500, bg=color_bg_dark)
eval_frame.grid(row=1, column=2)
eval_frame.grid_columnconfigure(0, weight=1)

bottom_frame = Frame(window, width=800, height=20, padx=10, pady=5, bg=color_bg_dark)
bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew")


# Defining ui elements

# - top frame
title_label = Label(top_frame, text="Bone Age X", fg=color_text_primary, bg=color_bg_dark, font=(font_title_header))
title_label.grid(row=1, column=0, sticky="w")
title_sublabel = Label(top_frame, text="ocena wieku kostnego", fg=color_text_sedondary, bg=color_bg_dark, font=(font_title_subheader))
title_sublabel.grid(row=0, column=0, sticky="w")

# - preview frame
section_preview_label = Label(preview_frame, text="Zdjęcie RTG", fg=color_text_primary, bg=color_bg_dark, font=(font_section_header))
section_preview_label.grid(row=0, column=0, sticky="w")

search_button = Button(preview_frame, image=img_search_button, bg=color_bg_dark, activebackground=color_bg_dark, bd=0, cursor="hand2")
search_button.grid(row=1, column=0, sticky="w")

preview_rtg_label = Label(preview_frame, image=img_rtg, bg=color_bg_dark)
preview_rtg_label.grid(row=2, column=0, padx=10, pady=50)

section_preview_label = Label(preview_frame, text="Płeć", fg=color_text_primary, bg=color_bg_dark, font=(font_section_header))
section_preview_label.grid(row=3, column=0, sticky="w")

gender_switch_button = Button(preview_frame, image=img_gender_switch_0, bg=color_bg_dark, activebackground=color_bg_dark, bd=0, cursor="hand2")
gender_switch_button.grid(row=4,column=0)


# - center frame
center_line_n = Label(center_frame, image=img_center_line_n, bg=color_bg_dark)
center_line_n.grid(row=0,column=0, sticky="s")
center_line_s = Label(center_frame, image=img_center_line_s, bg=color_bg_dark)
center_line_s.grid(row=2,column=0, sticky="n")

eval_button = Button(center_frame, image=img_eval_button, bg=color_bg_dark, activebackground=color_bg_dark, bd=0, cursor="hand2")
eval_button.grid(row=1, column=0, sticky="ns")

# - eval frame
section_eval_label = Label(eval_frame, text="Ocena wieku", fg=color_text_primary, bg=color_bg_dark, font=(font_section_header))
section_eval_label.grid(row=0, column=0, sticky="w")

doctors_frame = Frame(eval_frame, bg=color_bg_dark)
doctors_frame.grid(row=1, column=0, columnspan=3)

doctor_0 = Label(doctors_frame, image=doc_imgs[0][0], bg=color_bg_dark)
doctor_0.grid(row=0,column=0, padx=26, pady=10)
doctor_1 = Label(doctors_frame, image=doc_imgs[1][0], bg=color_bg_dark)
doctor_1.grid(row=0,column=1, padx=13, pady=10)
doctor_2 = Label(doctors_frame, image=doc_imgs[2][0], bg=color_bg_dark)
doctor_2.grid(row=0,column=2, padx=26, pady=10)
doctor_labels = [doctor_0, doctor_1, doctor_2]

eval_0_label = Label(doctors_frame, text=". . .", fg=color_text_primary, bg=color_bg_dark, font=(font_eval_doctor))
eval_0_label.grid(row=1, column=0)
eval_1_label = Label(doctors_frame, text=". . .", fg=color_text_primary, bg=color_bg_dark, font=(font_eval_doctor))
eval_1_label.grid(row=1, column=1)
eval_2_label = Label(doctors_frame, text=". . .", fg=color_text_primary, bg=color_bg_dark, font=(font_eval_doctor))
eval_2_label.grid(row=1, column=2)
eval_labels = [eval_0_label, eval_1_label, eval_2_label]

ladder_label = Label(eval_frame, image=img_eval_ladder, bg=color_bg_dark)
ladder_label.grid(row=2, column=0, columnspan=3, padx=10, pady=20)

final_eval_frame = Frame(eval_frame, bg=color_bg_lighter_dark)
final_eval_frame.grid(row=3, column=0, columnspan=3, sticky="we", padx=30)
final_eval_frame.grid_columnconfigure(0, weight=1)
final_eval_frame.grid_columnconfigure(1, weight=1)

final_eval_icon = Label(final_eval_frame, image=img_eval_placeholder, bg=color_bg_lighter_dark)
final_eval_icon.grid(row=0, column=0, rowspan=3, padx=10)

final_eval_year_label = Label(final_eval_frame, text="x lat", fg=color_text_primary, bg=color_bg_lighter_dark, font=(font_eval_y))
final_eval_year_label.grid(row=0, column=1, sticky="w")
final_eval_month_label = Label(final_eval_frame, text="x miesięcy", fg=color_text_primary, bg=color_bg_lighter_dark, font=(font_eval_m))
final_eval_month_label.grid(row=1, column=1, sticky="w")
final_eval_accuracy_label = Label(final_eval_frame, text="± 6M", fg=color_text_sedondary, bg=color_bg_lighter_dark, font=(font_eval_a))
final_eval_accuracy_label.grid(row=2, column=1, sticky="w")





# - bottom frame
logs = Label(bottom_frame, text="xxx", fg=color_text_sedondary, bg=color_bg_dark, font=(font_title_subheader))
logs.grid(row=0, column=0, sticky="w")


# Defining button behaviours

# - search button
def onSearchButtonEnter(event):
    global img_search_button
    img_search_button = img_search_button_hover
    search_button.config(image=img_search_button)

def onSearchButtonClick(event):
    global img_search_button, preview_rtg_label, rtg_val, img_rtg
    img_search_button = img_search_button_hover
    search_button.config(image=img_search_button)

    img = openFileDialog()
    changeRtgVal(img)
    img_rtg = img
    #preview_rtg_label.config(image=img)
    preview_rtg_label = Label(preview_frame, image=img_rtg, bg=color_bg_dark)
    preview_rtg_label.grid(row=2, column=0, padx=10, pady=50)

def onSearchButtonLeave(event):
    global img_search_button
    img_search_button = img_search_button_leave #ImageTk.PhotoImage(Image.open(r'img1'))
    search_button.config(image=img_search_button)

# - eval button
def onEvalButtonEnter(event):
    global img_eval_button

    if eval_button["state"] == "disabled":
        return

    img_eval_button = img_eval_button_hover
    eval_button.config(image=img_eval_button)

def onEvalButtonClick(event):
    global img_eval_button, evaluations

    if eval_button["state"] == "disabled":
        return

    img_eval_button = img_eval_button_hover
    eval_button.config(image=img_eval_button)

    clearEvals()
    updateLogs()

    t1 = Thread(target=launchEvaluation)
    t1.start()


def launchEvaluation():
    for model_no in [0,1,2]:
        evaluation = evalBoneAge(eval_rtg_file, gender_val, model_no=model_no)
        evaluations[model_no] = evaluation

        updateEvals()
        updateLogs()

    updateFinalEval()


def onEvalButtonLeave(event):
    global img_eval_button
    img_eval_button = img_eval_button_leave #ImageTk.PhotoImage(Image.open(r'img1'))
    eval_button.config(image=img_eval_button)


search_button.bind('<Enter>',  onSearchButtonEnter)
search_button.bind('<Button-1>',  onSearchButtonClick)
search_button.bind('<Leave>',  onSearchButtonLeave)

eval_button.bind('<Enter>', onEvalButtonEnter)
eval_button.bind('<Button-1>', onEvalButtonClick)
eval_button.bind('<Leave>', onEvalButtonLeave)

def updateLogs():
    global logs, rtg_val, gender_val
    logs.config(text="RTG: "+str(rtg_file_name)+", Gender: "+ str(gender_val) + " => Bone age: " + str(evaluations))
    #window.update()

def evalToTextShort(val):
    years, months = divmod(int(val), 12)
    return f"{years}L {months}M"

def getEvalImgByValAndGender(months, gender):
    global eval_imgs
    if months < 5*12:
        return eval_imgs[gender][0]
    elif months > 15*12:
        return eval_imgs[gender][2]
    else:
        return eval_imgs[gender][1]

def clearEvals():
    global evaluations, doctor_labels, eval_labels, final_eval_icon, final_eval_year_label, final_eval_month_label, final_eval_accuracy_label
    evaluations = [None, None, None]

    updateEvals()
    updateFinalEval()


def updateEvals():
    global evaluations, doctor_labels, eval_labels
    for i in range(len(evaluations)):
        if evaluations[i] is None:
            doctor_labels[i].config(image=doc_imgs[i][0])
            eval_labels[i].config(text=". . .")
        else:
            doctor_labels[i].config(image=doc_imgs[i][1])
            eval_labels[i].config(text=evalToTextShort(evaluations[i]))

def updateFinalEval():
    global final_eval_icon, final_eval_year_label, final_eval_month_label, final_eval_accuracy_label, evaluations

    if all(elem != None for elem in evaluations):
        mean_eval = int(np.mean(evaluations))

        final_eval_icon.config(image=getEvalImgByValAndGender(mean_eval, gender_val))
        years, months = divmod(int(mean_eval), 12)

        final_eval_year_label.config(text=f"{years} lat")
        final_eval_month_label.config(text=f"{months} miesięcy")
        final_eval_accuracy_label.config(text=f"± _M")

    else:
        final_eval_icon.config(image=img_eval_placeholder)
        final_eval_year_label.config(text=f" ")
        final_eval_month_label.config(text=f"wiek kostny ")
        final_eval_accuracy_label.config(text=f" ")



def updateEvalButtonState():
    if gender_val is None or rtg_val is None:
        eval_button["state"] = "disabled"
        eval_button.config(cursor="arrow")
    else:
        eval_button["state"] = "normal"
        eval_button.config(cursor="hand2")

def changeGenderVal(val):
    global gender_val
    gender_val = val
    updateEvalButtonState()
    updateLogs()

def changeRtgVal(val):
    global rtg_val
    rtg_val = val
    updateEvalButtonState()
    updateLogs()

updateEvalButtonState()
clearEvals()
updateLogs()




#####

# - gender button

def onGenderButtonEnter(event):
    global img_gender_button, logs

    if gender_val == None:
        img_gender_button = img_gender_switch_0_hover

    gender_switch_button.config(image=img_gender_button)

def isOnMaleClick(event):
    male_female_boundry_pos = 60
    return event.x > male_female_boundry_pos

def onGenderButtonClick(event):
    global img_gender_button, gender_val

    if isOnMaleClick(event) and gender_val != 1:
        changeGenderVal(1)
        img_gender_button = img_gender_switch_M
    elif isOnMaleClick(event) == False and gender_val != 0:
        changeGenderVal(0)
        img_gender_button = img_gender_switch_F
    else:
        changeGenderVal(None)
        img_gender_button = img_gender_switch_0_hover
    gender_switch_button.config(image=img_gender_button)

def onGenderButtonLeave(event):
    global img_gender_button
    if gender_val == None:
        img_gender_button = img_gender_switch_0

    gender_switch_button.config(image=img_gender_button)

gender_switch_button.bind('<Enter>',  onGenderButtonEnter)
gender_switch_button.bind('<Button-1>',  onGenderButtonClick)
gender_switch_button.bind('<Leave>',  onGenderButtonLeave)


def openFileDialog():
    global eval_rtg_file, rtg_file_name
    x = filedialog.askopenfilename(title='open')
    rtg_file_name = os.path.basename(x)

    img = Image.open(x)
    img = resize_and_crop(img, (500, 500))
    img.save(eval_rtg_file)

    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    # panel = Label(root, image=img)
    # panel.image = img
    # panel.pack()
    return img



window.mainloop()