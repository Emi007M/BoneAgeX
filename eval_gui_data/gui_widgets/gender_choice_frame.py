from tkinter import *

class GenderChoiceFrame:

    def __init__(self, root, img_dir, bg_color, text_color_1, font_1):
        self.__init_imgs(img_dir)
        self.gender_val = None

        self.gender_choice_frame = Frame(root, width=500, bg=bg_color)
        self.gender_choice_frame.grid(row=1, column=0, sticky="we")

        self.gender_choice_frame.grid_columnconfigure(0, weight=1)

        section_preview_label = Label(
            self.gender_choice_frame, text="Płeć", fg=text_color_1, bg=bg_color, font=(font_1))
        section_preview_label.grid(row=0, column=0, sticky="w")

        self.gender_switch_button = Button(
            self.gender_choice_frame, image=self.img_gender_switch_0, bg=bg_color, activebackground=bg_color, bd=0, cursor="hand2")
        self.gender_switch_button.grid(row=1, column=0)

        self.gender_switch_button.bind('<Enter>', self.__onGenderButtonEnter)
        self.gender_switch_button.bind('<Button-1>', self.__onGenderButtonClick)
        self.gender_switch_button.bind('<Leave>', self.__onGenderButtonLeave)


    def get(self) -> Frame:
        return self.gender_choice_frame

    def registerCallbacks(self, updateGenderValCallback):
        self.updateGenderValCallback = updateGenderValCallback

    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        self.img_gender_switch_0 = PhotoImage(file=img_dir + "gender_0.png")
        self.img_gender_switch_0_hover = PhotoImage(file=img_dir + "gender_0_hover.png")
        self.img_gender_switch_F = PhotoImage(file=img_dir + "gender_F.png")
        self.img_gender_switch_M = PhotoImage(file=img_dir + "gender_M.png")
        self.img_gender_button = self.img_gender_switch_0

    def __onGenderButtonEnter(self, event):
        if self.gender_val == None:
            self.img_gender_button = self.img_gender_switch_0_hover

        self.gender_switch_button.config(image=self.img_gender_button)

    def __onGenderButtonClick(self, event):
        if self.__isOnMaleClick(event) and self.gender_val != 1:
            self.__setGenderVal(1)
            self.img_gender_button = self.img_gender_switch_M
        elif self.__isOnMaleClick(event) == False and self.gender_val != 0:
            self.__setGenderVal(0)
            self.img_gender_button = self.img_gender_switch_F
        else:
            self.__setGenderVal(None)
            self.img_gender_button = self.img_gender_switch_0_hover

        self.gender_switch_button.config(image=self.img_gender_button)

    def __onGenderButtonLeave(self, event):
        if self.gender_val == None:
            self.img_gender_button = self.img_gender_switch_0

            self.gender_switch_button.config(image=self.img_gender_button)

    def __isOnMaleClick(self, event):
        male_female_boundry_pos = 60
        return event.x > male_female_boundry_pos

    def __setGenderVal(self, val):
        self.gender_val = val
        self.updateGenderValCallback(val)

