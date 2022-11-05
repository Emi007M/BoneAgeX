from tkinter import *

class CenterFrame:

    def __init__(self, root, img_dir, bg_color):
        self.__init_imgs(img_dir)
        self.bg_color = bg_color

        self.center_frame = Frame(root, width=50, bg=bg_color)
        self.center_frame.grid(row=0, column=0, sticky="ns")
        self.center_frame.grid_rowconfigure(0, weight=3)
        self.center_frame.grid_rowconfigure(2, weight=3)

        center_line_n = Label(self.center_frame, image=self.img_center_line_n, bg=bg_color)
        center_line_n.grid(row=0, column=0, sticky="s")
        center_line_s = Label(self.center_frame, image=self.img_center_line_s, bg=bg_color)
        center_line_s.grid(row=2, column=0, sticky="n")

        self.eval_button = Button(self.center_frame, image=self.img_eval_button, bg=bg_color, activebackground=bg_color,
                             bd=0, cursor="hand2")
        self.eval_button.grid(row=1, column=0, sticky="ns")

        self.eval_button.bind('<Enter>', self.__onEvalButtonEnter)
        self.eval_button.bind('<Button-1>', self.__onEvalButtonClick)
        self.eval_button.bind('<Leave>', self.__onEvalButtonLeave)


    def get(self) -> Frame:
        return self.center_frame

    def registerCallbacks(self, launchEvaluationCallback):
        self.launchEvaluationCallback = launchEvaluationCallback

    def updateEvalButtonState(self, gender_val, rtg_val):
        if gender_val is None or rtg_val is None:
            self.eval_button["state"] = "disabled"
            self.eval_button.config(cursor="arrow")
        else:
            self.eval_button["state"] = "normal"
            self.eval_button.config(cursor="hand2")

    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        self.img_center_line_n = PhotoImage(file=img_dir + "center_line_n.png")
        self.img_center_line_s = PhotoImage(file=img_dir + "center_line_s.png")
        self.img_eval_button_leave = PhotoImage(file=img_dir + "eval_button.png")
        self.img_eval_button_hover = PhotoImage(file=img_dir + "eval_button_hover.png")
        self.img_eval_button = self.img_eval_button_leave

    def __onEvalButtonEnter(self, event):
        if self.eval_button["state"] == "disabled":
            return

        self.img_eval_button = self.img_eval_button_hover
        self.eval_button.config(image=self.img_eval_button)

    def __onEvalButtonClick(self, event):
        global img_eval_button, evaluations

        if self.eval_button["state"] == "disabled":
            return

        self.img_eval_button = self.img_eval_button_hover
        self.eval_button.config(image=self.img_eval_button)

        self.launchEvaluationCallback()


    def __onEvalButtonLeave(self, event):
        self.img_eval_button = self.img_eval_button_leave
        self.eval_button.config(image=self.img_eval_button)

