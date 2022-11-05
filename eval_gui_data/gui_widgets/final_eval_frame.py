from tkinter import *
import numpy as np

class FinalEvalFrame:

    def __init__(self, root, img_dir, bg_color, text_color_1, font_y, font_m):
        self.__init_imgs(img_dir)
        self.bg_color = bg_color

        self.final_eval_frame = Frame(root, bg=bg_color)
        self.final_eval_frame.grid(row=3, column=0, columnspan=3, sticky="we", padx=30)
        self.final_eval_frame.grid_columnconfigure(0, weight=1)
        self.final_eval_frame.grid_columnconfigure(1, weight=1)

        self.final_eval_icon = Label(self.final_eval_frame, image=self.img_eval_placeholder, bg=bg_color)
        self.final_eval_icon.grid(row=0, column=0, rowspan=2, padx=10)

        self.final_eval_year_label = Label(self.final_eval_frame, text="x lat", fg=text_color_1, bg=bg_color,
                                      font=(font_y))
        self.final_eval_year_label.grid(row=0, column=1, sticky="w")
        self.final_eval_month_label = Label(self.final_eval_frame, text="x miesięcy", fg=text_color_1,
                                       bg=bg_color, font=(font_m))
        self.final_eval_month_label.grid(row=1, column=1, sticky="w")

    def get(self) -> Frame:
        return self.final_eval_frame

    def updateFinalEval(self, evaluations, gender_val):

        if all(elem is not None for elem in evaluations):
            mean_eval = int(np.mean(evaluations))

            self.final_eval_icon.config(image=self.__getEvalImgByValAndGender(mean_eval, gender_val))

            self.__updateEvalText(mean_eval)

        else:
            self.final_eval_icon.config(image=self.img_eval_placeholder)
            self.__clearEvalText()

    def __clearEvalText(self):
        self.final_eval_year_label.config(text=f" ")
        self.final_eval_month_label.config(text=f"wiek kostny")

    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        self.img_eval_placeholder = PhotoImage(file=img_dir + "gender_icon_placeholder.png")
        img_eval_F1 = PhotoImage(file=img_dir + "F1.png")
        img_eval_F2 = PhotoImage(file=img_dir + "F2.png")
        img_eval_F3 = PhotoImage(file=img_dir + "F3.png")
        img_eval_M1 = PhotoImage(file=img_dir + "M1.png")
        img_eval_M2 = PhotoImage(file=img_dir + "M2.png")
        img_eval_M3 = PhotoImage(file=img_dir + "M3.png")
        self.eval_imgs = [[img_eval_F1, img_eval_F2, img_eval_F3], [img_eval_M1, img_eval_M2, img_eval_M3]]

    def __getEvalImgByValAndGender(self, months, gender):
        if months < 5 * 12:
            return self.eval_imgs[gender][0]
        elif months > 15 * 12:
            return self.eval_imgs[gender][2]
        else:
            return self.eval_imgs[gender][1]

    def __updateEvalText(self, mean_eval):
        years, months = divmod(int(mean_eval), 12)

        self.final_eval_year_label.config(text=f"{years} {self.__getYearText(years)}")
        self.final_eval_month_label.config(text=f"{months} {self.__getMonthText(months)}")

    def __getYearText(self, years):
        if years is 1:
            return "rok"
        elif 1 < years < 5:
            return "lata"
        else:
            return "lat"

    def __getMonthText(self, months):
        if months is 1:
            return "miesiąc"
        elif 1 < months < 5:
            return "miesiące"
        else:
            return "miesięcy"
