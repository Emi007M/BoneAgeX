from tkinter import *
import numpy as np
import time
from threading import *

class PartialEvalsFrame:

    def __init__(self, root, img_dir, bg_color, text_color_1, font_1):
        self.__init_imgs(img_dir)
        self.bg_color = bg_color

        self.eval_loading_thread = None
        self.eval_loading_thread_no_stop = None

        self.partial_evals_frame = Frame(root, bg=bg_color)
        self.partial_evals_frame.grid(row=1, column=0, columnspan=3)

        doctor_0 = Label(self.partial_evals_frame, image=self.doc_imgs[0][0], bg=bg_color)
        doctor_0.grid(row=0, column=0, padx=26, pady=10)
        doctor_1 = Label(self.partial_evals_frame, image=self.doc_imgs[1][0], bg=bg_color)
        doctor_1.grid(row=0, column=1, padx=13, pady=10)
        doctor_2 = Label(self.partial_evals_frame, image=self.doc_imgs[2][0], bg=bg_color)
        doctor_2.grid(row=0, column=2, padx=26, pady=10)
        self.doctor_labels = [doctor_0, doctor_1, doctor_2]

        eval_0_label = Label(self.partial_evals_frame, text=". . .", fg=text_color_1, bg=bg_color,
                             font=(font_1))
        eval_0_label.grid(row=1, column=0)
        eval_1_label = Label(self.partial_evals_frame, text=". . .", fg=text_color_1, bg=bg_color,
                             font=(font_1))
        eval_1_label.grid(row=1, column=1)
        eval_2_label = Label(self.partial_evals_frame, text=". . .", fg=text_color_1, bg=bg_color,
                             font=(font_1))
        eval_2_label.grid(row=1, column=2)
        self.eval_labels = [eval_0_label, eval_1_label, eval_2_label]


    def get(self) -> Frame:
        return self.partial_evals_frame

    def launchPartialEvals(self, number):
        self.eval_loading_thread = Thread(target=self.__evalLoading, args=[self.eval_labels[number], number])
        self.eval_loading_thread.start()

    def updatePartialEvals(self, evaluations):
        for i in range(len(evaluations)):
            if evaluations[i] is None:
                self.doctor_labels[i].config(image=self.doc_imgs[i][0])
                self.eval_labels[i].config(text=" ")
            else:
                self.eval_loading_thread_no_stop = i
                self.doctor_labels[i].config(image=self.doc_imgs[i][1])
                self.eval_labels[i].config(text=self.__evalToTextShort(evaluations[i]))


    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        img_lek_0 = [PhotoImage(file=img_dir + "lekA_0.png"), PhotoImage(file=img_dir + "lekA.png")]
        img_lek_1 = [PhotoImage(file=img_dir + "lekB_0.png"), PhotoImage(file=img_dir + "lekB.png")]
        img_lek_2 = [PhotoImage(file=img_dir + "lekC_0.png"), PhotoImage(file=img_dir + "lekC.png")]
        self.doc_imgs = [img_lek_0, img_lek_1, img_lek_2]


    def __evalLoading(self, eval_label, eval_no, n=0):
        text = [".    ", ". .  ", ". . ."]

        while self.eval_loading_thread_no_stop is not eval_no:
            eval_label.config(text=text[n])
            self.partial_evals_frame.update()
            time.sleep(0.5)

            n = 0 if n is (len(text) - 1) else n + 1
        self.eval_loading_thread_no_stop = None


    def __evalToTextShort(self, val):
        years, months = divmod(int(val), 12)
        return f"{years}L {months}M"


