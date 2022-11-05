from tkinter import *

class BottomFrame:

    def __init__(self, root, bg_color, text_color_1, font_1):

        self.bottom_frame = Frame(root, width=800, height=20, padx=10, pady=5, bg=bg_color)
        self.bottom_frame.grid(row=0, column=0, columnspan=3, sticky="ew")

        self.logs_label = Label(self.bottom_frame, text="xxx", fg=text_color_1, bg=bg_color, font=(font_1))
        self.logs_label.grid(row=0, column=0, sticky="w")

    def get(self) -> Frame:
        return self.top_frame

    def updateLogs(self, rtg_file_name, gender_val, evaluations):
        self.logs_label.config(
            text="Dokładność wyniku: ±14M dla 90% ocen, ±6M dla 60% ocen")#  | " +
            # " RTG: " + str(rtg_file_name) + " + Płeć: " + str(gender_val) +
            # " => Wiek kostny: " + str(evaluations))

