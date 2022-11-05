from tkinter import *

class TopFrame:

    def __init__(self, window, bg_color, text_color_1, font_1, text_color2, font_2):

        self.top_frame = Frame(window, width=800, height=20, padx=10, pady=5, bg=bg_color)
        self.top_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        title_label = Label(self.top_frame, text="Bone Age X", fg=text_color_1, bg=bg_color, font=(font_1))
        title_label.grid(row=1, column=0, sticky="w")
        title_sublabel = Label(self.top_frame, text="ocena wieku kostnego", fg=text_color2, bg=bg_color, font=(font_2))
        title_sublabel.grid(row=0, column=0, sticky="w")

    def get(self) -> Frame:
        return self.top_frame