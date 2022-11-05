from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os


class RtgChoiceFrame:

    def __init__(self, root, img_dir, bg_color, text_color_1, font_1, text_color_2, font_2):
        self.__init_imgs(img_dir)
        self.bg_color = bg_color

        self.rtg_choice_frame = Frame(root, width=500, bg=bg_color)
        self.rtg_choice_frame.grid(row=0, column=0, sticky="we")

        section_preview_label = Label(self.rtg_choice_frame, text="Zdjęcie RTG", fg=text_color_1, bg=bg_color, font=(font_1))
        section_preview_label.grid(row=0, column=0, sticky="w")

        self.search_button = Button(self.rtg_choice_frame, image=self.img_search_button, bg=bg_color, activebackground=bg_color,
                               bd=0, cursor="hand2")
        self.search_button.grid(row=1, column=0, sticky="w")

        self.file_name_label = Label(self.rtg_choice_frame, text="...", fg=text_color_2, bg=bg_color, font=(font_2))
        self.file_name_label.grid(row=1, column=1, sticky="w")

        self.preview_rtg_label = Label(self.rtg_choice_frame, image=self.img_rtg, bg=bg_color)
        self.preview_rtg_label.grid(row=2, column=0, padx=10, pady=20)

        self.search_button.bind('<Enter>', self.__onSearchButtonEnter)
        self.search_button.bind('<Button-1>', self.__onSearchButtonClick)
        self.search_button.bind('<Leave>', self.__onSearchButtonLeave)


    def get(self) -> Frame:
        return self.rtg_choice_frame

    def registerCallbacks(self, getInputImageCallback, getInputFileNameCallback):
        self.getInputImageCallback = getInputImageCallback
        self.getInputFileNameCallback = getInputFileNameCallback

    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        self.img_search_button_leave = PhotoImage(file=img_dir + "search_button.png")
        self.img_search_button_hover = PhotoImage(file=img_dir + "search_button_hover.png")
        self.img_search_button = self.img_search_button_leave

        self.img_hand_placeholder = PhotoImage(file=img_dir + "hand.png")
        self.img_rtg = self.img_hand_placeholder

    def __onSearchButtonEnter(self, event):
        self.img_search_button = self.img_search_button_hover
        self.search_button.config(image=self.img_search_button)

    def __onSearchButtonClick(self, event):
        self.img_search_button = self.img_search_button_hover
        self.search_button.config(image=self.img_search_button)

        img = self.__openFileDialog()

        img = self.__imageToSquare(img)

        self.getInputImageCallback(img)

        self.__updateRtgPreview(img.copy())

    def __onSearchButtonLeave(self, event):
        self.img_search_button = self.img_search_button_leave
        self.search_button.config(image=self.img_search_button)

    def __openFileDialog(self):
        x = filedialog.askopenfilename(title='open')
        rtg_file_name = os.path.basename(x)
        self.getInputFileNameCallback(rtg_file_name)

        return Image.open(x)

    def __imageToSquare(self, img):
        size = 500

        img_bg_color = img.getpixel((1, 1))

        img_square = Image.new("RGB", (size, size), img_bg_color)
        img_ratio = img.width / img.height

        if img_ratio > 1:
            img = img.resize((size, int(size / img_ratio)))
        else:
            img = img.resize((int(size * img_ratio), size))

        img_square.paste(img, (int((size - img.width) / 2), int((size - img.height) / 2)))

        return img_square

    def __updateRtgPreview(self, img):
        # img = img.resize_and_crop((250, 250), Image.ANTIALIAS)
        size = 250

        img = img.resize((size, size))

        #img.thumbnail((250, 250))

        self.img_rtg = ImageTk.PhotoImage(img)
        self.preview_rtg_label = Label(self.rtg_choice_frame, image=self.img_rtg, bg=self.bg_color)
        self.preview_rtg_label.grid(row=2, column=0, padx=10, pady=20)

