from tkinter import *
from eval_gui_data.gui_widgets.final_eval_frame import FinalEvalFrame
from eval_gui_data.gui_widgets.partial_evals_frame import PartialEvalsFrame
from eval_gui_data.gui_widgets.atlas_button_frame import AtlasButtonFrame

class EvalFrame:

    def __init__(self, root, img_dir, bg_color, text_color_1, font_1, font_partial_eval, bg_color_final_eval, font_eval_y, font_eval_m):
        self.__init_imgs(img_dir)
        self.bg_color = bg_color

        self.eval_frame = Frame(root, bg=bg_color)
        self.eval_frame.grid(row=0, column=0, sticky="we")

        section_eval_label = Label(self.eval_frame, text="Ocena wieku", fg=text_color_1, bg=bg_color,
                                   font=(font_1))
        section_eval_label.grid(row=0, column=0, sticky="w")

        self.partial_evals_frame_widget = PartialEvalsFrame(
            self.eval_frame, img_dir, bg_color, text_color_1, font_partial_eval)

        ladder_label = Label(self.eval_frame, image=self.img_eval_ladder, bg=bg_color)
        ladder_label.grid(row=2, column=0, columnspan=3, padx=10, pady=20)

        self.final_eval_frame_widget = FinalEvalFrame(
            self.eval_frame, img_dir, bg_color_final_eval, text_color_1, font_eval_y, font_eval_m)

        self.atlas_button_widget = AtlasButtonFrame(self.eval_frame, img_dir, padx=30)

        self.hide_eval_frame = Frame(self.eval_frame, bg=bg_color)
        self.__hideEval()

    def get(self) -> Frame:
        return self.eval_frame

    def launchPartialEvals(self, number):
        self.partial_evals_frame_widget.launchPartialEvals(number)

    def updateEvals(self, evaluations, gender_val):
        self.__updatePartialEvals(evaluations)
        self.__updateFinalEval(evaluations, gender_val)

        if all(elem is not None for elem in evaluations):
            try:
                self.atlas_button_widget.update_context(evaluations, gender_val)
            except Exception:
                pass
            self.__showEval()
        else:
            self.__hideEval()


    def __updatePartialEvals(self, evaluations):
        self.partial_evals_frame_widget.updatePartialEvals(evaluations)

    def __updateFinalEval(self, evaluations, gender_val):
        self.final_eval_frame_widget.updateFinalEval(evaluations, gender_val)

    def __showEval(self):
        self.hide_eval_frame.grid_forget()
        self.atlas_button_widget.show()

    def __hideEval(self):
        self.hide_eval_frame.grid(row=2, column=0, columnspan=3, rowspan=2, sticky="nswe")
        self.atlas_button_widget.hide()

    def __init_imgs(self, img_dir):
        self.img_dir = img_dir

        self.img_eval_ladder = PhotoImage(file=img_dir + "ladder.png")
        








