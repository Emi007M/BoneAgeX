from tkinter import Button
import os
import webbrowser
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AtlasButtonFrame:
    def __init__(self, parent, img_dir, padx=30):
        self.parent = parent
        self.img_dir = img_dir
        self.padx = padx

        # create rounded image for the button (dark gray fill)
        self.btn_img = self.__create_rounded_button_image('Pokaż w Atlasie str. 0', width=170, height=34, radius=20, fill_color='#30393D', text_color='white', font_size=12)
        self.button = Button(self.parent, image=self.btn_img, bd=0, highlightthickness=0, relief='flat', command=self.__open_atlas, bg='#192229', activebackground='#192229', activeforeground='white')
        self.button.grid(row=4, column=0, columnspan=3, padx=self.padx, pady=20, sticky="we")
        # cursor change on hover
        self.button.bind('<Enter>', lambda e: self.button.config(cursor='hand2'))
        self.button.bind('<Leave>', lambda e: self.button.config(cursor=''))
        # start hidden
        self.button.grid_forget()
        # context for page calculation
        self.evaluations = None
        self.gender_val = None
        self.current_page = 0

    def show(self):
        try:
            self.button.grid()
        except Exception:
            pass

    def hide(self):
        try:
            self.button.grid_forget()
        except Exception:
            pass

    def __open_atlas(self):
        # use pre-computed page from context
        logger.info("__open_atlas called")
        page = self.current_page
        logger.info(f"Using pre-computed page: {page}")
        project_root = Path(__file__).resolve().parents[2]
        atlas_path = project_root.joinpath('Atlas_of_Hand_Bone_Age.pdf')
        logger.info(f"Atlas path: {atlas_path}")
        logger.info(f"Atlas path exists: {atlas_path.exists()}")
        if atlas_path.exists():
            try:
                # Create file URI with page fragment
                file_uri = atlas_path.as_uri() + f"#page={page}"
                logger.info(f"File URI with page: {file_uri}")
                
                # Try opening with browser/viewer (supports page fragment)
                if os.name == 'nt':
                    logger.info(f"Opening PDF on Windows with webbrowser: {file_uri}")
                    webbrowser.open_new(file_uri)
                    logger.info("webbrowser.open_new called successfully")
                else:
                    # Linux/Mac: use webbrowser
                    logger.info(f"Opening PDF on Unix-like with webbrowser: {file_uri}")
                    webbrowser.open_new(file_uri)
                    logger.info("webbrowser.open_new called successfully")
            except Exception as e:
                logger.error(f"Failed to open PDF with webbrowser: {e}", exc_info=True)
                # Fallback: try os.startfile without page fragment
                try:
                    if os.name == 'nt':
                        logger.info(f"Fallback: Opening PDF on Windows with os.startfile: {atlas_path}")
                        os.startfile(str(atlas_path))
                        logger.info("os.startfile called successfully")
                except Exception as e2:
                    logger.error(f"Fallback os.startfile also failed: {e2}", exc_info=True)
        else:
            logger.warning(f"Atlas PDF not found at {atlas_path}")

    def __create_rounded_button_image(self, text, width=200, height=40, radius=8, fill_color='#2E8B57', text_color='white', font_size=12):
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = 0, 0, width, height
        # Pillow >=6 supports rounded_rectangle
        try:
            draw.rounded_rectangle([(left, top), (right, bottom)], radius=radius, fill=fill_color)
        except Exception:
            # fallback: draw rounded corners manually
            draw.rectangle([(left, top), (right, bottom)], fill=fill_color)

        try:
            font = ImageFont.truetype('arial.ttf', font_size)
        except Exception:
            font = ImageFont.load_default()

        try:
            text_w, text_h = draw.textsize(text, font=font)
        except Exception:
            text_w, text_h = (len(text) * font_size // 2, font_size)
        text_x = (width - text_w) / 2
        text_y = (height - text_h) / 2 - 1
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        return ImageTk.PhotoImage(img)

    def update_context(self, evaluations, gender_val):
        self.evaluations = evaluations
        self.gender_val = gender_val
        # compute page and update button label
        self.current_page = self.__compute_page()
        logger.info(f"Context updated: evaluations={evaluations}, gender_val={gender_val}, page={self.current_page}")
        # regenerate button image with page number in label
        self.btn_img = self.__create_rounded_button_image(f'Pokaż w Atlasie str. {self.current_page}', width=170, height=34, radius=20, fill_color='#30393D', text_color='white', font_size=12)
        # keep reference to prevent garbage collection
        self.button.image = self.btn_img
        self.button.config(image=self.btn_img)

    def __compute_page(self):
        """
        Compute the atlas page based on evaluations (age in months) and gender.
        Returns the page for the closest age to the mean evaluation.
        """
        logger.info("__compute_page called")
        logger.debug(f"  evaluations: {self.evaluations}, gender_val: {self.gender_val}")
        
        # Male (gender_val == 1): age -> page
        male_pages = {
            8: 41, 10: 42, 12: 43, 14: 44, 16: 45, 18: 46, 20: 47,
            24: 48, 28: 49, 30: 50, 36: 51, 42: 52, 48: 53, 54: 54,
            60: 55, 66: 56, 72: 57, 78: 58, 96: 59, 108: 60, 120: 61,
            132: 62, 144: 63, 156: 64, 168: 65, 180: 66, 192: 67, 204: 68, 
            216: 69
        }
        # Female (gender_val == 0): age -> page
        female_pages = {
            8: 73, 10: 74, 12: 75, 14: 76, 16: 77, 18: 78, 20: 79, 28: 80,
            24: 81, 30: 82, 36: 83, 42: 84, 48: 85, 54: 86, 60: 87, 66: 88,
            72: 89, 84: 90, 96: 91, 108: 92, 120: 93, 132: 94, 144: 95, 156: 96, 
            168: 97, 180: 98, 192: 99, 204: 100, 216: 101
        }
        
        try:
            # get mean age from evaluations
            if self.evaluations is None or all(e is None for e in self.evaluations):
                logger.info("No evaluations available, returning default page 41")
                return 41 
            
            mean_age = int(sum(e for e in self.evaluations if e is not None) / 
                          len([e for e in self.evaluations if e is not None]))
            logger.info(f"Mean age computed: {mean_age} months")
            
            # select appropriate dictionary based on gender
            pages_dict = male_pages if self.gender_val == 1 else female_pages
            gender_str = "Male" if self.gender_val == 1 else "Female"
            logger.info(f"Using {gender_str} pages dictionary")
            
            # find the age closest to mean_age
            closest_age = min(pages_dict.keys(), key=lambda age: abs(age - mean_age))
            page = pages_dict[closest_age]
            logger.info(f"Closest age: {closest_age} months -> page: {page}")
            
            return page
        except Exception as e:
            logger.error(f"Error computing page: {e}", exc_info=True)
            return 41 


    def get(self):
        return self.button
