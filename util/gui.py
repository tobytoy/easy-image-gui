import cv2
from util import canvas
import numpy as np
from tkinter import *
from PIL import Image, ImageTk



class GUI():
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(GUI, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance
    
    
    def __init__(self, window):
        self.window = window
        self.window.title("easy-image-gui")
        self.width = 500
        self.height = 500
        self.canvas = canvas.Canvas()
        self.canvas.setting_pipeline()
        
    def setting_image(self):
        img = Image.fromarray(self.canvas.original['rgb'])
        self.image_focus_tk = ImageTk.PhotoImage(image=img)
        image = cv2.cvtColor(self.canvas.image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        self.image_working_tk = ImageTk.PhotoImage(image=img)
        
    def setting_working_image(self):
        image = cv2.cvtColor(self.canvas.image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        self.image_working_tk = ImageTk.PhotoImage(image=img)
        
        
    def before_image(self):
        self.canvas.before_image()
        self.setting_image()
        self.focus_text.set("Image Path: " + self.canvas.path['image'])
        self.focus.configure(image=self.image_focus_tk,
                             width=self.canvas.image_width,height=self.canvas.image_height)
        self.working.configure(image=self.image_working_tk,
                               width=self.canvas.image_width,height=self.canvas.image_height)
        
    def next_image(self):
        self.canvas.before_image()
        self.setting_image()
        self.focus_text.set("Image Path: " + self.canvas.path['image'])
        self.focus.configure(image=self.image_focus_tk,
                             width=self.canvas.image_width,height=self.canvas.image_height)
        self.working.configure(image=self.image_working_tk,
                               width=self.canvas.image_width,height=self.canvas.image_height)
    
    def save_image(self):
        self.canvas.save_image()
    
    
    def point_fun(self):
        self.canvas.deal_hsv2image()
        self.setting_working_image()
        self.working.configure(image=self.image_working_tk)
        
    
    def line_fun(self):
        gray = cv2.cvtColor(self.canvas.original['bgr'], cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 30, 150)
        self.canvas.image = canny
        self.setting_working_image()
        self.working.configure(image=self.image_working_tk)
        
    
    def rectangle_fun(self):
        self.canvas.deal_bdd2image()
        self.setting_working_image()
        self.working.configure(image=self.image_working_tk)
    
        
    def create_widgets(self):
        # Button
        self.before_button = Button(self.window, text='Before', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.before_image)
        self.next_button = Button(self.window, text='Next', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.next_image)
        self.save_button = Button(self.window, text='Save', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.save_image)
        
        self.point_button = Button(self.window, text='Point', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.point_fun)
        self.line_button = Button(self.window, text='Line', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.line_fun)
        self.rectangle_button = Button(self.window, text='Rctangle', height=2, width=10,fg='WhiteSmoke',bg="BurlyWood",command=self.rectangle_fun)
       
        
        # Button Position
        self.before_button.grid(row=0, column=0)
        self.next_button.grid(row=0, column=1)
        self.save_button.grid(row=0, column=2)
        
        self.point_button.grid(row=0, column=3)
        self.line_button.grid(row=0, column=4)
        self.rectangle_button.grid(row=0, column=5)
        
        
        
        # images setting
        self.focus_text = StringVar()
        self.focus_text.set("Image Path: " + self.canvas.path['image'])
        self.focus_title = Label(self.window, textvariable=self.focus_text,fg='Blue',bg="Silver")
    
        self.focus = Label(self.window,
                           image=self.image_focus_tk,
                           width=self.canvas.image_width,height=self.canvas.image_height)
        
        self.working_text = StringVar()
        self.working_text.set("Result" )
        self.working_title = Label(self.window, textvariable=self.working_text,fg='Blue',bg="Silver")
        
        self.working = Label(self.window,
                             image=self.image_working_tk,
                             width=self.canvas.image_width,height=self.canvas.image_height)
        
        # image position
        self.focus_title.grid(row=1, column=0, columnspan=3, padx=1, pady=1)
        self.focus.grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        self.working_title.grid(row=1, column=3, columnspan=3, padx=1, pady=1)
        self.working.grid(row=2, column=3, columnspan=3, padx=5, pady=5)
        
        
        
def gui_start(window, gui):
    gui.setting_image()
    gui.create_widgets()
    window.mainloop()
    
def begin():
    my_window = Tk()
    my_gui = GUI(my_window)

    gui_start(my_window, my_gui)
    
    
    
if __name__ == '__main__':
    begin()
        
        