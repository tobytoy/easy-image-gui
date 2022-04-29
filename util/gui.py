import cv2
import canvas
from tkinter import *

class GUI():
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(GUI, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance
    
    
    def __init__(self, window):
        self.window = window
        self.canvas = canvas.Canvas()
        self.canvas.setting_pipeline()
        
    def create_widgets(self):
        
        # images setting
        self.focus_text = StringVar()
        self.focus_text.set("Focus: ")
        self.focus_title = Label(self.window, textvariable=self.focus_text,fg='WhiteSmoke',bg="DimGray")
    
        #self.focus = Label(self.window,
        #                   image=self.image_focus_tk,
        #                   width=self.tk_width,height=self.tk_height)
        
        self.working_text = StringVar()
        self.working_text.set("Working: " )
        self.working_title = Label(self.window, textvariable=self.working_text,fg='WhiteSmoke',bg="DimGray")
        
        #self.working = Label(self.window,
        #                     image=self.image_working_tk,
        #                     width=self.tk_width,height=self.tk_height)
        
        # image position
        self.focus_title.grid(row=0, column=0, padx=1, pady=1)
        #self.focus.grid(row=1, column=0, rowspan=6, padx=5, pady=5)
        self.working_title.grid(row=0, column=1, padx=1, pady=1)
        #self.working.grid(row=1, column=1, rowspan=6, padx=5, pady=5)
        
        
        
def gui_start(window, gui):
    
    #gui.setting_image()
    gui.create_widgets()
    window.mainloop()
    
    
if __name__ == '__main__':
    my_window = Tk()
    my_gui = GUI(my_window)

    gui_start(my_window, my_gui)
        
        