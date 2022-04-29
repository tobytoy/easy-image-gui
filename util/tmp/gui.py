import cv2
import numpy as np
import math
from pathlib import Path
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf



IMAGE_SIZE = 500
MINAREA = 100
IMAGE_ROOT = '../data'
ICON_PATH = './util/icon_large.png'
MODEL_PATH = './util/od_DKbell.tflite'



COLLECT = {
    'hsv_list':[],
    'image_list':[]
}


class GUI():
    def __init__(self, window, image_root = IMAGE_ROOT):
        self.window = window
        self.hsv_index = 0
        self.bdd_index = 0
        self.index = 0
        self.focus_bdd = False
        self.contour_number = 0
        self.model_path = MODEL_PATH
        
        p = Path(image_root).glob('**/*.jpg')
        self.files = [x for x in p if x.is_file()]
        #self.image_path = './dumbbell.jpg'
        self.image_path = str(self.files[self.index])
        
        #self.screenwidth = self.window.winfo_screenwidth()
        #self.screenheight = self.window.winfo_screenheight()
        #title
        self.window.title("HsvMaster")
        #siz=800*600,position=(500,200)
        #self.window.geometry('1000x600+500+200')
        self.window["bg"] = "DimGray"
        # icon
        self.icon = ImageTk.PhotoImage(file=ICON_PATH)
        self.window.call('wm','iconphoto',self.window._w,self.icon)

        
    def focus_bdd_image(self):
        self.focus_bdd = True if (self.focus_bdd == False) else False
        self.working_text.set("Working: " + str(self.focus_bdd) + ", h_i: " + str(self.hsv_index) + ", b_i: "+ str(self.bdd_index) )
        self.change_working_image()

        
    def check_h_min(self, val):
        val = int(val)
        if val > self.hmax_scale.get():
            self.hmax_scale.set(val)
        self.change_working_image()
            
    def check_h_max(self, val):
        val = int(val)
        if self.hmin_scale.get() > val:
            self.hmin_scale.set(val)
        self.change_working_image()
            
    def check_s_min(self, val):
        val = int(val)
        if val > self.smax_scale.get():
            self.smax_scale.set(val)
        self.change_working_image()
            
    def check_s_max(self, val):
        val = int(val)
        if self.smin_scale.get() > val:
            self.smin_scale.set(val)
        self.change_working_image()
            
    def check_v_min(self, val):
        val = int(val)
        if val > self.vmax_scale.get():
            self.vmax_scale.set(val)
        self.change_working_image()
            
    def check_v_max(self, val):
        val = int(val)
        if self.vmin_scale.get() > val:
            self.vmin_scale.set(val)
        self.change_working_image()
    
    def setting_image(self):
        try:
            self.focus_text.set("Focus: " + self.image_path)
        except:
            pass
        
        self.image_bgr = cv2.imread(self.image_path)
        height, width = self.image_bgr.shape[:2]
        
        #self.image = Image.open(image_path)
        #width, height = self.image.size
        
        self.tk_height = IMAGE_SIZE
        self.tk_width  = int((IMAGE_SIZE * width) / height)
        
        self.image_focus_bgr = cv2.resize(self.image_bgr, (self.tk_width, self.tk_height), interpolation=cv2.INTER_AREA)
        
        self.image_focus_hsv = cv2.cvtColor(self.image_focus_bgr, cv2.COLOR_BGR2HSV)
        self.image_focus_rgb = cv2.cvtColor(self.image_focus_bgr, cv2.COLOR_BGR2RGB)
        self.image_working_rgb = cv2.cvtColor(self.image_focus_bgr, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(self.image_focus_rgb)
        self.image_focus_tk = ImageTk.PhotoImage(image=img)
        img = Image.fromarray(self.image_working_rgb)
        self.image_working_tk = ImageTk.PhotoImage(image=img)
        
        #self.image_bgr = cv2.imread()
        #self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        #self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        #img = Image.fromarray(self.image_rgb)
        #self.image_tk = ImageTk.PhotoImage(image=img) 
        
        #cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        #self.image.resize((tk_width, tk_height), Image.ANTIALIAS)
        #self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        
    def reflash_image(self):
        self.setting_image()
        self.focus.configure(image=self.image_focus_tk)
        self.working.configure(image=self.image_working_tk)
        self.change_working_image()
        
    def draw_contours(self):
        # Get current positions of all trackbars
        hMin = self.hmin_scale.get()
        sMin = self.smin_scale.get()
        vMin = self.vmin_scale.get()
        hMax = self.hmax_scale.get()
        sMax = self.smax_scale.get()
        vMax = self.vmax_scale.get()

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        try:
            if self.focus_bdd == False:
                mask = cv2.inRange(self.image_focus_hsv, lower, upper)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contours_filter_area = []

                for contour in contours:
                    if cv2.contourArea(contour) > MINAREA:
                        contours_filter_area.append(contour)
                self.contour_number = len(contours_filter_area)
                
                clone = self.image_focus_bgr.copy()
                cv2.drawContours(clone, contours_filter_area, -1, (0, 255, 0), 2)
                
                clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(clone)
                self.image_focus_tk = ImageTk.PhotoImage(image=img)
                self.focus.configure(image=self.image_focus_tk)
            
            elif COLLECT['image_list'] != []:        
                (x, y, w, h) = COLLECT['image_list'][self.bdd_index]
                
                image_focus_hsv = self.image_focus_hsv[y : y+h, x:x+w]
                image_focus_bgr = self.image_focus_bgr[y : y+h, x:x+w]
                
                mask = cv2.inRange(image_focus_hsv, lower, upper)
                
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contours_filter_area = []

                for contour in contours:
                    if cv2.contourArea(contour) > MINAREA:
                        contours_filter_area.append(contour)
                self.contour_number = len(contours_filter_area)
                
                clone = image_focus_bgr.copy()
                cv2.drawContours(clone, contours_filter_area, -1, (0, 255, 0), 2)
                
                clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(clone)
                self.image_focus_tk = ImageTk.PhotoImage(image=img)
                self.focus.configure(image=self.image_focus_tk)
                
        except:
            pass
        
        
        
        
    def before_image(self):
        if self.index > 0:
            self.index -= 1
            self.image_path = str(self.files[self.index])
            
        self.setting_image()
        self.focus.configure(image=self.image_focus_tk)
        self.working.configure(image=self.image_working_tk)
        self.change_working_image()
        
    def next_image(self):
        self.index += 1
        self.image_path = str(self.files[self.index])
        
        self.setting_image()
        self.focus.configure(image=self.image_focus_tk)
        self.working.configure(image=self.image_working_tk)
        self.change_working_image()
        
        
    def select_hsv(self):
        # Create a window
        cv2.namedWindow('select_hsv', 0)
        cv2.imshow('select_hsv', self.image_focus_bgr)
        
        cv2.setMouseCallback("select_hsv", self.getposHsv)
        
    
    def select_bdd(self):
        # Create a window
        cv2.namedWindow('select_bdd', 0)
        #cv2.imshow('select_bdd', self.image_focus_bgr)
        
        rect = cv2.selectROI("select_bdd", self.image_focus_bgr, showCrosshair=True, fromCenter=False)
        
        if len(COLLECT['image_list']) > 3:
            COLLECT['image_list'].pop(0)
        COLLECT['image_list'].append(rect)
        
        _ = ""
        for item in COLLECT['image_list']:
            _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+','+str(item[3])+"] ")
        self.bdd_text.set(_)
        
        #cv2.destroyAllWindows()
        cv2.destroyWindow('select_bdd')
    
    
    def deep_model(self):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        image_shape = self.image_focus_bgr.shape
        
        image = cv2.resize(self.image_focus_bgr, (320, 320), interpolation=cv2.INTER_AREA)
        image_ex = np.expand_dims(image, axis = 0).astype(np.uint8)
        
        
        interpreter.set_tensor(input_details[0]['index'], image_ex)
        interpreter.invoke()
        
        boxes = interpreter.get_tensor(output_details[0]['index'])     
        classes = interpreter.get_tensor(output_details[1]['index']) 
        scores = interpreter.get_tensor(output_details[2]['index'])
        
        for i, bdd in enumerate(boxes[0]):
            if scores[0][i] > 0.8:    
                x_y = (int(bdd[1]*image_shape[1]),int(bdd[0]*image_shape[0]))
                xw_yh = (int(bdd[3]*image_shape[1]), int(bdd[2]*image_shape[0]))
                
                COLLECT['image_list'].append([x_y[0], x_y[1], xw_yh[0]-x_y[0], xw_yh[1]-x_y[1]])
                
                print(x_y, xw_yh)

        _ = ""
        for item in COLLECT['image_list']:
            _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+','+str(item[3])+"] ")
        self.bdd_text.set(_)
                
        
        
    def getposHsv(self, event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            _hsv = self.image_focus_hsv[y, x]
            
            if len(COLLECT['hsv_list']) > 3:
                COLLECT['hsv_list'].pop(0)
            COLLECT['hsv_list'].append(_hsv)
            
            _ = ""
            for item in COLLECT['hsv_list']:
                _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+"] ")
            self.hsv_text.set(_)
            
            
            self.hmin_scale.set( max(_hsv[0] - 5, 0)     )
            self.hmax_scale.set( min(_hsv[0] + 5, 179)   )
            
            self.smin_scale.set( max(_hsv[1] - 100, 0)   )
            self.smax_scale.set( min(_hsv[1] + 100, 255) )
            
            self.vmin_scale.set( max(_hsv[2] - 100, 0)   )
            self.vmax_scale.set( min(_hsv[2] + 100, 255) )
            
            #cv2.destroyAllWindows()
            cv2.destroyWindow('select_hsv')
            self.change_working_image()
            
        
    def change_hsv_history(self):
        
        index = self.hsv_index
        try:
            _hsv = COLLECT['hsv_list'][index]
        
            self.hmin_scale.set( max(_hsv[0] - 5, 0)     )
            self.hmax_scale.set( min(_hsv[0] + 5, 179)   )
            
            self.smin_scale.set( max(_hsv[1] - 100, 0)   )
            self.smax_scale.set( min(_hsv[1] + 100, 255) )
            
            self.vmin_scale.set( max(_hsv[2] - 100, 0)   )
            self.vmax_scale.set( min(_hsv[2] + 100, 255) )
            
            self.change_working_image()
            
            
        except:
            pass
        
        if len(COLLECT['hsv_list']) == 0:
            self.hsv_index = 0
        else:
            self.hsv_index = (self.hsv_index + 1) % len(COLLECT['hsv_list'])
        
        
    def delete_hsv_history(self):
        COLLECT['hsv_list'].pop(0)
               
        _ = ""
        for item in COLLECT['hsv_list']:
            _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+"] ")
        self.hsv_text.set(_)
        
        
    def change_bdd_history(self):
        
        index = self.bdd_index
        try:
            #_ = ""
            #for item in COLLECT['image_list']:
            #    _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+','+str(item[3])+"] ")
            #self.bdd_text.set(_)
            self.change_working_image()
            
        except:
            pass
        
        if len(COLLECT['image_list']) == 0:
            self.bdd_index = 0
        else:
            self.bdd_index = (self.bdd_index + 1) % len(COLLECT['image_list'])
        
        
    def delete_bdd_history(self):
        COLLECT['image_list'].pop(0)
               
        _ = ""
        for item in COLLECT['image_list']:
            _ += ("["+str(item[0])+","+str(item[1])+","+str(item[2])+','+str(item[3])+"] ")
        self.bdd_text.set(_)
        
        
    
    def change_size(self, val):
        height, width = self.image_bgr.shape[:2]
        
        self.working_height = int(IMAGE_SIZE * int(val) / 10)
        self.working_width  = int((IMAGE_SIZE * width) / height * int(val) / 10)
        
        self.image_working_rgb = cv2.resize(self.image_focus_rgb, (self.working_width, self.working_height), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(self.image_working_rgb)
        self.image_working_tk = ImageTk.PhotoImage(image=img)
        
        self.working.configure(image=self.image_working_tk, width=self.working_width, height=self.working_height)
        self.change_working_image()
        
        
        
    def change_working_image(self):
        # working title
        self.working_text.set("Working: " + str(self.focus_bdd) + ", h_i: " + str(self.hsv_index) + ", b_i: "+ str(self.bdd_index) )
        
        # Get current positions of all trackbars
        hMin = self.hmin_scale.get()
        sMin = self.smin_scale.get()
        vMin = self.vmin_scale.get()
        hMax = self.hmax_scale.get()
        sMax = self.smax_scale.get()
        vMax = self.vmax_scale.get()

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        try:
            # change HSV working
            # Convert to HSV format and color threshold
            #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if self.focus_bdd == False:
                mask = cv2.inRange(self.image_focus_hsv, lower, upper)
                result = cv2.bitwise_and(self.image_focus_bgr, self.image_focus_bgr, mask=mask)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
                # resize
                self.image_working_rgb = cv2.resize(result, (self.working_width, self.working_height), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(self.image_working_rgb)
                self.image_working_tk = ImageTk.PhotoImage(image=img)
        
                self.working.configure(image=self.image_working_tk, width=self.working_width, height=self.working_height)
            elif COLLECT['image_list'] != []:        
                (x, y, w, h) = COLLECT['image_list'][self.bdd_index]                
                image_focus_hsv = self.image_focus_hsv[y : y+h, x:x+w]
                image_focus_bgr = self.image_focus_bgr[y : y+h, x:x+w]
                
                mask = cv2.inRange(image_focus_hsv, lower, upper)
                result = cv2.bitwise_and(image_focus_bgr, image_focus_bgr, mask=mask)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                # resize
                self.image_working_rgb = cv2.resize(result, (self.working_width, self.working_height), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(self.image_working_rgb)
                self.image_working_tk = ImageTk.PhotoImage(image=img)
        
                self.working.configure(image=self.image_working_tk, width=self.working_width, height=self.working_height)
                
            
        except:
            pass
  
    
    def estimate(self):
        self.info_text.set('The color estimate: ' +  str(math.ceil(self.contour_number/2)) )
        
    
    
    
    def create_widgets(self):
        #scale
        self.image_size_label = Label(self.window,text='Image Size',fg='WhiteSmoke',bg="DimGray")
        self.image_size_scale = Scale(self.window,orient=HORIZONTAL,from_=1,to=20,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.change_size)
        
        self.hmin_label = Label(self.window,text='色彩 H min',fg='WhiteSmoke',bg="DimGray")
        self.hmin_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=179,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_h_min)
        self.hmax_label = Label(self.window,text='色彩 H max',fg='WhiteSmoke',bg="DimGray")
        self.hmax_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=179,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_h_max)
        self.smin_label = Label(self.window,text='飽和度 S min',fg='WhiteSmoke',bg="DimGray")
        self.smin_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=255,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_s_min)
        self.smax_label = Label(self.window,text='飽和度 S max',fg='WhiteSmoke',bg="DimGray")
        self.smax_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=255,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_s_max)
        self.vmin_label = Label(self.window,text='亮度 V min',fg='WhiteSmoke',bg="DimGray")
        self.vmin_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=255,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_v_min)
        self.vmax_label = Label(self.window,text='亮度 V max',fg='WhiteSmoke',bg="DimGray")
        self.vmax_scale = Scale(self.window,orient=HORIZONTAL,from_=0,to=255,resolution=1,tickinterval=50,length=300,width=7,fg='WhiteSmoke',bg="DimGray",command=self.check_v_max)
        
        # scale position
        self.image_size_label.grid(row=0, column=0,padx=15)
        self.image_size_scale.grid(row=0,column=1,pady=2)
        
        self.hmin_label.grid(row=1, column=0)
        self.hmin_scale.grid(row=1,column=1,pady=2)
        self.hmax_label.grid(row=2, column=0)
        self.hmax_scale.grid(row=2,column=1,pady=2)
        self.smin_label.grid(row=3, column=0)
        self.smin_scale.grid(row=3,column=1,pady=2)
        self.smax_label.grid(row=4, column=0)
        self.smax_scale.grid(row=4,column=1,pady=2)
        self.vmin_label.grid(row=5, column=0)
        self.vmin_scale.grid(row=5,column=1,pady=2)
        self.vmax_label.grid(row=6, column=0)
        self.vmax_scale.grid(row=6,column=1,pady=2)
        
        
        # Set default value for Max HSV trackbars
        self.image_size_scale.set(10)
        self.hmax_scale.set(179)
        self.smax_scale.set(255)
        self.vmax_scale.set(255)
        
        
        # images setting
        self.focus_text = StringVar()
        self.focus_text.set("Focus: " + self.image_path)
        self.focus_title = Label(self.window, textvariable=self.focus_text,fg='WhiteSmoke',bg="DimGray")
    
        self.focus = Label(self.window,
                           image=self.image_focus_tk,
                           width=self.tk_width,height=self.tk_height)
        
        self.working_text = StringVar()
        self.working_text.set("Working: " + str(self.focus_bdd) + ", h_i: " + str(self.hsv_index) + ", b_i: "+ str(self.bdd_index) )
        self.working_title = Label(self.window, textvariable=self.working_text,fg='WhiteSmoke',bg="DimGray")
        
        self.working = Label(self.window,
                             image=self.image_working_tk,
                             width=self.tk_width,height=self.tk_height)
        #img position
        #self.orign.place(x=320, y=30)
        #self.work.place(x=650,y=30)
        self.focus_title.grid(row=0, column=2, padx=1, pady=1)
        self.focus.grid(row=1, column=2, rowspan=6, padx=5, pady=5)
        self.working_title.grid(row=0, column=3, padx=1, pady=1)
        self.working.grid(row=1, column=3, rowspan=6, padx=5, pady=5)
        
        
        self.info_text = StringVar()
        self.info_text.set("")
        self.info_label = Label(self.window, textvariable=self.info_text,fg='Tomato',bg="DimGray")
        
        
        # Button
        self.import_button = Button(self.window, text='import image', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.reflash_image)
        self.focus_bdd_button = Button(self.window, text='focus bdd image', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.focus_bdd_image)
        self.select_hsv_button = Button(self.window, text='point hsv', height=2, width=20,fg='Blue',bg="BurlyWood",command=self.select_hsv)
        self.before_image_button = Button(self.window, text='before image', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.before_image)
        self.next_image_button = Button(self.window, text='next image', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.next_image)
        self.select_bdd_button = Button(self.window, text='bounding box', height=2, width=20,fg='Blue',bg="BurlyWood",command=self.select_bdd)
        self.deep_model_button = Button(self.window, text='model inference', height=2, width=20,fg='Blue',bg="BurlyWood",command=self.deep_model)
        self.draw_contours_button = Button(self.window, text='draw contours', height=2, width=20,fg='Lime',bg="BurlyWood",command=self.draw_contours)
        self.estimate_button = Button(self.window, text='estimate', height=2, width=20,fg='Tomato',bg="BurlyWood",command=self.estimate)
        
        
        # Button Position
        self.import_button.grid(row=7, column=0)
        self.focus_bdd_button.grid(row=7, column=1)
        self.before_image_button.grid(row=7, column=2)
        self.next_image_button.grid(row=7, column=3)
        self.draw_contours_button.grid(row=8, column=0)
        self.deep_model_button.grid(row=8, column=1)
        self.estimate_button.grid(row=8, column=2)
        self.info_label.grid(row=8, column=3)
        
        self.select_hsv_button.grid(row=9, column=0)
        self.select_bdd_button.grid(row=9, column=1)
        
        
        # collect date
        self.change_hsv_history_button = Button(self.window, text='change hsv', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.change_hsv_history)
        self.change_hsv_history_button.grid(row=10, column=0)
        
        self.change_hsv_history_button = Button(self.window, text='delete hsv', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.delete_hsv_history)
        self.change_hsv_history_button.grid(row=10, column=1)
        
        self.change_bdd_history_button = Button(self.window, text='change bdd', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.change_bdd_history)
        self.change_bdd_history_button.grid(row=11, column=0)
        
        self.change_bdd_history_button = Button(self.window, text='delete bdd', height=2, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.delete_bdd_history)
        self.change_bdd_history_button.grid(row=11, column=1)
        
        
        self.hsv_text = StringVar()
        self.hsv_text.set("")
        self.hsv_label = Label(self.window, textvariable=self.hsv_text,fg='WhiteSmoke',bg="DimGray")
        self.hsv_label.grid(row=10, column=2)
        
        self.bdd_text = StringVar()
        self.bdd_text.set("")
        self.bdd_label = Label(self.window, textvariable=self.bdd_text,fg='WhiteSmoke',bg="DimGray")
        self.bdd_label.grid(row=11, column=2)
        
        
        
    
def gui_start(window, gui):
    
    gui.setting_image()
    gui.create_widgets()
    window.mainloop()
    
    
if __name__ == '__main__':
    IMAGE_ROOT = r'../../data'
    ICON_PATH = './icon_large.png'
    MODEL_PATH = './od_DKbell.tflite'

    my_window = Tk()
    my_gui = GUI(my_window, IMAGE_ROOT)

    gui_start(my_window, my_gui)
    
    