import cv2
import numpy as np
import math
from pathlib import Path
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf


IMAGE_SIZE = 500
MINAREA = 100



IMAGE_ROOT = '../data/'
FOLDER_LIST = ['classification/raw_img/dumbbell_color/color_7.5/',
               'classification/raw_img/dumbbell_color/color_10/',
               'classification/raw_img/dumbbell_color/color_12.5/',
               'classification/raw_img/dumbbell_color/color_15/',
               'classification/raw_img/dumbbell_color/color_17.5/',
               'classification/raw_img/dumbbell_color/color_20/',
               'classification/raw_img/dumbbell_color/color_22.5/',
               'classification/raw_img/dumbbell_color/color_25/']

DB_MODEL_PATH   = './util/od_DKbell.tflite'
POSE_MODEL_PATH = './util/singlepose.tflite'


# Get current positions
RED_LOWER = np.array([173,100,100])
RED_UPPER = np.array([179,255,255])

YELLOW_LOWER = np.array([20,20,20])
YELLOW_UPPER = np.array([30,255,255])

WHITE_LOWER = np.array([90,20,20])
WHITE_UPPER = np.array([110,255,255])


name_list = ['nose', 'left eye', 'right eye', 'left ear', 'right ear',    # 0-4
             'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',  # 5-10
             'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']        # 11-16
lines_human = [(0,1), (0,2), (1,3), (2,4),
               (5,6), (5,7), (5,11), (6,8), (6,12), (7,9), (8,10),
               (11,12), (11,13), (12,14), (13,15), (14,16)]



def keypoint_fit_image(keypoints, h_w=(480, 640)):
    keypoints_list = []
    for i, keypoint in enumerate(keypoints):
        _ = {'location_x' : int(keypoint[1]*h_w[1]),
             'location_y' : int(keypoint[0]*h_w[0]),
             'name' : name_list[i],
             'score' : keypoint[2] }
        keypoints_list.append(_)
    return keypoints_list


class FOLDER():
    def __init__(self, window):
        self.window = window
        self.folder_index = 0
        self.image_index = 0
        
        self.db_model_path = DB_MODEL_PATH
        self.db_output = []
        
        self.pose_model_path = POSE_MODEL_PATH
        self.pose_output =[]
        
        self.flag_ryw = {'left':[False,False,False,False],
                         'right':[False,False,False,False]}
        
        p = Path(IMAGE_ROOT + FOLDER_LIST[0]).glob('**/*.jpg')
        self.files = [x for x in p if x.is_file()]
        
        self.image_path = str(self.files[0])
        
        #title
        self.window.title("WeightMeasure")
        self.window["bg"] = "DimGray"

    def setting_image(self):
        try:
            self.focus_text.set("Focus: " + self.image_path)
        except:
            pass
        
        self.image_bgr = cv2.imread(self.image_path)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_bgr.shape[:2]
        
        #self.image = Image.open(image_path)
        #width, height = self.image.size
        
        self.tk_height = IMAGE_SIZE
        self.tk_width  = int((IMAGE_SIZE * width) / height)
        
        self.image_focus_bgr = cv2.resize(self.image_bgr, (self.tk_width, self.tk_height), interpolation=cv2.INTER_AREA)
        self.image_focus_rgb = cv2.cvtColor(self.image_focus_bgr, cv2.COLOR_BGR2RGB)
        self.image_focus_hsv = cv2.cvtColor(self.image_focus_bgr, cv2.COLOR_BGR2HSV)
        
        self.image_working_bgr = self.image_focus_bgr.copy()
        
        self.inference_pose()
        self.draw_pose()
        self.inference_db()
        self.draw_db()
        
        self.image_working_rgb = cv2.cvtColor(self.image_working_bgr, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(self.image_focus_rgb)
        self.image_focus_tk = ImageTk.PhotoImage(image=img)
        img = Image.fromarray(self.image_working_rgb)
        self.image_working_tk = ImageTk.PhotoImage(image=img)
        
        # change title
        
        
        
    def inference_pose(self):
        interpreter = tf.lite.Interpreter(model_path=POSE_MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        image_shape = self.image_focus_bgr.shape
        
        image = cv2.resize(self.image_rgb, (192, 192), interpolation=cv2.INTER_AREA)
        image_ex = np.expand_dims(image, axis = 0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], image_ex)
        interpreter.invoke()
        
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0][0]
        self.pose_output =  keypoint_fit_image(keypoints_with_scores, h_w=image_shape[:2])
        
        
    def draw_pose(self):
        point_size = 2
        point_color = (0, 255, 0)
        thickness = 2
        keypoints_fit = self.pose_output
        
        for keypoint in keypoints_fit:
            point = (keypoint['location_x'], keypoint['location_y'])
            cv2.circle(self.image_working_bgr, point, point_size, point_color, thickness)

        for line in lines_human:
            pt_0 = (keypoints_fit[line[0]]['location_x'], keypoints_fit[line[0]]['location_y'])
            pt_1 = (keypoints_fit[line[1]]['location_x'], keypoints_fit[line[1]]['location_y'])
            cv2.line(self.image_working_bgr, pt_0, pt_1, point_color, thickness)
        
    
    
    def inference_db(self):
        interpreter = tf.lite.Interpreter(model_path=DB_MODEL_PATH)

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        image_shape = self.image_focus_bgr.shape

        image = cv2.resize(self.image_rgb, (320, 320), interpolation=cv2.INTER_AREA)
        image_ex = np.expand_dims(image, axis = 0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], image_ex)
        interpreter.invoke()
        
        output_data_0 = interpreter.get_tensor(output_details[0]['index'])     # boxes
        output_data_1 = interpreter.get_tensor(output_details[1]['index'])     # classes
        output_data_2 = interpreter.get_tensor(output_details[2]['index'])     # scores
        
        self.db_output = []
        self.flag_ryw = {'left':[False,False,False,False],
                         'right':[False,False,False,False]}
        for i, bdd in enumerate(output_data_0[0]):
            if output_data_2[0][i] > 0.8: 
                x_1 = int(bdd[1]*image_shape[1])
                y_1 = int(bdd[0]*image_shape[0])
                x_2 = int(bdd[3]*image_shape[1])
                y_2 = int(bdd[2]*image_shape[0])
                
                self.db_output.append([x_1,y_1,x_2,y_2])
                
                # check flags
                clone_hsv = self.image_focus_hsv[y_1:y_2, x_1:x_2].copy()
                clone_bgr = self.image_focus_bgr[y_1:y_2, x_1:x_2].copy()
                
                flag_red = False
                mask = cv2.inRange(clone_hsv, RED_LOWER, RED_UPPER)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        if  min(contour[:,0,1]) < (clone_hsv.shape[0]/3) and max(contour[:,0,1]) > (clone_hsv.shape[0]*2/3):
                            flag_red = True
                            break
                
                flag_yellow = False
                mask = cv2.inRange(clone_hsv, YELLOW_LOWER, YELLOW_UPPER)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        if  min(contour[:,0,1]) < (clone_hsv.shape[0]/3) and max(contour[:,0,1]) > (clone_hsv.shape[0]*2/3):
                            flag_yellow = True
                            break
                            
                flag_white = False
                mask = cv2.inRange(clone_hsv, WHITE_LOWER, WHITE_UPPER)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        if  min(contour[:,0,1]) < (clone_hsv.shape[0]/4) and max(contour[:,0,1]) > (clone_hsv.shape[0]*3/4):
                            flag_white = True
                            break
                
                # check left right
                mx_1 = x_1
                my_1 = max(0, y_1-30)
                mx_2 = x_2
                my_2 = min(clone_hsv.shape[0], y_2+30)
                
                if mx_1 <= self.pose_output[9]['location_x'] <= x_2 and my_1 <= self.pose_output[9]['location_y'] <= y_2:
                    self.flag_ryw['left'] = [True, flag_red,flag_yellow,flag_white]
                elif mx_1 <= self.pose_output[10]['location_x'] <= x_2 and my_1 <= self.pose_output[10]['location_y'] <= y_2:
                    self.flag_ryw['right'] = [True, flag_red,flag_yellow,flag_white]
        
        # update title
        left_weight = 0
        if self.flag_ryw['left'][0]:
            left_weight += 7.5
            if self.flag_ryw['left'][1]:
                left_weight += 2.5
            if self.flag_ryw['left'][2]:
                left_weight += 5
            if self.flag_ryw['left'][3]:
                left_weight += 10
        
        right_weight = 0
        if self.flag_ryw['right'][0]:
            right_weight += 7.5
            if self.flag_ryw['right'][1]:
                right_weight += 2.5
            if self.flag_ryw['right'][2]:
                right_weight += 5
            if self.flag_ryw['right'][3]:
                right_weight += 10
        try:    
            self.working_text.set("Working: left: " + str(left_weight) + " , right: " + str(right_weight))
        except:
            pass
                
        
        
    def draw_db(self):
        box_color = (255, 0, 0)
        thickness = 2
        
        for bdd in self.db_output:
            cv2.rectangle(self.image_working_bgr, (bdd[0], bdd[1]), (bdd[2], bdd[3]), box_color, thickness)
        
        
    
    def update_fw_image(self):
        self.setting_image()
        self.focus.configure(image=self.image_focus_tk)
        self.working.configure(image=self.image_working_tk)
        
    
    def before_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.image_path = str(self.files[self.image_index])
            
        self.setting_image()
        self.focus.configure(image=self.image_focus_tk)
        self.working.configure(image=self.image_working_tk)
        
        
    def next_image(self):
        self.image_index += 1
        self.image_path = str(self.files[self.image_index])
        
        self.update_fw_image()
        
        
    
    def next_folder(self):
        self.folder_index = (self.folder_index + 1) % len(FOLDER_LIST)
        self.change_folder()
        
        
    def before_folder(self):
        self.folder_index = (self.folder_index - 1) % len(FOLDER_LIST)
        self.change_folder()
        
    
    def change_folder(self):
        p = Path(IMAGE_ROOT + FOLDER_LIST[self.folder_index]).glob('**/*.jpg')
        self.files = [x for x in p if x.is_file()]
        self.image_index = 0
        self.image_path = str(self.files[0])
        self.update_fw_image()
    
    
        
    def create_widgets(self):
        # images setting
        self.focus_text = StringVar()
        self.focus_text.set("Focus: " + self.image_path)
        self.focus_title = Label(self.window, textvariable=self.focus_text,fg='WhiteSmoke',bg="DimGray")
    
        self.focus = Label(self.window,
                           image=self.image_focus_tk,
                           width=self.tk_width,height=self.tk_height)
        
        self.working_text = StringVar()
        self.working_text.set("Working: ")
        self.working_title = Label(self.window, textvariable=self.working_text,fg='WhiteSmoke',bg="DimGray")
        
        self.working = Label(self.window,
                             image=self.image_working_tk,
                             width=self.tk_width,height=self.tk_height)
        
        self.focus_title.grid(row=0, column=1, padx=1, pady=1)
        self.focus.grid(row=1, column=1, rowspan=6, padx=5, pady=5)
        self.working_title.grid(row=0, column=2, padx=1, pady=1)
        self.working.grid(row=1, column=2, rowspan=6, padx=5, pady=5)
        
        
        # Button
        self.next_image_button = Button(self.window, text='next image', height=3, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.next_image)
        self.before_image_button = Button(self.window, text='before image', height=3, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.before_image)
        self.next_folder_button = Button(self.window, text='next folder', height=3, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.next_folder)
        self.before_folder_button = Button(self.window, text='before folder', height=3, width=20,fg='WhiteSmoke',bg="BurlyWood",command=self.before_folder)
        
        
        
        # Button Position
        self.next_image_button.grid(row=1, column=0)
        self.before_image_button.grid(row=2, column=0)
        self.next_folder_button.grid(row=3, column=0)
        self.before_folder_button.grid(row=4, column=0)
        
        
        
        
        
def gui_start(window, gui):
    
    gui.setting_image()
    gui.create_widgets()
    window.mainloop()
    
    
if __name__ == '__main__':
    IMAGE_ROOT = '../../data/'
    DB_MODEL_PATH   = './od_DKbell.tflite'
    POSE_MODEL_PATH = './singlepose.tflite'
    
    my_window = Tk()
    my_gui = FOLDER(my_window)

    gui_start(my_window, my_gui)

