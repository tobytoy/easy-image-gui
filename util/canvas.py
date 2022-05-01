import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging



class Canvas():
    def __init__(self, canvas_wh = (10,10), image_wh = (500,500)):
        logging.info('Canvas be created.')
        self.path = {'folder' : None,
                     'files' : None,
                     'image' : None,
                     'image_index' : 0}
        
        self.original = {'bgr' : None,
                         'rgb' : None,
                         'hsv' : None}
        
        # Get current positions of all trackbars
        self.hMin = 0
        self.sMin = 0
        self.vMin = 0
        self.hMax = 179
        self.sMax = 255
        self.vMax = 255

        # Set minimum and maximum HSV values to display
        self.lower = np.array([self.hMin, self.sMin, self.vMin])
        self.upper = np.array([self.hMax, self.sMax, self.vMax])
        
        
        self.image = None

        # canvas width, height use to set plt fig size 
        self.canvas_width, self.canvas_height = map(int,canvas_wh)
        # image and background have same width and height 
        self.image_width, self.image_height = map(int,image_wh)
        
        
    def __repr__(self):
        _ = f"Path: {self.path['image']} w,h: [{self.original['width']},{self.original['height']}] iw,ih:[{self.image_width},{self.image_height}]"
        return _
    # for print
    def __str__(self):
        return f"Path: {self.path['image']}"
    
    def setting_pipeline(self, folder_path = './datas/images/example/'):
        logging.info('Canvas: setting pipeline.')
        self.setting_folder_path(folder_path = folder_path)
        self.setting_image_path()
        self.setting_original()
        self.original2image()
    
    
    def setting_folder_path(self, folder_path = './datas/images/example/'):
        logging.info('Canvas: setting folder path.')
        path_dir = Path(folder_path)
        if path_dir.is_dir():
            path_files = sorted(list(path_dir.glob('**/*.jpg')) + list(path_dir.glob('**/*.jpeg')) + list(path_dir.glob('**/*.png')))
            self.path['files'] = path_files
            return True
        else:
            logging.warning('The folder path shold be wrong.')
            return False
        
    
    
    def setting_image_path(self, image_path = None, method = 'index'):
        logging.info('Canvas: setting image path.')
        if method == 'path':
            if image_path:
                self.path['image'] = image_path
                return True
            else:
                logging.warning('You forgot input image_path.')
                return False
        elif method == 'index':
            if self.path['files']:
                index = self.path['image_index']
                self.path['image'] = str(self.path['files'][index])
                return True
            else:
                logging.warning('You should set folder path.')
                return False
        else:
            logging.warning('You use the wrong method.')
            return False
            
            
    def setting_original(self, image = None, method = 'path'):
        logging.info('Canvas: setting image.')
        if method == 'path':
            if self.path['image']:
                self.original['bgr'] = cv2.imread(self.path['image'])
            else:
                logging.warning('You should set image path.')
                return False
        elif method == 'cv2':
            self.original['bgr'] = image
        else:
            logging.warning('You use the wrong method.')
            return False
        
        # convert
        self.original['rgb'] = cv2.cvtColor(self.original['bgr'], cv2.COLOR_BGR2RGB)
        height, width = self.original['bgr'].shape[:2]
        
        self.image_width = int(self.image_height * width / height)
        
        # shrink  -> INTER_AREA
        # enlarge -> (INTER_CUBIC slow best), (INTER_LINEAR faster ok)
        self.original['bgr'] = cv2.resize(self.original['bgr'], (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        self.original['rgb'] = cv2.cvtColor(self.original['bgr'], cv2.COLOR_BGR2RGB)
        self.original['hsv'] = cv2.cvtColor(self.original['bgr'], cv2.COLOR_BGR2HSV)
        return True

    
    def original2image(self):
        if self.original['bgr'] is not None:
            self.image = self.original['bgr']
            return True
        else:
            logging.warning('You should setting background image.')
            return False
        
            
    
    def next_image(self):
        logging.info('Canvas: next image.')
        self.path['image_index'] = (self.path['image_index'] + 1) % len(self.path['files'])
        self.setting_image_path(method = 'index')
        self.setting_original(method = 'path')
        self.original2image()
        
    def before_image(self):
        logging.info('Canvas: before image.')
        self.path['image_index'] = (self.path['image_index'] - 1) % len(self.path['files'])
        self.setting_image_path(method = 'index')
        self.setting_original(method = 'path')
        self.original2image()
    
    
    def show_original(self, method = ''):
        logging.info('Canvas: show original.')
        if method == 'plt':
            plt.figure(figsize=(self.canvas_width, self.canvas_height))
            plt.imshow(self.original['rgb'])
            plt.show()
        elif method == 'cv2':
            cv2.imshow('original', self.original['bgr'])
            while(True):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow('original')
        else:
            return self.original['rgb']
            
    def show_image(self, method = ''):
        logging.info('Canvas: show image.')
        if method == 'plt':
            plt.figure(figsize=(self.canvas_width, self.canvas_height))
            plt.imshow(self.image[:,:,::-1])
            plt.show()
        elif method == 'cv2':
            cv2.imshow('image', self.image)
            while(True):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow('image')
        else:
            return self.image[:,:,::-1]
            
        
    def save_image(self, name='output.jpg'):
        root_path = './'
        cv2.imwrite( root_path + name, self.image)
        
        
        
    def deal_hsv2image(self):
        cv2.namedWindow('select_hsv', 0)
        cv2.setMouseCallback("select_hsv", self.getposHsv)
        
        self.point_break = False
        while True:
            cv2.imshow("select_hsv", self.original['bgr'])
            
            if self.point_break:
                break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cv2.destroyWindow('select_hsv')
    
    def getposHsv(self, event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            _hsv = self.original['hsv'][y, x]
            
            self.hMin = max(_hsv[0] - 5, 0)
            self.sMin = max(_hsv[1] - 100, 0)
            self.vMin = max(_hsv[2] - 100, 0)
            self.hMax = min(_hsv[0] + 5, 179)
            self.sMax = min(_hsv[1] + 100, 255)
            self.vMax = min(_hsv[2] + 100, 255)

            self.lower = np.array([self.hMin, self.sMin, self.vMin])
            self.upper = np.array([self.hMax, self.sMax, self.vMax])
            
            mask = cv2.inRange(self.original['hsv'], self.lower, self.upper)
            self.image = cv2.bitwise_and(self.original['bgr'], self.original['bgr'], mask=mask)
            
            self.point_break = True
            
            
    def deal_bdd2image(self):
        cv2.namedWindow('select_bdd', 0)
        
        rect = cv2.selectROI("select_bdd", self.original['bgr'], showCrosshair=True, fromCenter=False)
        shape = self.original['bgr'].shape
        
        mask = np.zeros(shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(self.original['bgr'], mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        
        self.image = self.original['bgr']*mask2[:,:,np.newaxis]
        
        cv2.destroyWindow('select_bdd')
            
    
    





