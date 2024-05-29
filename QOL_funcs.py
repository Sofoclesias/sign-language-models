from multiprocessing import Lock
import pandas as pd
import numpy as np
import os
import torch
import functools
import cv2
import pickle

venv = os.path.dirname((os.path.abspath(__file__)))

# --------------
# Funciones generales

def create_multiproc_files(func: str,path: str):    
    if os.path.exists(path):
        pass
    else:
        commands = f"""\
import datetime
import warnings
import QOL_funcs as qol
warnings.filterwarnings('ignore')

def proc(path):
    ins = qol.{func}(path)
    ins.to_csv(normalize=True)
    qol.dump_object(ins,'dump.pkl')

if __name__ == '__main__':
    # Iniciar PyTorch con configuraciones
    config = qol.device_configuration()

    if config.processing_unit == 'cuda':
        import torch.multiprocessing as multiproc
        try:
            multiproc.set_start_method('spawn')
        except RuntimeError:
            pass
    else:
        import multiprocessing as multiproc

    train, test, _all = qol.retrieve_raw_paths()

    tiempo_0 = datetime.datetime.today()
    print('Procesamiento iniciado -',datetime.datetime.today())

    with multiproc.Pool(processes=config.max_cores//2) as pool:
        pool.map(proc, _all)
        pool.close()
        pool.join()

    print('Procesamiento termina -', datetime.datetime.today())
    print('Tiempo invertido: ',datetime.datetime.today()-tiempo_0)
    """

        with open(path, 'w') as file:
            file.write(commands)

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths

def retrieve_raw_paths():
    train = get_file_paths(venv + r'\synthetic-asl-alphabet\Train_Alphabet')
    test = get_file_paths(venv + r'\synthetic-asl-alphabet\Test_Alphabet')
    _all = [*train, *test]
    return train, test, _all

def dataset_exists():
    if os.path.exists(venv + r'\synthetic-asl-alphabet'):
        pass
    else:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet",)

def locking(lock):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@locking(Lock())
def append_csv(ruta: str, df: pd.DataFrame):
    try:
        df.to_csv(ruta, index=True, mode='a', header=not os.path.exists(ruta))
    except: pass

@locking(Lock())
def dump_object(obj,filename):
    if os.path.exists(filename):
        with open(filename,'rb') as file:
            data = pickle.load(file)
        data.append(obj)
    else:
        data = [obj]
        
    with open(filename,'wb') as file:
        pickle.dump(data,file)
    
def camera(): # FUNCIÓN NO UTILIZADA. SERVIRÁ PARA MÁS ADELANTE.
    cv2.imshow()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------
# Transformación de imágenes

class device_configuration:
    def __init__(self,preference: str = None):
        # Configura la unidad de procesamiento utilizada
        if preference is None:
            if torch.cuda.is_available():
                self.processing_unit = 'cuda'
                self.device = torch.device('cuda')
            else:
                self.processing_unit = 'cpu'
                self.device = torch.device('cpu')
        else:
            self.processing_unit = preference
            self.device = torch.device(preference)

        # Inicia la unidad de procesamiento para PyTorch
        torch.cuda.device(self.device)

        # Consolida como variable de interés los cores máximos
        if self.processing_unit=='cuda':
            self.max_cores = torch.cuda.get_device_properties(0).multi_processor_count
        else:
            self.max_cores = os.cpu_count()

class image_preprocessing:
    def __init__(self, image: str | np.ndarray, color: str = 'bgr'):
        if isinstance(image, str):    
            self.original_image = cv2.imread(image)
            self.image = cv2.imread(image)
            self.color = color
        elif isinstance(image, np.ndarray):
            self.original_image = image
            self.image = image
            self.color = color
        else:
            raise ValueError("No se pudo cargar la imagen.")
        
        self.size = self.image.shape
    
    def __to_self(func):
        def wrapper(self, *args, **kwargs):
            to_self = kwargs.pop('to_self', False)
            result = func(self, *args, **kwargs)
            if to_self:
                self.image, self.color = result
                self.size = self.image.shape
                                    
                return result
            else:
                image, color = result
                new_instance = image_preprocessing(image=image,color=color)
                
                return new_instance
        return wrapper
    
    @__to_self
    def resize_image(self, pixels: int):
        resized_image = cv2.resize(self.image, (pixels, pixels))
        return resized_image, self.color
    
    @__to_self
    def to_grayscale(self):
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return grayscale_image, 'gray'
        
    @__to_self
    def to_rgb(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return rgb_image, 'rgb'
    
    @__to_self
    def blur_image(self):
        blurred_image = cv2.GaussianBlur(self.image)
        return blurred_image, self.color
    
    def edge_detection(self):
        gray: image_preprocessing = self.to_grayscale(to_self=False)
        blurred: image_preprocessing = gray.blur_image(to_self=False)
        edges = cv2.Canny(blurred.image,50,150)
        return edges
        
    def __find_hand_rectangle(self):
        # Detectar los bordes
        edges = self.edge_detection()
        
        # Buscar contornos y seleccionar el mayor
        contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Crear el rectángulo de ajuste
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        # Expandirlo para cubrir toda la palma (padding manualmente ajustable)
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.image.shape[1] - x, w + 2 * padding)
        h = min(self.image.shape[0] - y, h + 2 * padding)  
    
        # (X_POINT_START, Y_POINT_START, WIDTH, HEIGHT)
        return (x, y, w, h)
    
    @__to_self
    def segment_image(self):
        mask = np.zeros(self.image.shape[:2], np.uint8)
        
        bgm = np.zeros((1,65), np.float64)
        fgm = np.zeros((1,65), np.float64)

        rectangle = self.__find_hand_rectangle()
        
        cv2.grabCut(self.image, mask, rectangle, bgm, fgm, 20, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
        
        segmented = self.image * mask2[:,:,np.newaxis]
        
        return segmented, self.color

class mediapipe_landmarks:
    def __init__(self, image_path: str):
        import mediapipe as mp
        
        self.image_path: str = image_path
        
        # Iniciar el framework de reconocimiento de coordenadas
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Leer y procesar la imagen
        image = cv2.imread(image_path)
        self.__image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(self.__image_rgb)
        self.coords: np.array = np.empty((0, 2))

        # Si hay resultados para las imágenes
        if results.multi_hand_landmarks:
            self.results: bool = True
            # Recuperar nodos de la imagen
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Extraer las coordenadas y guardarlas en lista
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    self.coords = np.vstack((self.coords, np.array([cx, cy]).reshape(1, -1)))
                mp_drawing.draw_landmarks(self.__image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            self.results: bool = False
            self.coords = np.zeros((21, 2))

        hands.close()
        
        self.__is_normalized: bool = False

    def visualize_landmarks(self):     
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(self.__image_rgb)
        plt.axis('off')  # Ocultar los ejes
        plt.show()
        
    def normalize_coords(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.__is_normalized: bool = True
        self.coords = scaler.fit_transform(self.coords)
        
    def to_csv(self,normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_coords()
        else: pass
        
        coords = [self.image_path.split('\\')[-2]] + self.coords.flatten().tolist()
        
        columns = ['letra']
        for i in range(21):
            columns.extend([f"x_{i}", f"y_{i}"])

        ruta = r'\graph-processing\processed_data\{}.csv'.format(self.image_path.split("\\")[-3])
        append_csv(venv + ruta, pd.DataFrame([coords], columns=columns))
        
        
class hog_transform(image_preprocessing):
    def __init__(self, image_path: str, color: str = 'bgr'):
        from skimage.feature import hog
        super().__init__(image_path,color)
        self.image_path = image_path

        # Preprocesamiento
        self.resize_image(64,to_self=True)
        self.segment_image(to_self=True)
        self.to_grayscale(to_self=True)
        
        # Extracción de características con HOG
        self.hog_features, self.hog_image = hog(self.image, orientations=9,pixels_per_cell=(8,8), cells_per_block=(2,2),block_norm='L2-Hys',visualize=True)        
        self.__is_normalized: bool = False
    
    def visualize_gradients(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(self.hog_image)
        plt.axis('off')  # Ocultar los ejes
        plt.show()
        
    def normalize_hog(self):
        from skimage import exposure
        self.hog_features = exposure.rescale_intensity(self.hog_features, in_range=(0,10))
        self.__is_normalized: bool = True
        
    def to_csv(self, normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_hog()
        else: pass
        
        feats = [self.image_path.split('\\')[-2]] + self.hog_features.flatten().tolist()
        
        columns = ['letra'] + [f'cell_{i}' for i in range(len(feats))]

        ruta = r'\gradient-processing\processed_data\{}.csv'.format(self.image_path.split("\\")[-3])
        append_csv(venv + ruta, pd.DataFrame([feats], columns=columns))

# En general, este archivo py no debería ser iniciado desde la raíz nunca, dado que es contraproducente. No obstante, lo haré acá para crear los multiprocs.
if __name__ == '__main__':
    # Graph
    create_multiproc_files('mediapipe_landmarks','graph-processing/multiproc-graph.py')
    
    # Gradient
    create_multiproc_files('hog_transform','gradient-processing/multiproc-gradient.py')