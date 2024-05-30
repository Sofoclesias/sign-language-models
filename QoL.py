import torch.multiprocessing as mp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import torch
import cv2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GLOG_minloglevel"] ="2"

venv = os.path.dirname((os.path.abspath(__file__)))

# --------------
# Funciones generales

def create_multiproc_files(type: str):    
    equiv = {'graph':'mediapipe_landmarks','gradient':'hog_transform'}
    
    if type not in list(equiv.keys()):
        raise ValueError('Palabra clave no identificada.')
    else:
        func = equiv[type]
        path = f'{type}-processing/multiproc-{type}.py'
        
        if os.path.exists(path):
            pass
        else:
            commands = f"""\
import datetime
import os
import sys
import shutil
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GLOG_minloglevel"] ="3"

def git_root(path):
    current_path = os.path.abspath(path)
    while current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    return None

def set_root():
    current = os.getcwd()
    sys.path.append(git_root(current))

set_root()
import QoL as qol
config = qol.device_configuration()

def proc(*args):
    path, lock = args
    ins = qol.{func}(path)
    with lock:
        ins.to_csv(normalize=True)
        qol.dump_object(ins, 'dump.pkl')

def main(paths):
    with mp.Manager() as manager:
        lock = manager.Lock()
        with mp.Pool(processes=config.max_cores//2) as pool:
            pool.starmap(proc, [(path, lock) for path in paths])
            pool.close()
            pool.join()
    shutil.move('dump.pkl', r'{type}-processing/')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    _, _, paths = qol.retrieve_raw_paths()
    tiempo_0 = datetime.datetime.today()
    
    print('Procesamiento iniciado -',datetime.datetime.today())
    main(paths)
    print('Procesamiento terminado -', datetime.datetime.today())
    print('Tiempo invertido: ',datetime.datetime.today()-tiempo_0)"""

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

def load_model(keywords: dict):
    # keywords = {'tecnica': (graph,gradient,neural), 'modelo':(knn,rf,rn)}
    path = venv + f'{keywords["tecnica"]}-processing/{keywords["modelo"]}-model.pkl'
    with open(path,'rb') as file:
        model = pickle.load(file)
    return model

def select_lists(*args):
    valid = [lst for lst in args if isinstance(lst.multi_hand_landmarks,list)]
    
    if valid:
        return valid[0]
    else:
        return None

def dump_object(obj,filename):
    if os.path.exists(filename):
        with open(filename,'rb') as file:
            data = pickle.load(file)
        data.append(obj)
    else:
        data = [obj]
        
    with open(filename,'wb') as file:
        pickle.dump(data,file)

# --------------
# Transformación de imágenes

class device_configuration:
    def __init__(self,preference: str = None):
        # Configura la unidad de procesamiento utilizada
        if preference is None:
            if torch.cuda.is_available():
                self.processing_unit = 'cuda:0'
                self.device = torch.device('cuda:0')
            else:
                self.processing_unit = 'cpu'
                self.device = torch.device('cpu')
        else:
            self.processing_unit = preference
            self.device = torch.device(preference)

        # Inicia la unidad de procesamiento para PyTorch
        torch.cuda.device(self.device)

        # Consolida como variable de interés los cores máximos
        if self.processing_unit=='cuda:0':
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
        blurred_image = cv2.GaussianBlur(self.image,(7,7),0)
        return blurred_image, self.color    
    
    @__to_self
    def manual_adjust_luminosity(self,brightness: int = 0,contrast: int = 0):
        if brightness==0 and contrast==0:
            image = self.image
        else:
            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                
                alpha_b = (highlight - shadow) / 255
                image = cv2.addWeighted(self.image, alpha_b, self.image, 0, shadow)
            
            if contrast != 0:
                f = 131 * (contrast + 127) / (127 * (131 - contrast))
                alpha_c = f
                gamma_c = 127 * (1 - f)
                image = cv2.addWeighted(self.image, alpha_c, self.image, 0, gamma_c)
        
        return image, self.color
    
    @__to_self
    def auto_adjust_luminosity(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        avg_bright = np.mean(gray)
        std_contrast = np.std(gray)
        
        brightness = int(128 - avg_bright)
        contrast = int((std_contrast / 64.0) * 127.0)
        
        im = self.manual_adjust_luminosity(brightness,contrast,to_self=False)
        return im.image, self.color
    
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
        padding = 5
        x = max(1, x - padding)
        y = max(1, y - padding)
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

class mediapipe_landmarks(image_preprocessing):
    def __init__(self, image_path, color: str = 'bgr'):
        import mediapipe as mp
        super().__init__(image_path,color)
        self.image_path = image_path
        #self.letter = self.image_path.split('\\')[-2]
        
        # Iniciar el framework de reconocimiento de coordenadas
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Leer y procesar la imagen
        image_rgb: image_preprocessing = self.to_rgb(to_self=False)
        
        results = select_lists(hands.process(image_rgb.image), hands.process(self.original_image))
        self.coords: np.array = np.empty((0, 2))

        # Si hay resultados para las imágenes        
        if results is not None:
            self.results: bool = True
            # Recuperar nodos de la imagen
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Extraer las coordenadas y guardarlas en lista
                    h, w, c = self.image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    self.coords = np.vstack((self.coords, np.array([cx, cy]).reshape(1, -1)))
                mp_drawing.draw_landmarks(image_rgb.image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            self.results: bool = False
            self.coords = np.zeros((21, 2))

        hands.close()
        
        self.image = image_rgb.image
        self.__is_normalized: bool = False

    def visualize_landmarks(self):     
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
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

        ruta = venv + r'\graph-processing\processed_data\{}.csv'.format(self.image_path.split("\\")[-3])
        df = pd.DataFrame([coords],columns=columns)
        
        df.to_csv(ruta, index=True, mode='a', header=not os.path.exists(ruta))
        
    def extract_values (self,normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_coords()
        else: pass
        
        return self.coords.flatten().tolist()
           
class hog_transform(image_preprocessing):
    def __init__(self, image_path, color: str = 'bgr'):
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
        columns = ['letra'] + [f'cell_{i}' for i in range(len(feats)-1)]
        df = pd.DataFrame([feats],columns=columns)
        ruta = venv + r'\gradient-processing\processed_data\{}.csv'.format(self.image_path.split("\\")[-3])
        
        df.to_csv(ruta, index=True, mode='a', header=not os.path.exists(ruta))
        
    def extract_values (self,normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_hog()
        else: pass
        
        self.image = self.hog_image
        return self.hog_features.flatten().tolist()

# Acá iría la clase de CNN



class model_trainer:
    def __init__(self,keywords: dict):
        # keywords = {'técnica': (graph,gradient) , 'modelo':(knn,rf)}
        # Claves
        if keywords['técnica'] in ['graph','gradient','neural'] and keywords['modelo'] in ['knn','rf','rn']:
            self.representacion = keywords['técnica']
            self.clave_modelo = keywords['modelo']
        else: 
            raise ValueError('Técnica de representación o modelo no identificado.')
        
        # Conjuntos de entrenamiento y prueba
        train_set = pd.DataFrame(venv + f'{self.representacion}-processing/processed_data/Train_Alphabet.csv')
        test_set = pd.DataFrame(venv + f'{self.representacion}-processing/processed_data/Test_Alphabet.csv')

        self.train_set = [train_set.iloc[:,1:],train_set.iloc[:,0]]
        self.test_set = [test_set.iloc[:,1:],test_set.iloc[:,0]]
        
        # Definición de los modelos e hiperparámetros
        self.modelo = None
        self.param_distributions = None
        self.__is_trained = False
        self.train_report = None
        self.test_report = None
        
    def setup_model(self):
        if self.clave_modelo == 'knn':
            self.modelo = KNeighborsClassifier()
            self.param_distributions = {
                'n_neighbors': np.arange(1, 31),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        elif self.clave_modelo == 'rf':
            self.modelo = RandomForestClassifier(random_state=42)
            self.param_distributions = {
                'n_estimators': np.arange(10, 200),
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None] + list(np.arange(5, 50, 5)),
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': np.arange(1, 11)
            }
        elif self.clave_modelo == 'rn':
            pass
        
    def train_model(self):
        # Configurar el modelo
        self.setup_model()
        
        # Configurar RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.modelo,
            param_distributions=self.param_distributions,
            n_iter=100, # Por ajustar
            cv = 5, # Por ajustar
            verbose = 2,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(*self.train_set)
        self.modelo = random_search.best_estimator_
        self.__is_trained = True
        
        # Reportes de error
        self.train_report = classification_report(self.train_set[1],self.modelo.predict(self.train_set[0]))
        self.test_report = classification_report(self.test_set[1],self.modelo.predict(self.test_set[0]))
    
    def export_model(self):
        path = venv + f'{self.representacion}-processing/{self.clave_modelo}-model.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self,file)

# En general, este archivo py no debería ser iniciado desde la raíz nunca, dado que es contraproducente. No obstante, lo haré acá para crear los multiprocs.
if __name__ == '__main__':
    create_multiproc_files('graph')
    create_multiproc_files('gradient')