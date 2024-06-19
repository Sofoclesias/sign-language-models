from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.multiprocessing import Pool, Manager
from scikeras.wrappers import KerasClassifier
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam, SGD, RMSprop,Adagrad
from PIL import Image, ImageOps
import random
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GLOG_minloglevel"] ="2"

venv = os.path.dirname((os.path.abspath(__file__)))
letras = list(string.ascii_uppercase)

# --------------
# Funciones generales
                  
def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths

def retrieve_raw_paths():
    train = get_file_paths(venv + r'\preprocessed-images\Train_Alphabet')
    test = get_file_paths(venv + r'\preprocessed-images\Test_Alphabet')
    _all = [*train, *test]
    return train, test, _all

def dataset_exists():
    if os.path.exists(venv + r'\synthetic-asl-alphabet'):
        pass
    else:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet")

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

def ANN(neurons, activation, input_shape, output_shape):
    model = Sequential()
    model.add(Dense(neurons[0], activation=activation, input_shape=(input_shape,)))
    model.add(Dense(neurons[1], activation=activation))
    model.add(Dense(neurons[2], activation=activation))
    model.add(Dense(neurons[3], activation=activation))
    model.add(Dense(output_shape, activation='softmax'))

    return model

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
            to_self = kwargs.pop('to_self', False) # Creo que es más intuitivo que to_self sea 'True' por defecto. Luego lo cambio
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
        
    @__to_self
    def edge_enhancement(self,contrast: str = 'hard'):
        # Filtro de convolución de 3x3
        if contrast=='hard':
            kernel = np.array([[-1, -1, -1],
                               [-1, 10, -1],
                               [-1, -1, -1]])
        elif contrast=='soft':
            kernel = np.array([[0,-1,0],
                              [-1,5,-1],
                              [0,-1,0]])
        else:
            raise ValueError('Tipo de contraste no aceptado.')
        
        enhanced_image = cv2.filter2D(self.image, -1, kernel)
        return enhanced_image, self.image
        
    @__to_self
    def histogram_equalization(self,contrast: str = 'global'):
        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        if contrast=='global': # Solo Histogram Equalization
            return cv2.equalizeHist(self.image), self.color
        elif contrast=='adaptive': # Adaptive Histogram Equalization
            eq = cv2.createCLAHE(clipLimit=40.0,tileGridSize=(8,8))
        elif contrast=='limited-adaptive': # Contrast Limited Adaptive Histogram Equalization
            eq = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        else:
            raise ValueError('Tipo de contraste no aceptado')

        l = eq.apply(l)  
        lab = cv2.merge((l,a,b)) 
        equalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) 
        return equalized_image, self.color
    
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
    
    @__to_self
    def adaptive_crop(self):
        x,y,w,h = self.__find_hand_rectangle()
        cropped_image = self.image[y:y+h,x:x+w]
        return cropped_image, self.color
    
    @staticmethod
    def save_image (path,image):
        parts = path.split(os.sep)
        type_dir,letter_dir = parts[-3],parts[-2]
        
        output_dir = os.path.join(venv,'preprocessed-images',type_dir,letter_dir)
        os.makedirs(output_dir,exist_ok=True)
        i = len(os.listdir(output_dir)) + 1
        
        output_path = os.path.join(output_dir,f'{i}.png')
        cv2.imwrite(output_path,image)
    
    @staticmethod
    def cuadrar(x,y,w,h):
        if w==h:
            pass
        else:
            missing,equal = int(round(abs((w-h)/2),0)), max(w,h)
            x -= 100
            y -= 100
            
            if w>h:
                x -= missing
            elif w<h:
                y -= missing
        return x,y,equal
    
    @staticmethod
    def augmentation_en_carpetas(ruta_principal,N):
        def contar_imagenes(carpeta):
            return len([f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))])

        def obtener_siguiente_nombre(carpeta):
            nombres = [f for f in os.listdir(carpeta) if f.endswith('.png')]
            numeros = sorted([int(f.split('.')[0]) for f in nombres])
            
            for i in range(1, len(numeros) + 1):
                if i not in numeros:
                    return f'{i}.png'
            return f'{len(numeros) + 1}.png'

        def azar(imagen):
            operaciones = [
                lambda x: x.rotate(random.uniform(-30, 30)),
                lambda x: ImageOps.mirror(x)
            ]
            
            operacion = random.choice(operaciones)
            return operacion(imagen)
        
        for carpeta in os.listdir(ruta_principal):
            ruta_carpeta = os.path.join(ruta_principal, carpeta)
            if os.path.isdir(ruta_carpeta):
                imagenes = [f for f in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, f))]
                while contar_imagenes(ruta_carpeta) < N:
                    # Selección de imagen aleatoria
                    imagen_aleatoria = random.choice(imagenes)
                    ruta_imagen = os.path.join(ruta_carpeta, imagen_aleatoria)
                    imagen = Image.open(ruta_imagen)
                    
                    # Transformación
                    imagen_transformada = azar(imagen)
                    
                    # Guardado en directorio respectivo
                    nuevo_nombre = obtener_siguiente_nombre(ruta_carpeta)
                    ruta_nueva_imagen = os.path.join(ruta_carpeta, nuevo_nombre)
                    imagen_transformada.save(ruta_nueva_imagen)
    
    @staticmethod
    def main_preprocess(path,save=True, *args):
        from rembg import remove
        
        whole, upframe, downframe = args
        rf, project, model = image_preprocessing.initialize_external()
        
        try:
            # Quitar fondo
            image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
            image = remove(image)
            
            # ---------
            # Sección agregada luego del preprocesamiento general
            if str(path.split(os.sep)[-2]) in whole:
                # Se utilizan un modelo de reconocimiento de objetos para la tarea de croppear
                result = model.predict(path, confidence=1, overlap=1).json()
                
                if result['predictions']:
                    modelled_bbox = (int(result['predictions'][0]['x']),
                                    int(result['predictions'][0]['width']),
                                    int(result['predictions'][0]['y']),
                                    int(result['predictions'][0]['height']))
                    x,y,h = image_preprocessing.cuadrar(*modelled_bbox)
                else:
                    raise ValueError('Comodín.')
                
            else:
                # Se utilizan los contornos máximos de cada imagen para la tarea de croppear
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                y -= 100
                
                if str(path.split(os.sep)[-2])=='B':
                    x -= 100
                elif str(path.split(os.sep)[-2]) in upframe:
                    x -= 100
                elif str(path.split(os.sep)[-2]) in downframe:
                    x += 100
                else:
                    pass
            # ----------    
            image = image[x:x+h,y:y+h] 
            
            # Guardar
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            if save==True:
                image_preprocessing.save_image(path,image)
            else:
                return image
        except:
            pass
    
    @staticmethod
    def extractor(extract_class, configs, path, lock):
        # No se considera 'convolutional' porque ya tiene métodos de multiprocessing internos.
        try:
            ins = extract_class(path,**configs)
            with lock:
                ins.to_csv()
        except:
            pass
            
    @staticmethod
    def multiextractor(extract_class,configs,paths):
        with Manager() as manager:
            lock = manager.Lock()
            with Pool(processes=8) as pool:
                pool.starmap(image_preprocessing.extractor, [(extract_class,configs,path,lock) for path in paths])
                pool.close()
                pool.join()
                
    @classmethod
    def initialize_external(cls, instance=None,inner=False):
        from roboflow import Roboflow
        
        if inner:
            if instance is None:
                raise ValueError('Se debe proveer una instancia cuando inner es True.')
            
            instance.rf = Roboflow(api_key="YU30pOFC9cA5yqoJUQij")
            instance.project = instance.rf.workspace().project("hands-rzvws")
            instance.model = instance.project.version(1).model
        else:
            rf = Roboflow(api_key="YU30pOFC9cA5yqoJUQij")
            project = rf.workspace().project("hands-rzvws")
            model = project.version(1).model
            
            return rf, project, model
        
class mediapipe_landmarks(image_preprocessing):
    def __init__(self, image_path, color: str = 'bgr'):
        import mediapipe as mp
        super().__init__(image_path,color)
        self.image_path = image_path
        self.letter = self.image_path.split('/')[-2]
        
        # Iniciar el framework de reconocimiento de coordenadas
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)
        mp_drawing = mp.solutions.drawing_utils

        # Leer y procesar la imagen
        self.to_rgb(to_self=True)
        results = hands.process(self.image)
        self.coords: np.array = np.empty((0, 2))

        # Si hay resultados para las imágenes        
        if results.multi_hand_landmarks:
            self.results: bool = True
            # Recuperar nodos de la imagen
            for hand_landmarks in results.multi_hand_landmarks:
                for _, landmark in enumerate(hand_landmarks.landmark):
                    # Extraer las coordenadas y guardarlas en lista
                    h, w, c = self.image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    self.coords = np.vstack((self.coords, np.array([cx, cy]).reshape(1, -1)))
                mp_drawing.draw_landmarks(self.image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            self.results: bool = False
            self.coords = np.zeros((21, 2))
            os.remove(self.image_path)

        hands.close()
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
        
    def to_csv(self,name='path'):
        if self.results:
            self.normalize_coords()
            
            coords = [self.image_path.split('/')[-2]] + self.coords.flatten().tolist()
            columns = ['letra']
            for i in range(21):
                columns.extend([f"x_{i}", f"y_{i}"])

            df = pd.DataFrame([coords],columns=columns)
            
            if name =='path':
                ruta = r'{}.csv'.format(self.image_path.split("/")[-3])
            else:
                ruta = name
        
            df.to_csv(ruta, index=False, mode='a', header=not os.path.exists(ruta))
        else:
            pass
            
    def extract_values (self,normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_coords()
        else: pass
        
        return self.coords.flatten().tolist()
           
class hog_transform(image_preprocessing):
    def __init__(self, image_path, color: str = 'bgr',ppc=(8,8),cpb=(3,3)):
        from skimage.feature import hog
        super().__init__(image_path,color)
        self.image_path = image_path

        # Preprocesamiento
        self.resize_image(128,to_self=True)
        self.to_grayscale(to_self=True)
        
        # Extracción de características con HOG
        self.hog_features, self.hog_image = hog(self.image, orientations=9,pixels_per_cell=ppc, cells_per_block=cpb,block_norm='L2',visualize=True)        
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
        
    def to_csv(self, name: str = 'path'):
        self.normalize_hog()
        
        feats = [self.image_path.split('/')[-2]] + self.hog_features.flatten().tolist()
        columns = ['letra'] + [f'cell_{i}' for i in range(len(feats)-1)]
        df = pd.DataFrame([feats],columns=columns)
        
        if name =='path':
            ruta = r'{}.csv'.format(self.image_path.split("/")[-3])
        else:
            ruta = name
        
        df.to_csv(ruta, index=True, mode='a', header=not os.path.exists(ruta))
        
    def extract_values (self,normalize: bool = True):
        if normalize==True and self.__is_normalized==False:
            self.normalize_hog()
        else: pass
        
        self.image = self.hog_image
        return self.hog_features.flatten().tolist()

class CustomCNN(nn.Module):
    def __init__(self, config):
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = 1  # Grayscale image

        for _ in range(config['NCB']):
            out_channels = config['Ncf']
            kernel_size = config['Sck']
            
            if config['Tacti'] == 'ReLU':
                activation = nn.ReLU()
            elif config['Tacti'] == 'PReLU':
                activation = nn.PReLU()
            
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            layers.append(activation)
            
            if config['Tpool'] == 'max':
                layers.append(nn.MaxPool2d(kernel_size=config['Spk'], stride=2))
            elif config['Tpool'] == 'average':
                layers.append(nn.AvgPool2d(kernel_size=config['Spk'], stride=2))
            
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * (config['input_size'] // (2 ** config['NCB'])) ** 2, config['cant_neurons'])  # Capa totalmente conectada con 256 neuronas
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x

class cnn_featurize(image_preprocessing):
            def __init__(self, image_path, color: str = 'bgr'):
                super().__init__(image_path,color)
                self.image_path = image_path
                self.letter = self.image_path.split('/')[-2]

                # Preprocesamiento
                self.resize_image(128,to_self=True)
                self.to_grayscale(to_self=True)
                self.image = self.image.astype('float32') / 255.0
                self.image_tensor = torch.tensor(self.image).unsqueeze(0)

class cnn_extractor:
    def __init__(self, train_paths, config, epochs=10):
        self.config = config
        self.model = CustomCNN(self.config)
        self.__get_dataset(train_paths)
        self.__train_model(epochs=epochs)

    @staticmethod
    def create_featurizer(path):
        try:
            return cnn_featurize(path)
        except:
            pass

    def __get_dataset(self,paths):
        with Pool(processes=8) as pool:
            featurizers = pool.map(cnn_extractor.create_featurizer, paths) 
        
        self.le = LabelEncoder()
        valid_featurizers = [featurizer for featurizer in featurizers if featurizer is not None]

        self.__tensors = torch.stack([featurizer.image_tensor for featurizer in valid_featurizers])
        self.__labels = torch.tensor(self.le.fit_transform([featurizer.letter for featurizer in valid_featurizers]), dtype=torch.long)

        self.dataset = TensorDataset(self.__tensors,self.__labels)

    def __train_model(self,epochs=20):
        # Optimizadores
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Datos de entrenamiento y validación
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        T_ds, v_ds = random_split(self.dataset,[train_size,val_size],generator=torch.Generator().manual_seed(42))
        T_loader, v_loader = DataLoader(T_ds,batch_size=1,shuffle=True), DataLoader(v_ds,batch_size=1,shuffle=False)
        
        # Inicio entrenamiento
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in T_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(T_loader)}")
            
            # Inicio validación
            self.model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            
            with torch.no_grad():
                for inputs, labels in v_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs,labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f"Validation Loss: {val_loss / len(v_loader)}, Accuracy: {100 * correct / total}%")
            self.model.train()   
    
    @staticmethod
    def extract_features(image_path,model):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype('float32') / 255.0
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        # Extraer características
        model.eval()
        with torch.no_grad():
            features = model(image_tensor)
        return features.numpy().flatten()

    @staticmethod
    def normalize_cnn(features):
            features_norm = (features - np.mean(features)) / np.std(features)
            return features_norm
    
    @staticmethod
    def multiexport(path,name,lock,features):
            feats = [path.split('/')[-2]] + features.tolist()
            columns = ['letra'] + [f'feature_{i}' for i in range(len(feats) - 1)]
            df = pd.DataFrame([feats],columns=columns)
            
            if name =='path':
                ruta = r'{}.csv'.format(path.split("/")[-3])
            else:
                ruta = name
            
            if lock is None:
                df.to_csv(ruta, index=False, mode='a', header=not os.path.exists(ruta))
            else:
                with lock:
                    df.to_csv(ruta, index=False, mode='a', header=not os.path.exists(ruta))

    @staticmethod    
    def tensorize_image(image_paths, model):
        tensors = []
        valid_paths = []
        for path in image_paths:
            try:
                tensors.append(cnn_extractor.normalize_cnn(cnn_extractor.extract_features(path, model)))
                valid_paths.append(path)
            except:
                pass
        return valid_paths, tensors

    def transform_to_csv(self,image_paths,name:str='path',n_jobs=-1):
        print('Exporting.')
        
        chunk_size = len(image_paths) // 16
        image_path_chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
        
        all_paths = []
        all_tensors = []
        
        with Pool(processes=8) as pool:
            for valid_paths, tensors in pool.starmap(cnn_extractor.tensorize_image, [(chunk, self.model) for chunk in image_path_chunks]):
                all_paths.extend(valid_paths)
                all_tensors.extend(tensors)
        
        if n_jobs==-1:
            with Manager() as manager:
                lock = manager.Lock()
                with Pool(processes=8) as pool:
                    pool.starmap(cnn_extractor.multiexport, [(all_paths[i],name,lock,all_tensors[i]) for i in range(len(all_paths))])
                    pool.close()
                    pool.join()
        else:
            for path in image_paths:
                cnn_extractor.multiexport(path,name, None, self.extract_features)

class model_trainer:
    knn_params = {'n_neighbors','weights','metric'}
    rf_params = {'n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf'}
    ann_params = {'model__neurons','model__activation','optimizer','optimizer__learning_rate','epochs','batch_size'}
    
    def __init__(self, tecnica: str, modelo: str, cnn_extractor=None):
        # keywords = {'técnica': (graph,gradient,convolutional) , 'modelo':(knn,rf,ann)}
        # Claves
        if tecnica in ['graph','gradient','convolutional'] and modelo in ['knn','rf','ann']:
            self.technique = tecnica
            self.key_model = modelo
            
            if tecnica=='convolutional':
                if cnn_extractor is not None:
                    self.convolutor = cnn_extractor
                else:
                    raise ValueError("El argumento 'convolutional' requiere su respectivo convolutor.")
        else: 
            raise ValueError('Técnica de representación o modelo no identificado.')
        

        self.dataset_path = {'train': '/content/drive/MyDrive/ml-processing/' + f'{self.technique}-processing/Train_Alphabet.csv',
                             'test': '/content/drive/MyDrive/ml-processing/' + f'{self.technique}-processing/Test_Alphabet.csv'}

        # Conjuntos de entrenamiento y prueba
        train_set = pd.read_csv(self.dataset_path['train'],sep=',', encoding='utf-8',on_bad_lines='skip',usecols=lambda column: column not in ['Unnamed: 0','origen' ,'    '])
        test_set = pd.read_csv(self.dataset_path['test'],sep=',', encoding='utf-8',on_bad_lines='skip',usecols=lambda column: column not in ['Unnamed: 0','origen' ,'    '])

        # Prueba de errores
        lines_train = sum(1 for _ in open(self.dataset_path['train']))
        lines_test = sum(1 for _ in open(self.dataset_path['test']))
        train_corr = train_set.isna().any(axis=1).sum()
        test_corr = test_set.isna().any(axis=1).sum()
        miss_T = train_set.dropna(axis=0).iloc[:,1:].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)
        miss_t = test_set.dropna(axis=0).iloc[:,1:].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)        
        
        if ((lines_train > train_set.shape[0] or lines_test > test_set.shape[0] or train_corr>0 or test_corr>0 or miss_T.sum()>0 or miss_t.sum()>0) and (not lines_train - train_set.shape[0]==1 and not lines_test - test_set.shape[0]==1)):
            print('----- Sumilla de errores -----\n')
            if (lines_train > train_set.shape[0] or lines_test > test_set.shape[0]):
                print(f'Se han encontrado {lines_train - train_set.shape[0]} errores de lectura en train y {lines_test - test_set.shape[0]} en test. Borrando.')
            if (train_corr>0 or test_corr>0):
                print(f'Se han encontrado {train_corr} filas corruptas en train y {test_corr} en test. Borrando.')
                train_set.dropna(axis=0,inplace=True)
                test_set.dropna(axis=0,inplace=True)
            if (miss_T.sum()>0 or miss_t.sum()>0):
                print(f'Se han encontrado {miss_T.sum()} filas con letras mal posicionadas en train y {miss_t.sum()} en test. Borrando.')
                try:
                    train_set.drop(index=miss_T[miss_T==True].index,inplace=True)
                except: pass
                try:
                    test_set.drop(index=miss_t[miss_t==True].index,inplace=True)
                except: pass
            print('\n------------------------------')
                
        # Almacenamiento de datasets
        X_train, Y_train = train_set.iloc[:,1:].apply(pd.to_numeric, errors='coerce'), train_set.iloc[:,0].apply(lambda x: str(x.decode('utf-8')).strip("b' ") if isinstance(x, bytes) else str(x).strip("b' ")).astype('str').apply(lambda x: x.strip())
        X_test, Y_test = test_set.iloc[:,1:].apply(pd.to_numeric, errors='coerce'), test_set.iloc[:,0].apply(lambda x: str(x.decode('utf-8')).strip("b' ") if isinstance(x, bytes) else str(x).strip("b' ")).astype('str').apply(lambda x: x.strip())
        
        self.train_set = [X_train,Y_train]
        self.test_set = [X_test,Y_test]
        
        # Definición de atributos auxiliares
        self.label = LabelEncoder()
        self.label.fit(letras)
        self.__is_trained = False
    
    def class_counts(self):  
        train_count = self.train_set[1].value_counts().reset_index()
        train_count.columns = ['Letra', 'En train']
        
        test_count = self.test_set[1].value_counts().reset_index()
        test_count.columns = ['Letra', 'En test']
        
        train_count.Letra = train_count.Letra.astype(str)
        test_count.Letra = test_count.Letra.astype(str)
        
        consolidate = pd.merge(train_count, test_count, on='Letra', how='outer')
        consolidate.fillna(0,inplace=True)
        
        consolidate['En train'] = consolidate['En train'].astype(int)
        consolidate['En test'] = consolidate['En test'].astype(int)
        
        consolidate.sort_values(by='Letra', inplace=True)
        consolidate.reset_index(drop=True, inplace=True)
            
        return consolidate
        
    def __setup_model(self):
        if self.key_model == 'knn':
            self.classificator = KNeighborsClassifier()
            self.param_distributions = {
                'n_neighbors': list(range(1,50)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski','chebyshev']
            }
        elif self.key_model == 'rf':
            self.classificator = RandomForestClassifier(random_state=42)
            self.param_distributions = {
                'n_estimators': np.linspace(10, 1000, num=100, dtype=int).tolist(),
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': np.linspace(10, 100, num=15, dtype=int).tolist() + [None],
                'min_samples_split': np.linspace(2, 20, num=10, dtype=int).tolist(),
                'min_samples_leaf': np.linspace(1, 20, num=10, dtype=int).tolist()
            }
        elif self.key_model == 'ann':
            self.classificator = KerasClassifier(model=ANN,
                                                 model__input_shape=self.train_set[0].shape[1],
                                                 model__output_shape=26,
                                                 metrics=['accuracy'],
                                                 loss='sparse_categorical_crossentropy',
                                                 random_state=42,
                                                 verbose=0)
            self.param_distributions = {
                'model__neurons': [(2056,1024,512,256),(1024,512,256,128),(512,256,128,64)],
                'model__activation': ['relu', 'sigmoid', 'tanh', 'elu'],
                'optimizer': [Adam, SGD, RMSprop, Adagrad],
                'optimizer__learning_rate': [0.0001,0.001,0.01,0.1],
                'epochs': np.linspace(2, 50, num=5, dtype=int).tolist(),
                'batch_size': [32, 64]
            }
            
    def __preset_model(self,X_T,Y_T,**kwargs):
        if self.key_model == 'knn':
            self.classificator = KNeighborsClassifier(**kwargs)
        elif self.key_model == 'rf':
            self.classificator = RandomForestClassifier(**kwargs,random_state=42)
        elif self.key_model == 'ann':
            self.classificator = KerasClassifier(model=ANN,
                                          model__input_shape=self.train_set[0].shape[1],
                                          model__output_shape=26,
                                          **kwargs,
                                          metrics=['accuracy'],
                                          loss='sparse_categorical_crossentropy',
                                          random_state=42,
                                          verbose=0)
        self.classificator.fit(X_T,Y_T)
            
    def train_model(self,how: str = 'optimal',**kwargs):
        X_T = self.train_set[0]
        Y_T = self.label.transform(self.train_set[1])
        
        # Configurar el modelo
        if how=='optimal': 
            self.__setup_model()
            self.classificator = model_trainer.optimize_model(self.classificator, self.param_distributions, X_T, Y_T)
        elif how=='random':
            self.__setup_model()
            random_search = RandomizedSearchCV(estimator=self.classificator, 
                                        param_distributions=self.param_distributions,
                                        n_iter=50,
                                        cv=5, 
                                        random_state=42,
                                        n_jobs=-1,
                                        verbose=1)
            random_search.fit(X_T, Y_T)
            
            self.classificator = random_search.best_estimator_
        elif how=='preset':
            if not kwargs:
                raise ValueError("Se requieren **kwargs para el método how='optimal'.")
            self.__preset_model(X_T,Y_T,**kwargs)
        else:
            raise AttributeError('Valor en "how" no reconocido.')
        
        self.__generate_error_reports()
        self.__is_trained = True
        
    def __generate_error_reports(self):
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
        from sklearn.preprocessing import label_binarize
        
        # Seteo de variables
        y_test = self.test_set[1]
        y_pred = self.predict(self.test_set[0])
        y_prob = self.classificator.predict_proba(self.test_set[0])
        
        # Reportes generales de error
        self.test_report = classification_report(y_test,y_pred)
        self.CM = pd.DataFrame(confusion_matrix(y_test,y_pred),columns=letras,index=letras)
        self.normalized_CM = pd.DataFrame(confusion_matrix(y_test,y_pred,normalize='true'),columns=letras,index=letras)

        # ROC AUC de cada clase
        y_bintest = label_binarize(y_test,
                                    classes=[letra for letra in np.sort(np.unique(y_test))])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(y_bintest.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_bintest[:,i],y_prob[:,i])
            roc_auc[i] = auc(fpr[i],tpr[i])
            
        # ROC AUC: media ponderada
        roc_auc_ovr = roc_auc_score(y_bintest, y_prob, multi_class='ovr')
        
        # ROC AUC: macro
        roc_auc_ovo = roc_auc_score(y_bintest,y_prob,multi_class='ovo')
        
        # Consolidación
        self.AUC_scores = {
            'perclass':roc_auc,
            'ovr':roc_auc_ovr,
            'ovo':roc_auc_ovo
        }
        
        self.__support_AUC = {
            'y_bintest':y_bintest,
            'fpr':fpr,
            'tpr':tpr,
        }
    
    def export_model(self):        
        path = '/content/drive/MyDrive/ml-processing/' + f'{self.technique}-processing/models/{self.key_model}-model.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self,file)
    
    def predict(self, X_test):
        # Como está en Label, la predicción arrojaría un número.
        # Con este nuevo método de reemplazo, te arroja la letra directamente.
        return self.label.inverse_transform(self.classificator.predict(X_test))
    
    def show_report(self):
        print(f'Reporte de modelo de {self.key_model} para la técnica {self.technique}\n')
        print(self.test_report)
    
    def plot_CM(self,way='pandas-style',normalized=True):
        if normalized:
            CM = self.normalized_CM
        else:
            CM = self.CM
        
        if way=='pandas-style':
            stylish = CM.style.background_gradient(cmap='coolwarm')
            return stylish
        elif way=='seaborn':
            plt.figure(figsize=(10, 7))
            sns.heatmap(CM, annot=True, fmt="d", cmap="coolwarm")
            plt.title(f'Matriz de Confusión de {self.key_model} para la técnica {self.technique}')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.show()
        else:
            raise AttributeError("'way' debe ser o 'pandas-style' (predeterminado) o 'seaborn'." )
    
    def plot_roc_curves(self,view='perclass',letter='A'):
        plt.figure(figsize=(10, 7))
        if view == 'perclass':
            for i in range(self.__support_AUC['y_bintest'].shape[1]):
                label = self.label.inverse_transform(i)
                
                plt.plot(self.__support_AUC['fpr'][i], self.__support_AUC['tpr'][i], label=f'Clase {label} (AUC = {self.AUC_scores["perclass"][i]:.2f})')
        elif view == 'ovr':
            label = self.label.transform([letter])[0]
            plt.plot(self.__support_AUC['fpr'][label], self.__support_AUC['tpr'][label], label=f'OVR (AUC = {self.AUC_scores["ovr"]:.2f})')
        elif view == 'ovo':
            label = self.label.transform([letter])[0]
            plt.plot(self.__support_AUC['fpr'][label], self.__support_AUC['tpr'][label], label=f'OVO (AUC = {self.AUC_scores["ovo"]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curvas ROC')
        plt.legend(bbox_to_anchor = (1.25, 0.6), loc='center right')
        plt.show()
    
    @staticmethod
    def optimize_model(model, param_distributions, X_T, Y_T, n_iter=3):
        """
        Optimiza un modelo dado utilizando múltiples iteraciones de RandomizedSearchCV, seguido de un GridSearchCV.
        
        Parámetros:
            model: El modelo de machine learning que será optimizado.
            param_distributions: listas de valores para hiperparámetros.
            X_T: Características de datos de entrenamiento.
            Y_T: Categorías de datos de entrenamieno.
            n_iter: Número de iteraciones a RandomizedSearchCV para agregar resultados.
            
        Resultados:
            El mejor estimador de GridSearchCV.    
        """
        best_params_list = []
        
        for i in range(n_iter):
            # Randomized Search CV
            random_search = RandomizedSearchCV(estimator=model, 
                                            param_distributions=param_distributions,
                                            n_iter=10,
                                            cv=3, 
                                            random_state=i,
                                            n_jobs=-1,
                                            verbose=1)
            random_search.fit(X_T, Y_T)
            best_params_list.append(random_search.best_params_)
        
        # Agrega los mejores parámetros para crear una grilla para GridSearchCV
        param_grid = {}
        
        for param in param_distributions:
            values = [best_params[param] for best_params in best_params_list]
            unique_values = list(set(values))  # Quita duplicados
            
            if all(isinstance(value, (int, float)) for value in unique_values):
                min_value = min(unique_values)
                max_value = max(unique_values)
                if isinstance(min_value, int):
                    param_grid[param] = np.linspace(min_value, max_value + 1,num=5,dtype=int)
                else:
                    param_grid[param] = np.linspace(min_value, max_value, num=5).tolist()
            else:
                param_grid[param] = unique_values
        
        # Grid Search CV
        grid_search = GridSearchCV(estimator=model, 
                                param_grid=param_grid,
                                cv=2, 
                                n_jobs=-1,
                                verbose=1)
        grid_search.fit(X_T, Y_T)
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    @staticmethod
    def kwargs_load(keywords: dict,version=""):
        # keywords = {'tecnica': (graph,gradient,neural), 'modelo':(knn,rf,rn)}
        path =  '/content/drive/MyDrive/ml-processing/' + f'{keywords["tecnica"]}-processing/models'+ version +f'/{keywords["modelo"]}-model.pkl'
        with open(path,'rb') as file:
            model = pickle.load(file)
        return model
    
    @staticmethod
    def update_model(kwargs,version="",model_label='classificator'):
        old_model = model_trainer.kwargs_load(kwargs,version)
        new_model = model_trainer(**kwargs)
        
        if kwargs['modelo']=='knn':
            params = model_trainer.knn_params
        elif kwargs['modelo']=='rf':
            params = model_trainer.rf_params
        elif kwargs['modelo']=='ann':
            params = model_trainer.ann_params
        
        true_params = {key:value for key,value in eval(f'old_model.{model_label}.get_params().items()') if key in params}
        new_model.train_model(how='preset',**true_params)
        return new_model