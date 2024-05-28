from multiprocessing import Lock
import pandas as pd
import numpy as np
import os
import torch
import functools

# --------------
# Funciones generales

def dataset_exists():
    if os.path.exists(venv + '\synthetic-asl-alphabet'):
        pass
    else:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet",)

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


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths

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

# --------------
# Transformación de imágenes

def mediapipe_landmarks(image_path, vis=False, app=False):
    import cv2
    import mediapipe as mp
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Iniciar el framework de reconocimiento de coordenadas
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Leer y procesar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    coords = np.empty((0, 2))

    # Si hay resultados para las imágenes
    if results.multi_hand_landmarks:
        # Recuperar nodos de la imagen
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extraer las coordenadas y guardarlas en lista
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                coords = np.vstack((coords, np.array([cx, cy]).reshape(1, -1)))
            mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        coords = np.zeros((21, 2))

    hands.close()

    # Crea un dataframe de una sola fila con todas las coordenadas

    if vis:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')  # Ocultar los ejes
        plt.show()

    if app:
        coords = [image_path.split('\\')[-2]] + scaler.fit_transform(coords).flatten().tolist()

        columns = ['letra']
        for i in range(21):
            columns.extend([f"x_{i}", f"y_{i}"])

        ruta = '\graph-processing\{}.csv'.format(image_path.split("\\")[-3])
        append_csv(venv + ruta, pd.DataFrame([coords], columns=columns))



# ------------------------
# Variables generales

venv = os.path.dirname((os.path.abspath(__file__)))
train = get_file_paths(venv + r'\synthetic-asl-alphabet\Train_Alphabet')
test = get_file_paths(venv + r'\synthetic-asl-alphabet\Test_Alphabet')
all = [*train, *test]