import cv2
import mediapipe as mp
import pandas as pd
import os
import functools
import datetime
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

partir = int(multiprocessing.cpu_count())
venv = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_hand_landmarks(image_path):
    # Iniciar el framework de reconocimiento de coordenadas
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    # Leer y procesar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    hand_landmarks_data = [image_path.split('\\')[-2]]

    # Si hay resultados para las imágenes
    if results.multi_hand_landmarks:
        # Recuperar nodos de la imagen
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extraer las coordenadas y guardarlas en lista
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hand_landmarks_data.extend([cx, cy])
    else:
        hand_landmarks_data += [0]*42

    hands.close()

    # Crea un dataframe de una sola fila con todas las coordenadas
    columns = ['letra']
    for i in range(21):
        columns.extend([f"x_{i}", f"y_{i}"])
    landmarks_df = pd.DataFrame([hand_landmarks_data], columns=columns)
    ruta = '\graph-processing\processed-data\{}.csv'.format(image_path.split("\\")[-3])

    append_csv(venv + ruta, landmarks_df)

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


@locking(multiprocessing.Lock())
def append_csv(ruta: str, df: pd.DataFrame):
    try:
        df.to_csv(ruta, index=True, mode='a', header=not os.path.exists(ruta))
    except: pass

if __name__=='__main__':
    # Chequear si el dataset está instalado
    if os.path.exists(venv + '\synthetic-asl-alphabet'):
        pass
    else:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet")

    train = get_file_paths(venv + r'\synthetic-asl-alphabet\Train_Alphabet')
    test = get_file_paths(venv + r'\synthetic-asl-alphabet\Test_Alphabet')
    all = [*train, *test]

    print('train: ',len(train))
    print('test: ',len(test))

    print('Procesamiento iniciado - ',datetime.datetime.today())

    with multiprocessing.Pool(processes=partir) as pool:
        pool.map(get_hand_landmarks, all)
        pool.close()
        pool.join()

    print('Procesamiento terminado - ',datetime.datetime.today())