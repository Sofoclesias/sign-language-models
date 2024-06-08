from roboflow import Roboflow
import QoL as qol
from rembg import remove
import numpy as np
import cv2
import multiprocessing as mp
import os
import time
import random
from PIL import Image, ImageOps

# Configuraciones
'''
En un primer procesamiento general, se descubrió que las
letras listadas abajo tienen un comportamiento peculiar
con el tratamiento del contorno de opencv. Por ello, han
sido separadas para ser manejadas de forma independiente.

upframe: imágenes a las que se les debe ajustar el marco 
         de captura hacia arriba.
downframe: * hacia abajo.
whole: opencv no funciona en este. Se utilizará un modelo
         de reconocimiento de objetos para apoyar al 
         preprocesamiento.
banned: solo está Blank, porque no debería capturar nada.
'''
upframe = ['B','C','I','R','U','V']
downframe = ['J','X','Y']
whole = ['P','Q']
banned = ['Blank']

# Para el preprocesamiento general

rf = Roboflow(api_key="YU30pOFC9cA5yqoJUQij")
project = rf.workspace().project("hands-rzvws")
model = project.version(1).model


def save_image (path,image):
    parts = path.split(os.sep)
    type_dir,letter_dir = parts[-3],parts[-2]
    
    output_dir = os.path.join(qol.venv,'preprocessed-images',type_dir,letter_dir)
    os.makedirs(output_dir,exist_ok=True)
    i = len(os.listdir(output_dir)) + 1
    
    output_path = os.path.join(output_dir,f'{i}.png')
    cv2.imwrite(output_path,image)

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

def preprocess(path):
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
                x,y,h = cuadrar(*modelled_bbox)
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
        save_image(path,image)
    except:
        pass

# Para imágenes de fondo negro   
def blank_generate():
    width, height = 410, 410
    color = (0, 0, 0)
    img = Image.new('RGB', (width, height), color)

    for i in range(900):
        img.save(os.path.join(qol.venv,'preprocessed-images','Train_Alphabet',f'{i+1}.png'))
        
    for i in range(100):
        img.save(os.path.join(qol.venv,'preprocessed-images','Test_Alphabet',f'{i+1}.png')) 
        
        
# Para el aumento de imágenes
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

def augmentation_en_carpetas(ruta_principal,N):
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

# Ejecución en archivo      
if __name__=='__main__':
    PHASE1 = True # Preprocesamiento general (ahora ajustado a los que se deben repreproocesar)
    PHASE2 = True # Generación de 1000 imágenes de fondo negro
    PHASE3 = False # Image augmentation
    
    _,_,todo = qol.retrieve_raw_paths()
    real = [path for path in todo if path.split(os.sep)[-2] in whole or path.split(os.sep)[-2] in upframe or path.split(os.sep)[-2] in downframe]
    not_blank = [path for path in todo if path.split(os.sep)[-2] not in banned]
    
    if PHASE1:
        pass
    else:  
        print('Fase 1: Preprocesamiento inicial.')
        with mp.Pool() as pool:
            pool.map(preprocess,real)
            pool.close()
            pool.join()
            
        time.sleep(2)
    
    if PHASE2:
        pass
    else:
        print('Fase 2: Generación de 1000 imágenes de fondo negro.')
        blank_generate()
        time.sleep(2)
    
    if PHASE3:
        pass
    else:
        print('Fase 3: Incremento de imágenes')
        processes = [mp.Process(target=augmentation_en_carpetas,args=(os.path.join(qol.venv,'preprocessed-images','Train_Alphabet'),900)),
                     mp.Process(target=augmentation_en_carpetas,args=(os.path.join(qol.venv,'preprocessed-images','Test_Alphabet'),100))]
        
        [p.start() for p in processes]
        [p.join() for p in processes]
        time.sleep(2)
        
    print('Preprocesamiento acabado.')