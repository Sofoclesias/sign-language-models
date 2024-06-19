import QoL_colab as qol
import multiprocessing as mp
import os
import time

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

# Ejecución en archivo      
if __name__=='__main__':
    PHASE1 = True # Preprocesamiento general (ahora ajustado a los que se deben repreproocesar)
    PHASE2 = False # Image augmentation
    
    _,_,todo = qol.retrieve_raw_paths()
    real = [path for path in todo if path.split(os.sep)[-2] in whole or path.split(os.sep)[-2] in upframe or path.split(os.sep)[-2] in downframe]
    not_blank = [path for path in todo if path.split(os.sep)[-2] not in banned]
    
    if PHASE1:
        pass
    else:  
        print('Fase 1: Preprocesamiento inicial.')
        with mp.Pool() as pool:
            pool.map(qol.image_preprocessing.main_preprocess,real)
            pool.close()
            pool.join()
            
        time.sleep(2)
    
    if PHASE2:
        pass
    else:
        print('Fase 3: Incremento de imágenes')
        processes = [mp.Process(target=qol.image_preprocessing.augmentation_en_carpetas,args=(os.path.join(qol.venv,'preprocessed-images','Train_Alphabet'),900)),
                     mp.Process(target=qol.image_preprocessing.augmentation_en_carpetas,args=(os.path.join(qol.venv,'preprocessed-images','Test_Alphabet'),100))]
        
        [p.start() for p in processes]
        [p.join() for p in processes]
        time.sleep(2)
        
    print('Preprocesamiento acabado.')