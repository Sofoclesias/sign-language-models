import cv2
import numpy as np
import tkinter as tk
import QoL as qol
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

cap = cv2.VideoCapture(0)
keywords = [{'tecnica': tec, 'modelo':mod} for tec in ['graph','gradient'] for mod in ['knn','rf']]

# Cargar modelos entrenados
models = {
    'mediapipe': {
        'knn': qol.load_model(keywords[0]),
        'rf': qol.load_model(keywords[1]),
    },
    'hog': {
        'knn': qol.load_model(keywords[2]),
        'rf': qol.load_model(keywords[3]),
    }
}
current_models = models['mediapipe']

def update_models(selected_technique):
    global current_models
    current_models = models[selected_technique]

# Función para procesar el frame de la cámara
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Extrae características según la técnica seleccionada
    if technique_var=='mediapipe':
        ins = qol.mediapipe_landmarks(frame)
        features = ins.extract_values(normalize=True)
    elif technique_var=='hog':
        ins = qol.hog_transform(frame)
        features = ins.extract_values(normalize=True)
    
    # Predicciones con los modelos
    knn_prediction = current_models['knn'].predict([features])[0]
    rf_prediction = current_models['rf'].predict([features])[0]
    knn_label.config(text=knn_prediction)
    rf_label.config(text=rf_prediction)

    # Mostrar el frame con el filtro de la técnica en la UI
    img = Image.fromarray(ins.image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    # Llamar a esta función nuevamente después de 10 ms
    camera_label.after(10, process_frame)

# ----------------------- 
# UI
root = tk.Tk()
root.title("Alfabeto de señas")

# Label para cámara
camera_label = tk.Label(root)
camera_label.grid(row=0, column=0, rowspan=4)

# Dropdown para técnica de representación
technique_var = tk.StringVar(value='mediapipe')
technique_dropdown = ttk.Combobox(root, textvariable=technique_var)
technique_dropdown['values'] = ('mediapipe', 'hog')
technique_dropdown.grid(row=0, column=1)
technique_dropdown.bind('<<ComboboxSelected>>', lambda e: update_models(technique_var.get()))

# Labels para mostrar las predicciones
tk.Label(root, text="KNN", font=("Helvetica", 14)).grid(row=1, column=1, sticky='W')
tk.Label(root, text="RF", font=("Helvetica", 14)).grid(row=2, column=1, sticky='W')

knn_label = tk.Label(root, text="", font=("Helvetica", 24))
knn_label.grid(row=1, column=2, sticky='W')
rf_label = tk.Label(root, text="", font=("Helvetica", 24))
rf_label.grid(row=2, column=2, sticky='W')

# Empezar a procesar frames de la cámara
process_frame()
root.mainloop()

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()