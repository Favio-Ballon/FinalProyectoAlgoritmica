import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import prepararDatos

# Cargar el modelo entrenado
model = tf.keras.models.load_model('face_recognition_model.h5')

# Mapeo de índices de clase a nombres de usuarios
class_names = list(model.train_gen.class_indices.keys())

# Función para capturar y predecir la imagen
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "No se pudo acceder a la cámara.")
        return
    
    img = cv2.resize(frame, (100, 100))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    messagebox.showinfo("Resultado", f"Hola, {predicted_class}!")

# Función para registrarse
def register_user():
    name = simpledialog.askstring("Registro", "Ingrese su nombre:")
    if name:
        prepararDatos.capture_images(name)

# Interfaz gráfica
root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")

register_button = tk.Button(root, text="Registrarse", command=register_user)
register_button.pack(pady=20)

login_button = tk.Button(root, text="Iniciar Sesión", command=capture_and_predict)
login_button.pack(pady=20)

root.mainloop()