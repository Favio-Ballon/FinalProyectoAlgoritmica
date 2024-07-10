import os
import cv2
import tkinter as tk
from tkinter import simpledialog

def capture_images(user_name):
    os.makedirs(f'fotos/{user_name}', exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Capturing', frame)
        cv2.imwrite(f'fotos/{user_name}/{count}.jpg', frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Interfaz para capturar imágenes
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal
name = simpledialog.askstring("Registro", "Ingrese su nombre:")
if name:
    capture_images(name)
