import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Label
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image, ImageTk


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
def register_user():
    name = simpledialog.askstring("Registro", "Ingrese su nombre:")
    if name:
        confirm_capture = messagebox.askyesno("Registro", "¿Desea abrir la cámara para capturar imágenes?")
        if confirm_capture:
                capture_images(name)
        else:
            messagebox.showinfo("Registro", "Puede registrar imágenes más tarde desde la pantalla principal.")


# Configuración del generador de datos
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    'fotos',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    'fotos',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Construcción del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_gen, epochs=10, validation_data=val_gen)

# Guardar el modelo
model.save('face_recognition_model.h5')

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
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    confidence_threshold = 0.75  # Umbral de confianza del 75%

    if confidence < confidence_threshold:
        messagebox.showerror("Error", "Usuario no registrado o confianza insuficiente.")
    else:
        predicted_class = class_names[predicted_class_index]
        show_greeting_screen(predicted_class)

# Función para mostrar la pantalla de saludo
def show_greeting_screen(user_name):
    for widget in root.winfo_children():
        widget.destroy()

    greeting_label = tk.Label(root, text=f"Hola, {user_name}!", font=("Arial", 24), bg="lightblue", fg="darkblue")
    greeting_label.pack(pady=20)

    back_button = tk.Button(root, text="Volver", command=show_main_screen, font=("Arial", 14), bg="darkblue", fg="white", width=15)
    back_button.pack(pady=10)

# Función para mostrar la pantalla principal
def show_main_screen():
    for widget in root.winfo_children():
        widget.destroy()

    title_label = tk.Label(root, text="Sistema de Reconocimiento Facial", font=("Arial", 18), bg="lightblue", fg="darkblue")
    title_label.pack(pady=20)

    register_button = tk.Button(root, text="Registrarse", command=register_user, font=("Arial", 14), bg="darkblue", fg="white", width=15)
    register_button.pack(pady=10)

    login_button = tk.Button(root, text="Iniciar Sesión", command=capture_and_predict, font=("Arial", 14), bg="darkblue", fg="white", width=15)
    login_button.pack(pady=10)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('face_recognition_model.h5')

# Mapeo de índices de clase a nombres de usuarios
class_names = list(train_gen.class_indices.keys())

# Interfaz gráfica
root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
root.geometry("500x300")
root.configure(bg="lightblue")

show_main_screen()

root.mainloop()