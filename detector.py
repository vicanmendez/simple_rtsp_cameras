# -*- coding: utf-8 -*- 
"""
Created on Thu Nov 14 22:58:22 2024

@author: vicmn
"""

import cv2
import tkinter as tk
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk  # Para convertir frames en imágenes compatibles con tkinter
import os
import datetime
import queue
import time
import numpy as np

# Función para leer el archivo de configuración
def leer_camaras(archivo):
    with open(archivo, 'r') as f:
        urls = f.read().strip().split(',')
    return urls

# Clase para manejar cada cámara
class Camara:
    def __init__(self, url, nombre, frame_padre, root, config=None):
        self.url = url
        self.nombre = nombre
        self.cap = cv2.VideoCapture(url)
        self.grabando = False
        self.background = None
        self.ultimo_movimiento = 0
        self.cooldown = 5  # segundos entre grabaciones
        self.frame_padre = frame_padre  # Frame donde se mostrará el video
        self.canvas = tk.Canvas(self.frame_padre, width=400, height=300, bg="black")
        self.canvas.pack()
        self.queue = queue.Queue(maxsize=10)  # Cola para frames
        self.running = True
        self.root = root  # Referencia al root de Tkinter
        self.photo = None
        self.image_item = None

        # Configuración de grabación
        self.config = config or {
            'modo_grabacion': 'motion',
            'duracion_grabacion': 15,
            'segmento_continuo': 5,
            'sensibilidad_movimiento': 100
        }

        # Variables para grabación continua
        self.ultimo_segmento = 0
        self.segmento_actual = None

    def iniciar(self):
        Thread(target=self.capturar_video, daemon=True).start()
        self.root.after(100, self.mostrar_video)  # Iniciar mostrar_video en el hilo principal

    def capturar_video(self):
        frame_count = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[Error] No se puede acceder al stream de {self.nombre}, intentando reconectar...")
                    self.cap.release()
                    time.sleep(1)  # Esperar antes de reconectar
                    self.cap = cv2.VideoCapture(self.url)
                    continue

                tiempo_actual = time.time()

                # Manejar grabación según el modo configurado
                if self.config['modo_grabacion'] == 'motion':
                    # Detectar movimiento y grabar
                    movimiento_detectado = self.detectar_movimiento(frame)
                    if movimiento_detectado and not self.grabando and (tiempo_actual - self.ultimo_movimiento) > self.cooldown:
                        self.ultimo_movimiento = tiempo_actual
                        self.iniciar_grabacion(frame)
                elif self.config['modo_grabacion'] == 'continuous':
                    # Verificar si es tiempo de crear un nuevo segmento
                    if tiempo_actual - self.ultimo_segmento >= self.config['segmento_continuo'] * 60:
                        if self.segmento_actual:
                            self.segmento_actual.release()
                        self.ultimo_segmento = tiempo_actual
                        self.iniciar_grabacion_continua(frame)
                    # Grabar el frame actual en el segmento continuo
                    self.grabar_frame_continuo(frame)
                # Para 'none', no hacer nada

                # Intentar poner el frame en la cola sin bloquear
                try:
                    self.queue.put_nowait(frame)
                except queue.Full:
                    # Si la cola está llena, descartar el frame más antiguo
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                # Pequeña pausa para no sobrecargar
                time.sleep(0.01)
            except Exception as e:
                print(f"[Error] Error en {self.nombre}: {e}, intentando reconectar...")
                try:
                    self.cap.release()
                except:
                    pass
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url)
                time.sleep(0.5)
                continue

    def mostrar_video(self):
        if not self.running:
            return

        try:
            frame = self.queue.get_nowait()
        except queue.Empty:
            # Si no hay frames disponibles, programar la próxima verificación
            self.root.after(10, self.mostrar_video)
            return

        # Redimensionar el frame al tamaño del canvas
        frame_resized = cv2.resize(frame, (400, 300))  # Dimensiones del Canvas
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image=frame_pil)

        # Mostrar el frame en el canvas
        if self.image_item is None:
            self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.image_item, image=self.photo)

        # Programar la próxima actualización
        self.root.after(10, self.mostrar_video)

    def detectar_movimiento(self, frame):
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (31, 31), 0)  # Mayor blur para reducir ruido

            # Inicializar el fondo si es necesario
            if self.background is None:
                self.background = gray.astype(np.float32)
                return False

            # Calcular la diferencia absoluta entre el fondo y el frame actual
            frame_delta = cv2.absdiff(self.background.astype(np.uint8), gray)
            thresh = cv2.threshold(frame_delta, self.config['sensibilidad_movimiento'], 255, cv2.THRESH_BINARY)[1]  # Usar sensibilidad configurada

            # Dilatar el umbral para llenar agujeros
            thresh = cv2.dilate(thresh, None, iterations=3)  # Más iteraciones para reducir ruido

            # Encontrar contornos
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Verificar si hay movimiento significativo (ignorar contornos muy pequeños)
            movimiento_significativo = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 5000:  # Área mínima más alta para ignorar cambios pequeños como ruido
                    continue
                # Verificar si el contorno no está en la parte inferior (posible área de timestamp)
                x, y, w, h = cv2.boundingRect(contour)
                if y > gray.shape[0] * 0.8:  # Ignorar contornos en el 20% inferior (posible timestamp)
                    continue
                movimiento_significativo = True
                break

            # Actualizar el fondo gradualmente para adaptarse a cambios de iluminación
            # Solo actualizar si no hay movimiento para evitar incluir objetos en movimiento en el fondo
            if not movimiento_significativo:
                cv2.accumulateWeighted(gray.astype(np.float32), self.background, 0.005)  # Actualización aún más lenta

            # Mostrar mensaje solo cuando se detecta movimiento
            if movimiento_significativo:
                print(f"[Movimiento] Detectado en {self.nombre}")

            return movimiento_significativo
        except Exception as e:
            print(f"[Error] Error en detectar_movimiento para {self.nombre}: {e}")
            return False

    def iniciar_grabacion(self, frame_inicial):
        if not self.grabando:
            self.grabando = True
            Thread(target=self.grabar_video, args=(frame_inicial,)).start()

    def iniciar_grabacion_continua(self, frame_inicial):
        if self.segmento_actual:
            self.segmento_actual.release()
        carpeta = f"./videos/{self.nombre}"
        os.makedirs(carpeta, exist_ok=True)
        fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_salida = f"{carpeta}/continuo_{fecha}.avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.segmento_actual = cv2.VideoWriter(archivo_salida, fourcc, 20.0, (frame_inicial.shape[1], frame_inicial.shape[0]))
        print(f"[Grabación] Iniciando segmento continuo en {self.nombre}: {archivo_salida}")

    def grabar_video(self, frame_inicial):
        carpeta = f"./videos/{self.nombre}"
        os.makedirs(carpeta, exist_ok=True)
        fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_salida = f"{carpeta}/{fecha}.avi"

        # Usar un VideoCapture separado para grabación
        cap_grabacion = cv2.VideoCapture(self.url)
        if not cap_grabacion.isOpened():
            print(f"[Error] No se pudo abrir el stream para grabación en {self.nombre}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(archivo_salida, fourcc, 20.0, (frame_inicial.shape[1], frame_inicial.shape[0]))

        # Grabar por la duración configurada
        contador_frames = 0
        max_frames = self.config['duracion_grabacion'] * 20  # 20 FPS
        while self.grabando and contador_frames < max_frames:
            try:
                ret, frame = cap_grabacion.read()
                if not ret:
                    break
                out.write(frame)
                contador_frames += 1
                time.sleep(0.01)  # Controlar la velocidad de grabación
            except cv2.error:
                print(f"[Error] Error de OpenCV al grabar {self.nombre}")
                break

        out.release()
        cap_grabacion.release()
        self.grabando = False
        print(f"[Grabación] Finalizada en {self.nombre}: {archivo_salida}")

    def grabar_frame_continuo(self, frame):
        if self.segmento_actual and self.config['modo_grabacion'] == 'continuous':
            try:
                self.segmento_actual.write(frame)
            except Exception as e:
                print(f"[Error] Error al escribir frame continuo en {self.nombre}: {e}")

# Clase principal para la interfaz de usuario
class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Reproductor RTSP con detección de movimiento")
        self.camaras = []
        self.root_window = root  # Guardar referencia al root
        self.urls = []  # Lista de URLs cargadas

        # Configuración de grabación
        self.modo_grabacion = tk.StringVar(value="motion")  # "motion", "continuous", "none"
        self.duracion_grabacion = tk.IntVar(value=15)  # segundos para motion
        self.segmento_continuo = tk.IntVar(value=5)  # minutos para continuous
        self.sensibilidad_movimiento = tk.IntVar(value=100)  # threshold para motion detection

        # Crear UI
        ttk.Label(self.root, text="Lista de cámaras:").pack(pady=5)
        self.lista_camaras = tk.Listbox(self.root, width=50, height=10)
        self.lista_camaras.pack(pady=5)

        # Botones para gestión de cámaras
        frame_botones = tk.Frame(self.root)
        frame_botones.pack(pady=5)
        ttk.Button(frame_botones, text="Cargar cámaras", command=self.cargar_camaras).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Agregar cámara", command=self.agregar_camara).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Editar cámara", command=self.editar_camara).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Eliminar cámara", command=self.eliminar_camara).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Guardar cambios", command=self.guardar_camaras).pack(side=tk.LEFT, padx=5)

        # Configuración de grabación
        frame_config = tk.LabelFrame(self.root, text="Configuración de Grabación", padx=10, pady=5)
        frame_config.pack(pady=10, fill="x")

        # Modo de grabación
        ttk.Label(frame_config, text="Modo de grabación:").grid(row=0, column=0, sticky="w", pady=2)
        modos_frame = tk.Frame(frame_config)
        modos_frame.grid(row=0, column=1, sticky="w", pady=2)
        ttk.Radiobutton(modos_frame, text="Por movimiento", variable=self.modo_grabacion, value="motion").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(modos_frame, text="Continua", variable=self.modo_grabacion, value="continuous").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(modos_frame, text="Sin grabación", variable=self.modo_grabacion, value="none").pack(side=tk.LEFT, padx=5)

        # Duración para modo movimiento
        ttk.Label(frame_config, text="Duración grabación (seg):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame_config, from_=5, to=300, textvariable=self.duracion_grabacion, width=10).grid(row=1, column=1, sticky="w", pady=2)

        # Segmento para modo continuo
        ttk.Label(frame_config, text="Segmento continuo (min):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame_config, from_=1, to=60, textvariable=self.segmento_continuo, width=10).grid(row=2, column=1, sticky="w", pady=2)

        # Sensibilidad del movimiento
        ttk.Label(frame_config, text="Sensibilidad movimiento:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame_config, from_=50, to=255, textvariable=self.sensibilidad_movimiento, width=10).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Button(self.root, text="Iniciar cámaras", command=self.iniciar_camaras).pack(pady=5)

        # Frame para mostrar los videos
        self.frame_videos = tk.Frame(self.root)
        self.frame_videos.pack(pady=10)

    def cargar_camaras(self):
        try:
            self.urls = leer_camaras("camaras.txt")
        except FileNotFoundError:
            self.urls = []
        self.actualizar_lista_camaras()

    def actualizar_lista_camaras(self):
        self.lista_camaras.delete(0, tk.END)
        for i, url in enumerate(self.urls):
            self.lista_camaras.insert(tk.END, f"Cámara {i+1}: {url}")

    def agregar_camara(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Agregar nueva cámara")
        dialog.geometry("400x150")

        ttk.Label(dialog, text="URL RTSP:").pack(pady=5)
        url_entry = ttk.Entry(dialog, width=50)
        url_entry.pack(pady=5)

        def guardar():
            url = url_entry.get().strip()
            if url:
                self.urls.append(url)
                self.actualizar_lista_camaras()
                dialog.destroy()

        ttk.Button(dialog, text="Agregar", command=guardar).pack(pady=10)

    def editar_camara(self):
        seleccion = self.lista_camaras.curselection()
        if not seleccion:
            return
        indice = seleccion[0]

        dialog = tk.Toplevel(self.root)
        dialog.title("Editar cámara")
        dialog.geometry("400x150")

        ttk.Label(dialog, text="URL RTSP:").pack(pady=5)
        url_entry = ttk.Entry(dialog, width=50)
        url_entry.insert(0, self.urls[indice])
        url_entry.pack(pady=5)

        def guardar():
            url = url_entry.get().strip()
            if url:
                self.urls[indice] = url
                self.actualizar_lista_camaras()
                dialog.destroy()

        ttk.Button(dialog, text="Guardar", command=guardar).pack(pady=10)

    def eliminar_camara(self):
        seleccion = self.lista_camaras.curselection()
        if not seleccion:
            return
        indice = seleccion[0]
        del self.urls[indice]
        self.actualizar_lista_camaras()

    def guardar_camaras(self):
        with open("camaras.txt", 'w') as f:
            f.write(','.join(self.urls))
        print("[Info] Cámaras guardadas en camaras.txt")

    def iniciar_camaras(self):
        # Limpiar frames anteriores
        for widget in self.frame_videos.winfo_children():
            widget.destroy()
        self.camaras = []

        # Configurar parámetros según el modo seleccionado
        config = {
            'modo_grabacion': self.modo_grabacion.get(),
            'duracion_grabacion': self.duracion_grabacion.get(),
            'segmento_continuo': self.segmento_continuo.get(),
            'sensibilidad_movimiento': self.sensibilidad_movimiento.get()
        }

        for i, url in enumerate(self.urls):
            frame_camara = tk.Frame(self.frame_videos, bd=2, relief=tk.SUNKEN)
            frame_camara.grid(row=i // 2, column=i % 2, padx=5, pady=5)
            camara = Camara(url, f"Cámara_{i+1}", frame_camara, self.root_window, config)
            # Bind double-click to open full screen
            frame_camara.bind("<Double-Button-1>", lambda e, c=camara: self.abrir_ventana_completa(c))
            camara.canvas.bind("<Double-Button-1>", lambda e, c=camara: self.abrir_ventana_completa(c))
            self.camaras.append(camara)
            camara.iniciar()

    def detener_camaras(self):
        for camara in self.camaras:
            camara.running = False
            if camara.cap.isOpened():
                camara.cap.release()
            # Cerrar grabación continua si está activa
            if hasattr(camara, 'segmento_actual') and camara.segmento_actual:
                camara.segmento_actual.release()

    def abrir_ventana_completa(self, camara):
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Stream completo - {camara.nombre}")
        ventana.geometry("800x600")

        canvas = tk.Canvas(ventana, width=800, height=600, bg="black")
        canvas.pack()

        # Crear una nueva instancia de Camara para la ventana completa
        camara_completa = CamaraCompleta(camara.url, camara.nombre, canvas, ventana)
        camara_completa.iniciar()

# Clase para mostrar stream en ventana completa
class CamaraCompleta:
    def __init__(self, url, nombre, canvas, ventana):
        self.url = url
        self.nombre = nombre
        self.cap = cv2.VideoCapture(url)
        self.canvas = canvas
        self.ventana = ventana
        self.queue = queue.Queue(maxsize=10)
        self.running = True
        self.photo = None
        self.image_item = None

    def iniciar(self):
        Thread(target=self.capturar_video, daemon=True).start()
        self.ventana.after(100, self.mostrar_video)

    def capturar_video(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[Error] No se puede acceder al stream de {self.nombre}")
                    time.sleep(1)
                    continue

                try:
                    self.queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                time.sleep(0.01)
            except Exception as e:
                print(f"[Error] Error en {self.nombre}: {e}")
                time.sleep(1)

    def mostrar_video(self):
        if not self.running:
            return

        try:
            frame = self.queue.get_nowait()
        except queue.Empty:
            self.ventana.after(10, self.mostrar_video)
            return

        # Redimensionar al tamaño del canvas
        frame_resized = cv2.resize(frame, (800, 600))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image=frame_pil)

        if self.image_item is None:
            self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.image_item, image=self.photo)

        self.ventana.after(10, self.mostrar_video)

# Crear carpeta de videos si no existe
os.makedirs("./videos", exist_ok=True)

# Ejecutar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.detener_camaras(), root.destroy()))
    root.mainloop()
