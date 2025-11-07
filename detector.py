# -*- coding: utf-8 -*- 
"""
Created on Thu Nov 14 22:58:22 2024

@author: vicmn
"""

import cv2
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Style
from tkinter import messagebox, filedialog
from threading import Thread, Lock
from PIL import Image, ImageTk  # Para convertir frames en imágenes compatibles con tkinter
import os
import datetime
import queue
import time
import numpy as np
import ffmpeg
import subprocess
import tempfile
import ftplib
import json
import schedule
import logging
from collections import deque

# Configuración de estilo moderno con ttkbootstrap
STYLE_CONFIG = {
    'font_family': 'Segoe UI',
    'font_size_normal': 10,
    'font_size_title': 12,
    'font_size_small': 9,
    'theme': 'litera'  # Tema moderno de ttkbootstrap
}

# Configuración global para FTP
FTP_QUEUE = deque()
FTP_QUEUE_LOCK = Lock()
FTP_RETRY_DELAY = 120  # 2 minutos en segundos

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

        # Variables para audio
        self.audio_process = None
        self.audio_temp_file = None
        self.has_audio = None  # Cache para saber si la cámara tiene audio

    def iniciar(self):
        Thread(target=self.capturar_video, daemon=True).start()
        self.root.after(100, self.mostrar_video)  # Iniciar mostrar_video en el hilo principal

    def capturar_video(self):
        frame_count = 0
        ultimo_refresh = time.time()
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

                # Refresh periódico del stream para evitar congelamiento (cada 30 minutos)
                if tiempo_actual - ultimo_refresh > 1800:  # 30 minutos
                    print(f"[Stream] Refrescando stream de {self.nombre}")
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(self.url)
                    ultimo_refresh = tiempo_actual
                    continue  # Saltar al siguiente ciclo después del refresh

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
            self.iniciar_grabacion_audio()

    def iniciar_grabacion_continua(self, frame_inicial):
        # Detener grabación de audio anterior si existe
        if self.audio_process:
            self.detener_grabacion_audio()

        if self.segmento_actual:
            self.segmento_actual.release()
            # Combinar audio con el segmento anterior si existe
            if hasattr(self, 'ultimo_video_segmento') and self.ultimo_video_segmento:
                self.combinar_audio_video(self.ultimo_video_segmento)
                # Subir a FTP si está configurado
                self.subir_a_ftp(self.ultimo_video_segmento)

        # Usar directorio configurado
        directorio_base = self.config.get('directorio_videos', './videos')
        carpeta = f"{directorio_base}/{self.nombre}"
        os.makedirs(carpeta, exist_ok=True)
        fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_salida = f"{carpeta}/continuo_{fecha}.avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.segmento_actual = cv2.VideoWriter(archivo_salida, fourcc, 20.0, (frame_inicial.shape[1], frame_inicial.shape[0]))
        self.ultimo_video_segmento = archivo_salida  # Guardar para combinar con audio después
        print(f"[Grabación] Iniciando segmento continuo en {self.nombre}: {archivo_salida}")
        self.iniciar_grabacion_audio_continua()

    def grabar_video(self, frame_inicial):
        # Usar directorio configurado
        directorio_base = self.config.get('directorio_videos', './videos')
        carpeta = f"{directorio_base}/{self.nombre}"
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
        self.detener_grabacion_audio()
        self.combinar_audio_video(archivo_salida)
        # Subir a FTP si está configurado
        self.subir_a_ftp(archivo_salida)
        self.grabando = False
        print(f"[Grabación] Finalizada en {self.nombre}: {archivo_salida}")

    def grabar_frame_continuo(self, frame):
        if self.segmento_actual and self.config['modo_grabacion'] == 'continuous':
            try:
                self.segmento_actual.write(frame)
            except Exception as e:
                print(f"[Error] Error al escribir frame continuo en {self.nombre}: {e}")

    def detectar_audio(self):
        """Detecta si la cámara tiene audio disponible."""
        if self.has_audio is not None:
            return self.has_audio

        try:
            # Verificar si ffmpeg está disponible
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[Audio] FFmpeg no está instalado. No se puede detectar audio en {self.nombre}")
            self.has_audio = False
            return False

        try:
            # Intentar obtener información del stream con ffmpeg
            cmd = ['ffmpeg', '-i', self.url, '-f', 'null', '-']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Si ffmpeg puede acceder al stream sin errores de audio, asumimos que tiene audio
            if 'Audio:' in result.stderr or 'audio:' in result.stderr.lower():
                self.has_audio = True
                print(f"[Audio] Cámara {self.nombre} tiene audio disponible")
                return True
            else:
                self.has_audio = False
                print(f"[Audio] Cámara {self.nombre} no tiene audio")
                return False
        except subprocess.TimeoutExpired:
            print(f"[Audio] Timeout al detectar audio en {self.nombre}")
            self.has_audio = False
            return False
        except Exception as e:
            print(f"[Audio] Error al detectar audio en {self.nombre}: {e}")
            self.has_audio = False
            return False

    def iniciar_grabacion_audio(self):
        try:
            # Verificar si la cámara tiene audio
            if not self.detectar_audio():
                print(f"[Audio] Omitiendo grabación de audio en {self.nombre} (sin audio)")
                self.audio_process = None
                return

            # Usar directorio configurado
            directorio_base = self.config.get('directorio_videos', './videos')
            carpeta = f"{directorio_base}/{self.nombre}"
            os.makedirs(carpeta, exist_ok=True)
            fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.audio_temp_file = f"{carpeta}/temp_audio_{fecha}.aac"

            # Usar ffmpeg para capturar audio del stream RTSP
            cmd = [
                'ffmpeg', '-y', '-i', self.url, '-vn', '-acodec', 'aac',
                '-b:a', '128k', '-f', 'adts', self.audio_temp_file
            ]

            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Audio] Iniciando grabación de audio en {self.nombre}")
        except Exception as e:
            print(f"[Error] No se pudo iniciar grabación de audio en {self.nombre}: {e}")
            self.audio_process = None

    def iniciar_grabacion_audio_continua(self):
        try:
            # Verificar si la cámara tiene audio
            if not self.detectar_audio():
                print(f"[Audio] Omitiendo grabación continua de audio en {self.nombre} (sin audio)")
                self.audio_process = None
                return

            # Usar directorio configurado
            directorio_base = self.config.get('directorio_videos', './videos')
            carpeta = f"{directorio_base}/{self.nombre}"
            os.makedirs(carpeta, exist_ok=True)
            fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.audio_temp_file = f"{carpeta}/temp_audio_continuo_{fecha}.aac"

            cmd = [
                'ffmpeg', '-y', '-i', self.url, '-vn', '-acodec', 'aac',
                '-b:a', '128k', '-f', 'adts', self.audio_temp_file
            ]

            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Audio] Iniciando grabación continua de audio en {self.nombre}")
        except Exception as e:
            print(f"[Error] No se pudo iniciar grabación continua de audio en {self.nombre}: {e}")
            self.audio_process = None

    def detener_grabacion_audio(self):
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=5)
                print(f"[Audio] Grabación de audio detenida")
            except subprocess.TimeoutExpired:
                print(f"[Audio] Timeout al detener grabación de audio, forzando cierre")
                try:
                    self.audio_process.kill()
                    self.audio_process.wait(timeout=2)
                except:
                    pass
            except Exception as e:
                print(f"[Error] Error al detener grabación de audio: {e}")
                try:
                    self.audio_process.kill()
                except:
                    pass
            self.audio_process = None

    def combinar_audio_video(self, video_file):
        if not self.audio_temp_file or not os.path.exists(self.audio_temp_file):
            print(f"[Audio] No hay archivo de audio para combinar con {video_file}")
            return

        try:
            # Verificar si el archivo de audio tiene contenido
            if os.path.getsize(self.audio_temp_file) == 0:
                print(f"[Audio] Archivo de audio vacío, guardando solo video: {video_file}")
                try:
                    os.remove(self.audio_temp_file)
                except:
                    pass
                self.audio_temp_file = None
                return

            # Crear archivo de salida con audio
            video_con_audio = video_file.replace('.avi', '_con_audio.mp4')

            # Usar ffmpeg para combinar video y audio
            cmd = [
                'ffmpeg', '-y', '-i', video_file, '-i', self.audio_temp_file,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', video_con_audio
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Si la combinación fue exitosa, reemplazar el archivo original
                os.replace(video_con_audio, video_file)
                print(f"[Audio] Audio combinado exitosamente en {video_file}")
            else:
                print(f"[Error] Error al combinar audio: {result.stderr}")
                # Mantener el archivo de video original si falla la combinación

            # Limpiar archivo temporal de audio
            try:
                os.remove(self.audio_temp_file)
            except:
                pass

        except Exception as e:
            print(f"[Error] Error al combinar audio y video: {e}")
        finally:
            self.audio_temp_file = None

    def subir_a_ftp(self, archivo_video):
        """Sube un archivo de video a un servidor FTP si está configurado."""
        ftp_config = self.config.get('ftp_config', {})
        if not ftp_config.get('host') or not ftp_config.get('user') or not ftp_config.get('password'):
            return  # FTP no configurado

        # Agregar a la cola de FTP con timestamp de creación
        with FTP_QUEUE_LOCK:
            FTP_QUEUE.append({
                'archivo': archivo_video,
                'timestamp': time.time(),
                'intentos': 0,
                'camara': self.nombre,
                'ftp_config': ftp_config.copy()
            })

        # Iniciar el procesamiento de la cola si no está corriendo
        Thread(target=self.procesar_cola_ftp, daemon=True).start()
        print(f"[FTP] Archivo {os.path.basename(archivo_video)} agregado a la cola de subida")

    def procesar_cola_ftp(self):
        """Procesa la cola de subida FTP con reintentos."""
        while True:
            try:
                with FTP_QUEUE_LOCK:
                    if not FTP_QUEUE:
                        break
                    tarea = FTP_QUEUE.popleft()

                archivo = tarea['archivo']
                timestamp = tarea['timestamp']
                intentos = tarea['intentos']
                camara = tarea['camara']
                ftp_config = tarea['ftp_config']

                # Verificar si el archivo aún existe
                if not os.path.exists(archivo):
                    print(f"[FTP] Archivo {os.path.basename(archivo)} ya no existe, omitiendo")
                    continue

                try:
                    with ftplib.FTP(ftp_config['host']) as ftp:
                        ftp.login(ftp_config['user'], ftp_config['password'])
                        remote_path = ftp_config.get('remote_path', '/videos')

                        # Crear directorio remoto si no existe
                        try:
                            ftp.mkd(remote_path)
                        except ftplib.error_perm:
                            pass  # El directorio ya existe

                        # Cambiar al directorio remoto
                        ftp.cwd(remote_path)

                        # Crear subdirectorio para la cámara si no existe
                        try:
                            ftp.mkd(camara)
                        except ftplib.error_perm:
                            pass
                        ftp.cwd(camara)

                        # Subir el archivo
                        with open(archivo, 'rb') as file:
                            filename = os.path.basename(archivo)
                            ftp.storbinary(f'STOR {filename}', file)

                        print(f"[FTP] Archivo {filename} subido exitosamente a {ftp_config['host']}")
                        # Log en la interfaz si existe
                        if hasattr(self, 'app') and hasattr(self.app, 'log_ftp'):
                            self.app.log_ftp(f"✅ Archivo {filename} enviado por FTP")

                except Exception as e:
                    intentos += 1
                    if intentos < 3:  # Máximo 3 intentos
                        print(f"[FTP] Error al subir {os.path.basename(archivo)} (intento {intentos}/3): {e}")
                        # Re-agregar a la cola con delay
                        with FTP_QUEUE_LOCK:
                            FTP_QUEUE.append({
                                'archivo': archivo,
                                'timestamp': timestamp,
                                'intentos': intentos,
                                'camara': camara,
                                'ftp_config': ftp_config
                            })
                        time.sleep(FTP_RETRY_DELAY)
                    else:
                        print(f"[FTP] Fallaron todos los intentos para {os.path.basename(archivo)}")
                        # Log en la interfaz si existe
                        if hasattr(self, 'app') and hasattr(self.app, 'log_ftp'):
                            self.app.log_ftp(f"❌ Error al transmitir {os.path.basename(archivo)} por FTP")

            except Exception as e:
                print(f"[FTP] Error en procesamiento de cola: {e}")
                time.sleep(1)

# Clase principal para la interfaz de usuario
class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Vigilancia RTSP")
        self.camaras = []
        self.root_window = root  # Guardar referencia al root
        self.urls = []  # Lista de URLs cargadas
        self.nombres_camaras = []  # Lista de nombres personalizados

        # Configurar estilo moderno con ttkbootstrap
        self.style = Style(theme=STYLE_CONFIG['theme'])
        self.setup_style()

        # Configurar ventana principal
        self.setup_window()

        # Configuración de grabación
        self.modo_grabacion = tk.StringVar(value="motion")  # "motion", "continuous", "none"
        self.duracion_grabacion = tk.IntVar(value=15)  # segundos para motion
        self.segmento_continuo = tk.IntVar(value=5)  # minutos para continuous
        self.sensibilidad_movimiento = tk.IntVar(value=100)  # threshold para motion detection

        # Configuración adicional
        self.directorio_videos = tk.StringVar(value="./videos")
        self.ftp_config = {
            'host': '',
            'user': '',
            'password': '',
            'remote_path': '/videos'
        }

        # Crear barra de menú
        self.crear_menu()

        # Crear interfaz principal (solo videos, sin lista de cámaras visible)
        self.crear_interfaz_principal()

        # Iniciar scheduler para limpieza diaria
        self.iniciar_scheduler_limpieza()

        # Cargar configuración y cámaras al inicio
        self.cargar_configuracion()
        self.cargar_camaras()

    def iniciar_scheduler_limpieza(self):
        """Inicia el scheduler para limpieza diaria de archivos FTP."""
        def limpieza_diaria():
            Thread(target=self.limpiar_archivos_ftp_antiguos, daemon=True).start()

        # Programar limpieza diaria a las 00:00
        schedule.every().day.at("00:00").do(limpieza_diaria)

        # Ejecutar scheduler en un hilo separado
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Revisar cada minuto

        Thread(target=run_scheduler, daemon=True).start()
        print("[Scheduler] Limpieza diaria programada para las 00:00")

    def log_ftp(self, mensaje):
        """Agrega un mensaje al panel de logs FTP."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        mensaje_completo = f"[{timestamp}] {mensaje}\n"

        # Agregar al text widget desde el hilo principal
        def update_log():
            self.log_text.insert(tk.END, mensaje_completo)
            self.log_text.see(tk.END)  # Auto-scroll al final
            # Limitar a 100 líneas para evitar que crezca demasiado
            if int(self.log_text.index('end-1c').split('.')[0]) > 100:
                self.log_text.delete('1.0', '2.0')

        self.root.after(0, update_log)

    def limpiar_archivos_ftp_antiguos(self):
        """Limpia archivos remotos FTP que tengan más de 96 horas."""
        ftp_config = self.ftp_config
        if not ftp_config.get('host') or not ftp_config.get('user') or not ftp_config.get('password'):
            print("[FTP] FTP no configurado, omitiendo limpieza")
            return

        try:
            with ftplib.FTP(ftp_config['host']) as ftp:
                ftp.login(ftp_config['user'], ftp_config['password'])
                remote_path = ftp_config.get('remote_path', '/videos')

                # Cambiar al directorio remoto
                try:
                    ftp.cwd(remote_path)
                except ftplib.error_perm:
                    print(f"[FTP] Directorio remoto {remote_path} no existe")
                    return

                # Obtener lista de directorios de cámaras
                try:
                    dirs = []
                    ftp.retrlines('LIST', lambda x: dirs.append(x.split()[-1]) if x.startswith('d') else None)
                except:
                    dirs = []

                archivos_eliminados = 0
                tiempo_limite = 96 * 3600  # 96 horas en segundos
                ahora = time.time()

                for cam_dir in dirs:
                    try:
                        ftp.cwd(cam_dir)
                        # Obtener archivos en el directorio de la cámara
                        archivos = []
                        ftp.retrlines('LIST', lambda x: archivos.append(x.split()))

                        for archivo_info in archivos:
                            if len(archivo_info) >= 9:
                                # Parsear fecha del archivo (formato FTP: -rw-r--r-- 1 user group size month day time filename)
                                try:
                                    mes, dia, hora_o_ano = archivo_info[5], archivo_info[6], archivo_info[7]
                                    filename = ' '.join(archivo_info[8:])

                                    # Convertir fecha a timestamp (aproximado)
                                    fecha_archivo = self.parsear_fecha_ftp(mes, dia, hora_o_ano)
                                    if fecha_archivo and (ahora - fecha_archivo) > tiempo_limite:
                                        # Eliminar archivo
                                        ftp.delete(filename)
                                        archivos_eliminados += 1
                                        print(f"[FTP] Eliminado archivo antiguo: {cam_dir}/{filename}")
                                except:
                                    continue

                        ftp.cwd('..')  # Volver al directorio padre
                    except:
                        continue

                if archivos_eliminados > 0:
                    self.log_ftp(f"Eliminados {archivos_eliminados} archivos antiguos del FTP")
                print(f"[FTP] Limpieza completada: {archivos_eliminados} archivos eliminados")

        except Exception as e:
            print(f"[FTP] Error en limpieza: {e}")
            self.log_ftp(f"Error en limpieza FTP: {str(e)[:50]}...")

    def parsear_fecha_ftp(self, mes, dia, hora_o_ano):
        """Parsea fecha de listado FTP a timestamp."""
        try:
            meses = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

            ahora = datetime.datetime.now()
            mes_num = meses.get(mes, 1)

            if ':' in hora_o_ano:
                # Formato: HH:MM (archivo de este año)
                hora, minuto = map(int, hora_o_ano.split(':'))
                fecha = datetime.datetime(ahora.year, mes_num, int(dia), hora, minuto)
            else:
                # Formato: YYYY (archivo de años anteriores)
                ano = int(hora_o_ano)
                fecha = datetime.datetime(ano, mes_num, int(dia))

            return fecha.timestamp()
        except:
            return None

    def ejecutar_limpieza_videos(self, dias_conservar):
        """Ejecuta la limpieza de videos antiguos."""
        from threading import Thread

        def limpieza():
            try:
                directorio_base = self.directorio_videos.get()
                if not os.path.exists(directorio_base):
                    messagebox.showerror("Error", f"El directorio {directorio_base} no existe.")
                    return

                # Calcular fecha límite
                fecha_limite = datetime.datetime.now() - datetime.timedelta(days=dias_conservar)
                limite_timestamp = fecha_limite.timestamp()

                archivos_eliminados = 0
                espacio_liberado = 0

                # Recorrer todas las carpetas de cámaras
                for carpeta_camara in os.listdir(directorio_base):
                    ruta_camara = os.path.join(directorio_base, carpeta_camara)
                    if not os.path.isdir(ruta_camara):
                        continue

                    print(f"[Limpieza] Revisando carpeta: {carpeta_camara}")

                    # Obtener lista de archivos
                    archivos = []
                    for archivo in os.listdir(ruta_camara):
                        ruta_completa = os.path.join(ruta_camara, archivo)
                        if os.path.isfile(ruta_completa):
                            try:
                                # Extraer fecha del nombre del archivo
                                # Formato esperado: continuo_YYYYMMDD_HHMMSS.avi o YYYYMMDD_HHMMSS.avi
                                if archivo.startswith('continuo_'):
                                    partes = archivo.split('_')
                                    if len(partes) >= 3:
                                        fecha_str = partes[1] + '_' + partes[2].split('.')[0]
                                    else:
                                        continue
                                else:
                                    fecha_str = archivo.split('.')[0]

                                # Convertir a timestamp
                                fecha_archivo = datetime.datetime.strptime(fecha_str, "%Y%m%d_%H%M%S")
                                archivos.append((ruta_completa, fecha_archivo.timestamp(), os.path.getsize(ruta_completa)))
                            except (ValueError, IndexError):
                                # Si no se puede parsear la fecha, omitir
                                continue

                    # Eliminar archivos antiguos
                    for ruta_archivo, timestamp_archivo, tamano in archivos:
                        if timestamp_archivo < limite_timestamp:
                            try:
                                os.remove(ruta_archivo)
                                archivos_eliminados += 1
                                espacio_liberado += tamano
                                print(f"[Limpieza] Eliminado: {os.path.basename(ruta_archivo)}")
                            except Exception as e:
                                print(f"[Limpieza] Error eliminando {ruta_archivo}: {e}")

                # Convertir bytes a MB
                espacio_mb = espacio_liberado / (1024 * 1024)

                mensaje = f"Limpieza completada:\n\n"
                mensaje += f"📁 Archivos eliminados: {archivos_eliminados}\n"
                mensaje += f"💾 Espacio liberado: {espacio_mb:.2f} MB\n"
                mensaje += f"📅 Archivos anteriores a: {fecha_limite.strftime('%Y-%m-%d %H:%M:%S')}"

                messagebox.showinfo("Limpieza Completada", mensaje)
                print(f"[Limpieza] {archivos_eliminados} archivos eliminados, {espacio_mb:.2f} MB liberados")

            except Exception as e:
                messagebox.showerror("Error", f"Error durante la limpieza: {str(e)}")
                print(f"[Error] Limpieza fallida: {e}")

        # Ejecutar en un hilo separado para no bloquear la interfaz
        Thread(target=limpieza, daemon=True).start()

    def setup_style(self):
        """Configura el estilo moderno de la aplicación con ttkbootstrap."""
        # Configurar fuente moderna
        default_font = (STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_normal'])
        self.root.option_add("*Font", default_font)

        # Configurar estilos personalizados usando ttkbootstrap
        self.style.configure("Modern.TButton", font=default_font)
        self.style.configure("Modern.TLabel", font=default_font)
        self.style.configure("Modern.TFrame")
        self.style.configure("Modern.TLabelframe", font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_title'], "bold"))
        self.style.configure("Modern.TLabelframe.Label", font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_title'], "bold"))

    def setup_window(self):
        """Configura el tamaño y comportamiento de la ventana principal."""
        # Obtener tamaño de pantalla
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Configurar ventana para ocupar la mayor parte de la pantalla
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.85)

        # Centrar ventana
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(800, 600)
        # ttkbootstrap maneja el fondo automáticamente

    def crear_interfaz_principal(self):
        """Crea la interfaz principal mostrando solo los videos."""
        # Frame principal con scroll
        self.main_frame = ttk.Frame(self.root, style="Modern.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas y scrollbar para scroll
        self.canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="Modern.TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Título de la aplicación
        title_label = ttk.Label(self.scrollable_frame,
                               text="Sistema de Vigilancia RTSP",
                               font=(STYLE_CONFIG['font_family'], 16, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Información del estado
        self.status_frame = ttk.Frame(self.scrollable_frame, style="Modern.TFrame")
        self.status_frame.pack(fill=tk.X, pady=(0, 20))

        self.status_label = ttk.Label(self.status_frame,
                                     text="Cámaras: 0 | Estado: Detenido",
                                     font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_small']),
                                     style="Modern.TLabel")
        self.status_label.pack(side=tk.LEFT)

        # Botón para iniciar/detener cámaras
        self.start_button = ttk.Button(self.status_frame,
                                      text="▶ Iniciar Cámaras",
                                      style="success.TButton",
                                      command=self.toggle_camaras)
        self.start_button.pack(side=tk.RIGHT)

        # Panel de logs FTP
        self.log_frame = ttk.Labelframe(self.scrollable_frame, text="📋 Logs FTP",
                                        style="Modern.TLabelframe")
        self.log_frame.pack(fill=tk.X, pady=(0, 10))

        # Text widget para logs con scrollbar
        self.log_text = tk.Text(self.log_frame, height=4, wrap=tk.WORD,
                               font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_small']))
        log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=2)

        # Frame para mostrar los videos
        self.frame_videos = ttk.Frame(self.scrollable_frame, style="Modern.TFrame")
        self.frame_videos.pack(fill=tk.BOTH, expand=True)

    def toggle_camaras(self):
        """Inicia o detiene las cámaras según el estado actual."""
        if self.camaras:
            self.detener_camaras()
        else:
            self.iniciar_camaras()

    def crear_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menú Archivo
        archivo_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=archivo_menu)
        archivo_menu.add_command(label="Salir", command=self.confirmar_cierre)

        # Menú Cámaras
        camaras_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Cámaras", menu=camaras_menu)

        camaras_menu.add_command(label="Gestión de Cámaras", command=self.gestionar_camaras)
        camaras_menu.add_separator()
        camaras_menu.add_command(label="Cargar Cámaras", command=self.cargar_camaras)
        camaras_menu.add_command(label="Guardar Cámaras", command=self.guardar_camaras)

        # Menú Configuración
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Configuración", menu=config_menu)

        config_menu.add_command(label="Configuración de Grabación", command=self.mostrar_config_grabacion)
        config_menu.add_command(label="Directorio de Videos", command=self.cambiar_directorio_videos)
        config_menu.add_command(label="Configuración FTP", command=self.configurar_ftp)
        config_menu.add_command(label="Limpiar Videos Antiguos", command=self.limpiar_videos_antiguos)
        config_menu.add_separator()
        config_menu.add_command(label="Guardar Configuración", command=self.guardar_configuracion)

        # Menú Ayuda
        ayuda_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=ayuda_menu)
        ayuda_menu.add_command(label="Acerca de", command=self.mostrar_acerca_de)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def mostrar_config_grabacion(self):
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Configuración de Grabación")
        dialog.geometry("500x350")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Configuración de Grabación",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Modo de grabación
        modo_frame = ttk.LabelFrame(main_frame, text="Modo de Grabación",
                                   style="Modern.TLabelframe")
        modo_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Radiobutton(modo_frame, text="📹 Por movimiento", variable=self.modo_grabacion, value="motion").pack(anchor=tk.W, pady=2, padx=10)
        ttk.Radiobutton(modo_frame, text="🔄 Continua", variable=self.modo_grabacion, value="continuous").pack(anchor=tk.W, pady=2, padx=10)
        ttk.Radiobutton(modo_frame, text="🚫 Sin grabación", variable=self.modo_grabacion, value="none").pack(anchor=tk.W, pady=2, padx=10)

        # Configuración de tiempo
        tiempo_frame = ttk.LabelFrame(main_frame, text="Configuración de Tiempo",
                                     style="Modern.TLabelframe")
        tiempo_frame.pack(fill=tk.X, pady=(0, 15))

        # Duración para modo movimiento
        dur_frame = ttk.Frame(tiempo_frame, style="Modern.TFrame")
        dur_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(dur_frame, text="Duración grabación (segundos):",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(dur_frame, from_=5, to=300, textvariable=self.duracion_grabacion, width=10).pack(side=tk.RIGHT)

        # Segmento para modo continuo
        seg_frame = ttk.Frame(tiempo_frame, style="Modern.TFrame")
        seg_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(seg_frame, text="Segmento continuo (minutos):",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(seg_frame, from_=1, to=60, textvariable=self.segmento_continuo, width=10).pack(side=tk.RIGHT)

        # Sensibilidad del movimiento
        sens_frame = ttk.Frame(tiempo_frame, style="Modern.TFrame")
        sens_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(sens_frame, text="Sensibilidad movimiento:",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(sens_frame, from_=50, to=255, textvariable=self.sensibilidad_movimiento, width=10).pack(side=tk.RIGHT)

        # Botón aceptar
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(btn_frame, text="✅ Aceptar",
                  style="success.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def gestionar_camaras(self):
        """Muestra el diálogo de gestión de cámaras."""
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Gestión de Cámaras")
        dialog.geometry("700x500")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Gestión de Cámaras RTSP",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Lista de cámaras
        list_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        ttk.Label(list_frame, text="Cámaras configuradas:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))

        # Frame con scrollbar para la lista
        listbox_frame = ttk.Frame(list_frame, style="Modern.TFrame")
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.gestion_lista_camaras = tk.Listbox(listbox_frame,
                                               width=80,
                                               height=10,
                                               font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_normal']),
                                               selectmode=tk.SINGLE,
                                               yscrollcommand=scrollbar.set)
        self.gestion_lista_camaras.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.gestion_lista_camaras.yview)

        # Actualizar lista
        self.actualizar_gestion_lista()

        # Botones de acción
        buttons_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        buttons_frame.pack(fill=tk.X, pady=(0, 20))

        btn_add = ttk.Button(buttons_frame, text="➕ Agregar",
                            style="success.TButton",
                            command=lambda: self.agregar_camara_desde_gestion(dialog))
        btn_add.pack(side=tk.LEFT, padx=(0, 10))

        btn_edit = ttk.Button(buttons_frame, text="✏️ Editar",
                             style="primary.TButton",
                             command=lambda: self.editar_camara_desde_gestion(dialog))
        btn_edit.pack(side=tk.LEFT, padx=(0, 10))

        btn_rename = ttk.Button(buttons_frame, text="🏷️ Renombrar",
                               style="info.TButton",
                               command=lambda: self.renombrar_camara_desde_gestion(dialog))
        btn_rename.pack(side=tk.LEFT, padx=(0, 10))

        btn_delete = ttk.Button(buttons_frame, text="🗑️ Eliminar",
                               style="danger.TButton",
                               command=lambda: self.eliminar_camara_desde_gestion(dialog))
        btn_delete.pack(side=tk.LEFT, padx=(0, 10))

        # Botones de guardar/cancelar
        bottom_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        bottom_frame.pack(fill=tk.X)

        btn_save = ttk.Button(bottom_frame, text="💾 Guardar Cambios",
                             style="success.TButton",
                             command=lambda: self.guardar_camaras_desde_gestion(dialog))
        btn_save.pack(side=tk.RIGHT, padx=(10, 0))

        btn_cancel = ttk.Button(bottom_frame, text="❌ Cancelar",
                               style="secondary.TButton",
                               command=dialog.destroy)
        btn_cancel.pack(side=tk.RIGHT)

    def actualizar_gestion_lista(self):
        """Actualiza la lista en el diálogo de gestión."""
        if hasattr(self, 'gestion_lista_camaras'):
            self.gestion_lista_camaras.delete(0, tk.END)
            for i, (url, nombre) in enumerate(zip(self.urls, self.nombres_camaras)):
                # Mostrar nombre y URL (ocultando credenciales por seguridad)
                display_url = self.ocultar_credenciales(url)
                self.gestion_lista_camaras.insert(tk.END, f"{nombre}: {display_url}")

    def ocultar_credenciales(self, url):
        """Oculta las credenciales en la URL para mostrar."""
        # Ejemplo: rtsp://user:pass@192.168.1.100:554/stream -> rtsp://***:***@192.168.1.100:554/stream
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)

    def agregar_camara_desde_gestion(self, parent_dialog):
        """Agrega una cámara desde el diálogo de gestión."""
        self.agregar_camara()
        # La lista se actualizará cuando se cierre el diálogo de agregar
        self.actualizar_gestion_lista()

    def editar_camara_desde_gestion(self, parent_dialog):
        """Edita una cámara desde el diálogo de gestión."""
        seleccion = self.gestion_lista_camaras.curselection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Seleccione una cámara para editar.")
            return
        # Usar directamente el índice de la selección
        indice = seleccion[0]
        self.editar_camara_por_indice(indice)
        self.actualizar_gestion_lista()

    def renombrar_camara_desde_gestion(self, parent_dialog):
        """Renombra una cámara desde el diálogo de gestión."""
        seleccion = self.gestion_lista_camaras.curselection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Seleccione una cámara para renombrar.")
            return
        # Usar directamente el índice de la selección
        indice = seleccion[0]
        self.renombrar_camara_por_indice(indice)
        self.actualizar_gestion_lista()

    def eliminar_camara_desde_gestion(self, parent_dialog):
        """Elimina una cámara desde el diálogo de gestión."""
        seleccion = self.gestion_lista_camaras.curselection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Seleccione una cámara para eliminar.")
            return
        # Usar directamente el índice de la selección
        indice = seleccion[0]
        self.eliminar_camara_por_indice(indice)
        self.actualizar_gestion_lista()

    def guardar_camaras_desde_gestion(self, parent_dialog):
        """Guarda los cambios desde el diálogo de gestión."""
        self.guardar_camaras()
        parent_dialog.destroy()

    def mostrar_acerca_de(self):
        """Muestra información sobre la aplicación."""
        about_text = """Sistema de Vigilancia RTSP

Versión 2.0

Características:
• Detección de movimiento inteligente
• Grabación continua configurable
• Soporte para audio (detección automática)
• Subida automática a FTP
• Interfaz moderna y responsiva

Desarrollado con Python y OpenCV"""
        messagebox.showinfo("Acerca de", about_text)

    def cambiar_directorio_videos(self):
        directorio = filedialog.askdirectory(title="Seleccionar directorio para videos")
        if directorio:
            self.directorio_videos.set(directorio)
            messagebox.showinfo("Configuración", f"Directorio de videos cambiado a:\n{directorio}")
            print(f"[Config] Directorio de videos cambiado a: {directorio}")

    def configurar_ftp(self):
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Configuración FTP")
        dialog.geometry("500x300")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Configuración FTP",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Campos de configuración
        fields_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        fields_frame.pack(fill=tk.X, pady=(0, 20))

        # Servidor FTP
        host_frame = ttk.Frame(fields_frame, style="Modern.TFrame")
        host_frame.pack(fill=tk.X, pady=2)
        ttk.Label(host_frame, text="🌐 Servidor FTP:",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        host_entry = ttk.Entry(host_frame, width=30)
        host_entry.insert(0, self.ftp_config['host'])
        host_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # Usuario
        user_frame = ttk.Frame(fields_frame, style="Modern.TFrame")
        user_frame.pack(fill=tk.X, pady=2)
        ttk.Label(user_frame, text="👤 Usuario:",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        user_entry = ttk.Entry(user_frame, width=30)
        user_entry.insert(0, self.ftp_config['user'])
        user_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # Contraseña
        pass_frame = ttk.Frame(fields_frame, style="Modern.TFrame")
        pass_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pass_frame, text="🔒 Contraseña:",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        pass_entry = ttk.Entry(pass_frame, width=30, show="*")
        pass_entry.insert(0, self.ftp_config['password'])
        pass_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # Ruta remota
        path_frame = ttk.Frame(fields_frame, style="Modern.TFrame")
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="📁 Ruta remota:",
                 style="Modern.TLabel").pack(side=tk.LEFT)
        path_entry = ttk.Entry(path_frame, width=30)
        path_entry.insert(0, self.ftp_config['remote_path'])
        path_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # Botones
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=(20, 0))

        def guardar():
            self.ftp_config['host'] = host_entry.get().strip()
            self.ftp_config['user'] = user_entry.get().strip()
            self.ftp_config['password'] = pass_entry.get().strip()
            self.ftp_config['remote_path'] = path_entry.get().strip()
            messagebox.showinfo("Configuración", "Configuración FTP guardada correctamente.")
            dialog.destroy()

        ttk.Button(btn_frame, text="💾 Guardar",
                  style="success.TButton",
                  command=guardar).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="❌ Cancelar",
                  style="secondary.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def limpiar_videos_antiguos(self):
        """Muestra el diálogo para limpiar videos antiguos."""
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Limpiar Videos Antiguos")
        dialog.geometry("400x200")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Limpiar Videos Antiguos",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Campo para días a conservar
        days_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        days_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(days_frame, text="📅 Días a conservar:",
                 style="Modern.TLabel").pack(side=tk.LEFT)

        days_var = tk.IntVar(value=1)  # Valor por defecto: 1 día
        days_spin = ttk.Spinbox(days_frame, from_=1, to=365, textvariable=days_var, width=10)
        days_spin.pack(side=tk.RIGHT, padx=(10, 0))

        # Botones
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=(20, 0))

        def ejecutar_limpieza():
            dias = days_var.get()
            dialog.destroy()
            self.ejecutar_limpieza_videos(dias)

        ttk.Button(btn_frame, text="🗑️ Limpiar",
                  style="danger.TButton",
                  command=ejecutar_limpieza).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="❌ Cancelar",
                  style="secondary.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def guardar_configuracion(self):
        config = {
            'modo_grabacion': self.modo_grabacion.get(),
            'duracion_grabacion': self.duracion_grabacion.get(),
            'segmento_continuo': self.segmento_continuo.get(),
            'sensibilidad_movimiento': self.sensibilidad_movimiento.get(),
            'directorio_videos': self.directorio_videos.get(),
            'ftp_config': self.ftp_config
        }
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Éxito", "Configuración guardada correctamente.")
            print("[Config] Configuración guardada en config.json")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración: {e}")
            print(f"[Error] No se pudo guardar la configuración: {e}")

    def cargar_configuracion(self):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            self.modo_grabacion.set(config.get('modo_grabacion', 'motion'))
            self.duracion_grabacion.set(config.get('duracion_grabacion', 15))
            self.segmento_continuo.set(config.get('segmento_continuo', 5))
            self.sensibilidad_movimiento.set(config.get('sensibilidad_movimiento', 100))
            self.directorio_videos.set(config.get('directorio_videos', './videos'))
            self.ftp_config.update(config.get('ftp_config', {}))
            print("[Config] Configuración cargada desde config.json")
        except FileNotFoundError:
            print("[Config] No se encontró archivo de configuración, usando valores por defecto")
        except Exception as e:
            print(f"[Error] Error al cargar configuración: {e}")

    def cargar_camaras(self):
        try:
            self.urls = leer_camaras("camaras.txt")
        except FileNotFoundError:
            self.urls = []
        # Cargar nombres de cámaras si existe el archivo
        try:
            with open("nombres_camaras.txt", 'r') as f:
                self.nombres_camaras = f.read().strip().split(',')
        except FileNotFoundError:
            self.nombres_camaras = [f"Cámara_{i+1}" for i in range(len(self.urls))]
        # Asegurar que tengamos nombres para todas las URLs
        while len(self.nombres_camaras) < len(self.urls):
            self.nombres_camaras.append(f"Cámara_{len(self.nombres_camaras)+1}")
        # Actualizar estado en lugar de lista (ya no hay lista visible)
        self.actualizar_estado()

    def actualizar_lista_camaras(self):
        # Método mantenido por compatibilidad, pero ya no se usa en la interfaz principal
        pass

    def agregar_camara(self):
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Agregar nueva cámara")
        dialog.geometry("500x250")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Agregar Nueva Cámara",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Campos
        ttk.Label(main_frame, text="🏷️ Nombre de la cámara:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))
        nombre_entry = ttk.Entry(main_frame, width=50)
        nombre_entry.pack(pady=(0, 15))

        ttk.Label(main_frame, text="🔗 URL RTSP:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))
        url_entry = ttk.Entry(main_frame, width=50)
        url_entry.pack(pady=(0, 20))

        # Botones
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X)

        def guardar():
            nombre = nombre_entry.get().strip()
            url = url_entry.get().strip()
            if url:
                if not nombre:
                    nombre = f"Cámara_{len(self.urls)+1}"
                self.urls.append(url)
                self.nombres_camaras.append(nombre)
                self.actualizar_lista_camaras()
                dialog.destroy()
                messagebox.showinfo("Éxito", f"Cámara '{nombre}' agregada correctamente.")
            else:
                messagebox.showerror("Error", "La URL RTSP es obligatoria.")

        ttk.Button(btn_frame, text="➕ Agregar",
                  style="success.TButton",
                  command=guardar).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="❌ Cancelar",
                  style="secondary.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def editar_camara_por_indice(self, indice):
        """Edita una cámara por su índice."""
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Editar cámara")
        dialog.geometry("500x250")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Editar Cámara",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        # Campos
        ttk.Label(main_frame, text="🏷️ Nombre de la cámara:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))
        nombre_entry = ttk.Entry(main_frame, width=50)
        nombre_entry.insert(0, self.nombres_camaras[indice])
        nombre_entry.pack(pady=(0, 15))

        ttk.Label(main_frame, text="🔗 URL RTSP:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))
        url_entry = ttk.Entry(main_frame, width=50)
        url_entry.insert(0, self.urls[indice])
        url_entry.pack(pady=(0, 20))

        # Botones
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X)

        def guardar():
            nombre = nombre_entry.get().strip()
            url = url_entry.get().strip()
            if url:
                if not nombre:
                    nombre = f"Cámara_{indice+1}"
                self.urls[indice] = url
                self.nombres_camaras[indice] = nombre
                dialog.destroy()
                messagebox.showinfo("Éxito", f"Cámara '{nombre}' actualizada correctamente.")
            else:
                messagebox.showerror("Error", "La URL RTSP es obligatoria.")

        ttk.Button(btn_frame, text="💾 Guardar",
                  style="success.TButton",
                  command=guardar).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="❌ Cancelar",
                  style="secondary.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def editar_camara(self):
        """Método de compatibilidad - ya no se usa."""
        pass

    def eliminar_camara_por_indice(self, indice):
        """Elimina una cámara por su índice."""
        nombre = self.nombres_camaras[indice]
        if messagebox.askyesno("Confirmar eliminación", f"¿Está seguro de que desea eliminar la cámara '{nombre}'?"):
            del self.urls[indice]
            del self.nombres_camaras[indice]
            messagebox.showinfo("Éxito", f"Cámara '{nombre}' eliminada correctamente.")

    def eliminar_camara(self):
        """Método de compatibilidad - ya no se usa."""
        pass

    def renombrar_camara_por_indice(self, indice):
        """Renombra una cámara por su índice."""
        dialog = ttk.Window(themename=STYLE_CONFIG['theme'])
        dialog.title("Renombrar cámara")
        dialog.geometry("400x180")

        # Frame principal
        main_frame = ttk.Frame(dialog, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = ttk.Label(main_frame,
                               text="Renombrar Cámara",
                               font=(STYLE_CONFIG['font_family'], 14, "bold"),
                               style="Modern.TLabel")
        title_label.pack(pady=(0, 20))

        ttk.Label(main_frame, text="🏷️ Nuevo nombre:",
                 style="Modern.TLabel").pack(anchor=tk.W, pady=(0, 5))
        nombre_entry = ttk.Entry(main_frame, width=40)
        nombre_entry.insert(0, self.nombres_camaras[indice])
        nombre_entry.pack(pady=(0, 20))

        # Botones
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X)

        def guardar():
            nombre = nombre_entry.get().strip()
            if nombre:
                self.nombres_camaras[indice] = nombre
                dialog.destroy()
                messagebox.showinfo("Éxito", f"Cámara renombrada a '{nombre}'.")
            else:
                messagebox.showerror("Error", "El nombre no puede estar vacío.")

        ttk.Button(btn_frame, text="🏷️ Renombrar",
                  style="success.TButton",
                  command=guardar).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="❌ Cancelar",
                  style="secondary.TButton",
                  command=dialog.destroy).pack(side=tk.RIGHT)

    def renombrar_camara(self):
        """Método de compatibilidad - ya no se usa."""
        pass

    def guardar_camaras(self):
        with open("camaras.txt", 'w') as f:
            f.write(','.join(self.urls))
        with open("nombres_camaras.txt", 'w') as f:
            f.write(','.join(self.nombres_camaras))
        messagebox.showinfo("Éxito", "Configuración de cámaras guardada correctamente.")
        print("[Info] Cámaras guardadas en camaras.txt y nombres_camaras.txt")

    def iniciar_camaras(self):
        if not self.urls:
            messagebox.showwarning("Advertencia", "No hay cámaras configuradas. Use 'Cámaras > Gestión de Cámaras' para agregar cámaras.")
            return

        # Limpiar frames anteriores
        for widget in self.frame_videos.winfo_children():
            widget.destroy()
        self.camaras = []

        # Configurar parámetros según el modo seleccionado
        config = {
            'modo_grabacion': self.modo_grabacion.get(),
            'duracion_grabacion': self.duracion_grabacion.get(),
            'segmento_continuo': self.segmento_continuo.get(),
            'sensibilidad_movimiento': self.sensibilidad_movimiento.get(),
            'directorio_videos': self.directorio_videos.get(),
            'ftp_config': self.ftp_config.copy()
        }

        # Calcular layout óptimo para las cámaras
        num_camaras = len(self.urls)
        cols = min(4, max(1, int(num_camaras ** 0.5) + 1))  # Máximo 4 columnas
        rows = (num_camaras + cols - 1) // cols

        for i, (url, nombre) in enumerate(zip(self.urls, self.nombres_camaras)):
            row = i // cols
            col = i % cols

            # Frame para cada cámara con estilo moderno
            frame_camara = ttk.Frame(self.frame_videos,
                                     style="Modern.TFrame")
            frame_camara.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')

            # Configurar pesos para que se expandan
            self.frame_videos.grid_rowconfigure(row, weight=1)
            self.frame_videos.grid_columnconfigure(col, weight=1)

            # Etiqueta con nombre de la cámara
            nombre_label = ttk.Label(frame_camara,
                                    text=nombre,
                                    font=(STYLE_CONFIG['font_family'], STYLE_CONFIG['font_size_small'], "bold"),
                                    style="Modern.TLabel",
                                    anchor='center')
            nombre_label.pack(fill=tk.X, pady=(5, 0))

            camara = Camara(url, nombre, frame_camara, self.root_window, config)
            # Asignar referencia a la aplicación para logging
            camara.app = self
            # Bind double-click to open full screen
            frame_camara.bind("<Double-Button-1>", lambda e, c=camara: self.abrir_ventana_completa(c))
            camara.canvas.bind("<Double-Button-1>", lambda e, c=camara: self.abrir_ventana_completa(c))
            self.camaras.append(camara)
            camara.iniciar()

        # Actualizar estado
        self.actualizar_estado()
        self.start_button.config(text="⏹️ Detener Cámaras", bg='#d13438')

    def detener_camaras(self):
        for camara in self.camaras:
            camara.running = False
            if camara.cap.isOpened():
                camara.cap.release()
            # Cerrar grabación continua si está activa
            if hasattr(camara, 'segmento_actual') and camara.segmento_actual:
                camara.segmento_actual.release()
            # Detener grabación de audio
            camara.detener_grabacion_audio()

        self.camaras = []
        # Limpiar frames de video
        for widget in self.frame_videos.winfo_children():
            widget.destroy()

        # Actualizar estado
        self.actualizar_estado()
        self.start_button.config(text="▶ Iniciar Cámaras", bg=STYLE_CONFIG['accent_color'])

    def actualizar_estado(self):
        """Actualiza la información de estado en la interfaz."""
        num_camaras = len(self.camaras) if self.camaras else len(self.urls)
        estado = "Activo" if self.camaras else "Detenido"
        modo = self.modo_grabacion.get()
        modo_texto = {
            'motion': 'Movimiento',
            'continuous': 'Continuo',
            'none': 'Sin grabación'
        }.get(modo, modo)

        self.status_label.config(text=f"Cámaras: {num_camaras} | Estado: {estado} | Modo: {modo_texto}")

    def confirmar_cierre(self):
        if messagebox.askyesno("Confirmar cierre", "¿Está seguro de que desea cerrar la aplicación?\n\nEsto detendrá todas las cámaras y grabaciones en curso."):
            self.detener_camaras()
            self.root.destroy()
        # Si el usuario cancela, no hacer nada (la aplicación continúa ejecutándose)

    def abrir_ventana_completa(self, camara):
        ventana = ttk.Window(themename=STYLE_CONFIG['theme'])
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
    root = ttk.Window(themename=STYLE_CONFIG['theme'])
    app = Aplicacion(root)
    # La configuración ya se carga en el constructor
    root.protocol("WM_DELETE_WINDOW", app.confirmar_cierre)
    root.mainloop()
