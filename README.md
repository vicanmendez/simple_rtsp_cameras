# Sistema de Seguridad con Cámaras RTSP

Un sistema de monitoreo de cámaras de seguridad basado en Python con detección de movimiento y grabación de video.

## Características

- **Streaming RTSP multi-cámara**: Monitorea múltiples cámaras IP simultáneamente
- **Detección de movimiento**: Detección automática de movimiento con sensibilidad configurable
- **Grabación de video**: Graba clips de video cuando se detecta movimiento
- **Gestión de cámaras**: Agrega, edita y elimina streams de cámaras a través de la interfaz gráfica
- **Vista en pantalla completa**: Haz doble clic en cualquier cámara para verla en modo pantalla completa
- **Operación segura con hilos**: Reproducción de video fluida sin congelar la interfaz

## Requisitos

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Pillow (`pip install pillow`)
- NumPy (`pip install numpy`)

## Instalación

1. Clona o descarga los archivos del proyecto
2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloca las URLs RTSP de tus cámaras en `camaras.txt`, separadas por comas

## Uso

### Iniciando la Aplicación

Ejecuta el script principal:
```bash
python detector.py
```

### Gestión de Cámaras

1. **Cargar Cámaras**: Haz clic en "Cargar cámaras" para cargar las cámaras desde `camaras.txt`
2. **Agregar Cámara**: Haz clic en "Agregar cámara" para añadir manualmente un nuevo stream RTSP
3. **Editar Cámara**: Selecciona una cámara de la lista y haz clic en "Editar cámara" para modificar su URL
4. **Eliminar Cámara**: Selecciona una cámara y haz clic en "Eliminar cámara" para removerla
5. **Guardar Cambios**: Haz clic en "Guardar cambios" para guardar tu lista de cámaras en `camaras.txt`

### Monitoreo

1. Haz clic en "Iniciar cámaras" para comenzar el streaming de todas las cámaras cargadas
2. Las cámaras se mostrarán en un diseño de cuadrícula
3. La detección de movimiento funciona automáticamente en segundo plano
4. Cuando se detecta movimiento, se guarda un clip de video en el directorio `videos/`

### Vista en Pantalla Completa

Haz doble clic en cualquier cámara de la cuadrícula para abrirla en una ventana dedicada de pantalla completa.

## Configuración

### Modos de Grabación

La aplicación ofrece tres modos de grabación configurables desde la interfaz:

1. **Por movimiento**: Graba cuando se detecta movimiento
   - Duración configurable (5-300 segundos)
   - Sensibilidad ajustable (50-255)

2. **Continua**: Graba continuamente en segmentos fijos
   - Segmentos de 1-60 minutos
   - Archivos separados por período

3. **Sin grabación**: Solo visualización, sin almacenamiento

### Configuración de Detección de Movimiento

Los parámetros de detección de movimiento se pueden ajustar desde la interfaz:

- **Sensibilidad del movimiento**: Controla qué tan sensible es la detección (50-255)
- **Kernel de desenfoque**: `(31, 31)` - Reduce el ruido de la imagen
- **Área mínima**: `5000` píxeles - Tamaño mínimo de contorno para considerar como movimiento
- **Tasa de actualización del fondo**: `0.005` - Qué tan rápido se adapta el modelo de fondo

### Configuración de Grabación

- **Duración de grabación**: Configurable por el usuario (5-300 segundos para modo movimiento)
- **Segmentos continuos**: Configurable por el usuario (1-60 minutos)
- **Tasa de frames**: 20 FPS
- **Códec**: XVID
- **Tiempo de enfriamiento**: 5 segundos entre grabaciones por movimiento

## Estructura de Archivos

```
├── detector.py          # Aplicación principal
├── camaras.txt          # URLs de cámaras (separadas por comas)
├── requirements.txt     # Dependencias del proyecto
├── videos/              # Clips de video grabados
│   └── Cámara_1/
│       ├── 20241114_231632.avi
│       └── ...
└── README.md           # Este archivo
```

## Encontrando URLs RTSP

Las URLs RTSP generalmente se pueden encontrar en la interfaz web de tu cámara o en su documentación. Los formatos comunes incluyen:

- `rtsp://usuario:contraseña@direccion_ip:puerto/stream`
- `rtsp://direccion_ip:puerto/live/ch0`

Herramientas como ONVIF Device Manager (ODM) pueden ayudar a descubrir y configurar cámaras IP en tu red.

## Solución de Problemas

- **No se muestra video**: Verifica el formato de la URL RTSP y la conectividad de red
- **Detección de movimiento demasiado sensible**: Aumenta los valores de umbral y área mínima
- **Detección de movimiento poco sensible**: Disminuye los valores de umbral y área mínima
- **La grabación no funciona**: Asegúrate de tener permisos de escritura en el directorio `videos/`
- **Interfaz gráfica se congela**: La aplicación usa hilos para prevenir congelamientos durante el streaming

## Notas de Seguridad

- Almacena las credenciales de las cámaras de forma segura
- Usa HTTPS cuando sea posible para las interfaces web
- Mantén actualizada la aplicación y las dependencias
- Monitorea el uso de almacenamiento del directorio `videos/`

## Licencia

Este proyecto se proporciona tal cual para uso educativo y personal.