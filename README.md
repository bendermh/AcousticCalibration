# AcousticCalibration

**AcousticCalibration** es una herramienta de análisis espectral y calibración acústica en tiempo real, diseñada para mediciones fiables con interfaz sencilla. Permite visualizar el espectro de cualquier entrada de micrófono, ajustar los parámetros de la ventana de análisis y exportar resultados fácilmente.

![Screenshot](./docs/screenshot.png)

## Características principales

- Selección de dispositivo de entrada (micrófono)
- Control en tiempo real de rango de frecuencias (mínimo/máximo)
- Ajuste de rango de amplitud en dBFS (Y min/max)
- Smoothing configurable en tiempo real
- Marcador de pico automático y manual (con anotaciones en la gráfica)
- Guardado de la gráfica como imagen (PNG) incluyendo anotaciones
- Interfaz en inglés, apta para uso clínico y de laboratorio
- Protocolo de cierre seguro, robusto para evitar errores de threads/GUI

## Instalación

Este software requiere Python 3.8+ y los siguientes paquetes:

- `numpy`
- `matplotlib`
- `sounddevice`
- `tkinter` (incluido en la mayoría de instalaciones de Python)
- `scipy` (solo si quieres ampliar cálculos futuros)

Instala dependencias así:

```sh
pip install numpy matplotlib sounddevice
