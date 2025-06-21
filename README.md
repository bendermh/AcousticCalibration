# Acoustic Calibration Tool

**Acoustic Calibration Tool** is an open-source real-time spectrum analyzer and acoustic calibration software developed in Python. It is designed for fast and reliable measurement, analysis, and documentation of audio signals using standard USB microphones or soundcards. The tool features a user-friendly graphical interface (Tkinter), live FFT visualization, dBFS calibration, and is ideal for both professional and educational use.

---

## Features

- **Real-time spectrum analysis** with adjustable FFT window and smoothing  
- **dBFS amplitude scale** for calibration and objective measurements  
- **Peak detection:** automatic and manual (user-selectable frequency)  
- **Configurable frequency and amplitude ranges** (min/max, smoothing, peak)  
- **Supports all standard audio input devices** (USB soundcards, built-in mics)  
- **Save spectrum image** (including annotated peaks and configuration)  
- **Safe cross-platform GUI:** runs on Windows, macOS, and Linux (requires Python 3.9+)  
- **Modern dark interface** for clear visualization and screenshot-ready graphics

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bendermh/AcousticCalibration.git
   cd AcousticCalibration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   or, individually:
   ```bash
   pip install numpy sounddevice matplotlib
   ```

---

## Usage

1. Run the tool:
   ```bash
   python calibration_tool.py
   ```

2. Select your audio device and configure the spectrum parameters as needed.

3. Click "Start" to begin real-time analysis.

4. Use the interface to adjust smoothing, frequency range, or select peaks (auto/manual).

5. Save the spectrum image for documentation or further analysis.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Project maintained by Jorge Rey-Martinez.**  
[https://github.com/bendermh/AcousticCalibration](https://github.com/bendermh/AcousticCalibration)
