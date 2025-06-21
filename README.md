# Acoustic Calibration Tool

A simple Python tool for acoustic calibration and tone generation, with a minimal GUI.

## Features
- Generate pure tones at user-selected frequency, amplitude, and duration.
- Play tones via your audio output.
- (Planned) Real-time visualization of microphone input and spectrum.
- Easy-to-use graphical interface, suitable for education and quick checks.

## How to Use
1. Set the desired frequency (Hz), gain (0-1), and duration (seconds).
2. Click **Play Tone** to generate and play the tone.
3. (Planned) Observe the real-time plot of the microphone input.

For advanced calibration or analysis, see future updates or contribute via GitHub.

## Requirements

- Python 3.8+
- `numpy`
- `sounddevice`
- `matplotlib`

Install with:
```bash
pip install numpy sounddevice matplotlib
