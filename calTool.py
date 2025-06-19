# -*- coding: utf-8 -*-
"""
Acoustic calibration tool
19/06/2025 Jorge Rey-Martinez

"""

import tkinter as tk
from tkinter import ttk, filedialog
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SpectrumAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Spectrum Analyzer")
        self.root.geometry("1000x620")
        self.running = False
        self.stream = None
        self.update_interval = 100
        self.fft_size = 2048
        self.sample_rate = 44100

        self.freq_min = tk.DoubleVar(value=20)
        self.freq_max = tk.DoubleVar(value=8000)
        self.center_freq = tk.DoubleVar(value=1000)
        self.manual_freq = tk.DoubleVar(value=0)
        self.ymin = tk.DoubleVar(value=-100)
        self.ymax = tk.DoubleVar(value=0)
        self.smooth_factor = tk.DoubleVar(value=0.4)

        self.selected_device = tk.StringVar()
        self.device_list = self.get_input_devices()
        if self.device_list:
            self.selected_device.set(self.device_list[0])

        self.init_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Row 0: Device + Buttons
        ttk.Label(frame, text="Input device:").grid(row=0, column=0, sticky=tk.W)
        self.device_menu = ttk.OptionMenu(frame, self.selected_device, self.selected_device.get(), *self.device_list)
        self.device_menu.grid(row=0, column=1, sticky=tk.W)

        ttk.Button(frame, text="Start", command=self.start_stream).grid(row=0, column=2)
        ttk.Button(frame, text="Stop", command=self.stop_stream).grid(row=0, column=3)
        ttk.Button(frame, text="Save Image", command=self.save_image).grid(row=0, column=4)

        # Row 1: Frequency and Y-axis range
        ttk.Label(frame, text="Min freq (Hz):").grid(row=1, column=0, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.freq_min, width=8).grid(row=1, column=1)

        ttk.Label(frame, text="Max freq (Hz):").grid(row=1, column=2, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.freq_max, width=8).grid(row=1, column=3)

        ttk.Label(frame, text="Y min (dBFS):").grid(row=1, column=4, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.ymin, width=8).grid(row=1, column=5)

        ttk.Label(frame, text="Y max (dBFS):").grid(row=1, column=6, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.ymax, width=8).grid(row=1, column=7)

        # Row 2: Manual peak and smoothing
        ttk.Label(frame, text="Manual peak (Hz):").grid(row=2, column=0, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.manual_freq, width=8).grid(row=2, column=1)

        ttk.Label(frame, text="Smoothing [0.0–0.999]:").grid(row=2, column=2, sticky=tk.E)
        ttk.Entry(frame, textvariable=self.smooth_factor, width=8).grid(row=2, column=3)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(9, 4), dpi=100)
        self.line, = self.ax.plot([], [], color="orange")
        self.auto_marker = self.ax.axvline(0, color="red", linestyle="--")
        self.manual_marker = self.ax.axvline(0, color="yellow", linestyle=":")
        self.auto_text = self.ax.text(0, 0, "", color="red", fontsize=10, ha="left", va="bottom")
        self.manual_text = self.ax.text(0, 0, "", color="yellow", fontsize=10, ha="left", va="bottom")

        self.ax.set_facecolor("#2e2e2e")
        self.fig.patch.set_facecolor("#2e2e2e")
        self.ax.tick_params(colors="white")
        self.ax.spines[:].set_color("white")
        self.ax.set_xlabel("Frequency (Hz)", color="white")
        self.ax.set_ylabel("Amplitude (dBFS)", color="white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=8, pady=(10, 5))

        self.status = tk.StringVar(value="Idle")
        ttk.Label(frame, textvariable=self.status).grid(row=4, column=0, columnspan=8, pady=(5, 10))

    def get_input_devices(self):
        devices = sd.query_devices()
        return [d["name"] for d in devices if d["max_input_channels"] > 0]

    def validate_parameters(self):
        msg = []
        try:
            smooth = float(self.smooth_factor.get())
        except:
            smooth = 0.0
            self.smooth_factor.set(0.0)
            msg.append("Smoothing set to 0.0")
        if smooth >= 1.0:
            smooth = 0.9999
            self.smooth_factor.set(0.9999)
            msg.append("Smoothing limited to 0.9999")
        elif smooth < 0.0:
            smooth = 0.0
            self.smooth_factor.set(0.0)
            msg.append("Smoothing limited to 0.0")

        try:
            manual = float(self.manual_freq.get())
        except:
            manual = 0.0
            self.manual_freq.set(0.0)
            msg.append("Manual peak set to 0")

        if msg:
            self.status.set(" | ".join(msg))
        return smooth, manual

    def audio_callback(self, indata, frames, time, status):
        if not self.running:
            return
        data = indata[:, 0] * np.hanning(len(indata))
        spectrum = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1 / self.sample_rate)
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-10)

        if not hasattr(self, "smooth_mag"):
            self.smooth_mag = magnitude
        else:
            self.smooth_mag = self.smooth_factor_value * self.smooth_mag + (1 - self.smooth_factor_value) * magnitude

        self.freqs = freqs
        self.magnitude = self.smooth_mag

    def update_plot(self):
        if not self.running:
            return
        if hasattr(self, "freqs") and hasattr(self, "magnitude"):
            self.line.set_data(self.freqs, self.magnitude)
            self.ax.set_xlim(self.freq_min.get(), self.freq_max.get())
            self.ax.set_ylim(self.ymin.get(), self.ymax.get())

            mask = (self.freqs >= self.freq_min.get()) & (self.freqs <= self.freq_max.get())
            if np.any(mask):
                max_idx = np.argmax(self.magnitude[mask])
                masked_freqs = self.freqs[mask]
                masked_mag = self.magnitude[mask]
                peak_freq = masked_freqs[max_idx]
                peak_val = masked_mag[max_idx]
                self.auto_marker.set_xdata([peak_freq])
                self.auto_text.set_position((peak_freq, peak_val))
                self.auto_text.set_text(f"{int(peak_freq)} Hz\n{peak_val:.1f} dB")
            else:
                peak_freq = 0
                peak_val = 0
                self.auto_marker.set_xdata([0])
                self.auto_text.set_text("")

            manual_freq = self.manual_freq.get()
            if manual_freq == 0:
                self.manual_marker.set_xdata([0])
                self.manual_text.set_text("")
                manual_val = 0
            else:
                idx = np.argmin(np.abs(self.freqs - manual_freq))
                manual_val = self.magnitude[idx]
                self.manual_marker.set_xdata([manual_freq])
                self.manual_text.set_position((manual_freq, manual_val))
                self.manual_text.set_text(f"{int(manual_freq)} Hz\n{manual_val:.1f} dB")

            self.status.set(
                f"Auto peak: {int(peak_freq)} Hz / {peak_val:.1f} dB   |   "
                f"Manual peak: {int(manual_freq)} Hz / {manual_val:.1f} dB"
            )

            self.canvas.draw()

        self.root.after(self.update_interval, self.update_plot)

    def start_stream(self):
        if self.running:
            return
        self.smooth_factor_value, _ = self.validate_parameters()
        self.disable_controls()
        self.running = True
        device_index = self.device_list.index(self.selected_device.get())
        self.stream = sd.InputStream(
            channels=1,
            callback=self.audio_callback,
            device=device_index,
            samplerate=self.sample_rate,
            blocksize=self.fft_size,
        )
        self.stream.start()
        self.update_plot()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        self.status.set("Stopped")
        self.enable_controls()

    def disable_controls(self):
        for child in self.root.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, ttk.Entry) or isinstance(subchild, ttk.OptionMenu):
                    subchild.configure(state="disabled")

    def enable_controls(self):
        for child in self.root.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, ttk.Entry) or isinstance(subchild, ttk.OptionMenu):
                    subchild.configure(state="normal")

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')

    def on_close(self):
        try:
            self.stop_stream()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrumAnalyzerApp(root)
    root.mainloop()