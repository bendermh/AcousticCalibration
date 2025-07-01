# -*- coding: utf-8 -*-
"""
Acoustic Calibration Tool
Tabbed interface: Spectrum Analyzer & Calibration Assistant (dBFS only)
2025, Jorge Rey-Martinez
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

FREQUENCIES = [500, 1000, 2000, 4000, 8000]
DB_HL_STEPS = list(range(0, 80, 5))  # 0 to 75 dB HL
# Data for clinical audiometer in lour laboratory, only as demo purposes or to be used as example reference
DEFAULT_DBFS_REF = {
    500:  ["-76.0", "-71.0", "-66.0", "-61.0", "-56.0", "-51.0", "-46.0", "-41.0", "-36.0", "-31.0", "-26.0", "-21.0", "-16.0", "-11.0", "-6.0", "-1.0"],
    1000: ["-87.0", "-82.0", "-77.0", "-72.0", "-67.0", "-63.0", "-58.0", "-53.0", "-48.0", "-43.0", "-38.0", "-33.0", "-28.0", "-23.0", "-18.0", "-13.0"],
    2000: ["-86.0", "-81.0", "-76.0", "-71.0", "-66.0", "-61.0", "-56.0", "-51.0", "-46.0", "-41.0", "-36.0", "-31.0", "-26.0", "-21.0", "-16.0", "-11.0"],
    4000: ["-76.0", "-71.0", "-66.0", "-61.0", "-56.0", "-51.0", "-46.0", "-41.0", "-36.0", "-31.0", "-26.0", "-21.0", "-16.0", "-11.0", "-6.0", "-1.0"],
    8000: ["-91.0", "-86.0", "-81.0", "-76.0", "-72.0", "-67.0", "-62.0", "-57.0", "-52.0", "-47.0", "-42.0", "-37.0", "-32.0", "-27.0", "-22.0", "-17.0"]
}

class SpectrumAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Calibration Tool")
        self.root.geometry("1440x750")
        self.running = False
        self.stream = None
        self.update_interval = 100
        self.fft_size = 2048
        self.sample_rate = 44100

        # Analyzer variables
        self.freq_min = tk.DoubleVar(value=250)
        self.freq_max = tk.DoubleVar(value=8500)
        self.manual_freq = tk.DoubleVar(value=0)
        self.ymin = tk.DoubleVar(value=-100)
        self.ymax = tk.DoubleVar(value=0)
        self.smooth_factor = tk.DoubleVar(value=0.8)

        self.selected_device = tk.StringVar()
        self.device_list = self.get_input_devices()
        if self.device_list:
            self.selected_device.set(self.device_list[0])

        # Calibration variables
        self.calib_freq = tk.IntVar(value=1000)
        self.calib_data = {
            f: {
                "db_hl": [v for v in DB_HL_STEPS],
                "dbfs_ref": ["" for _ in DB_HL_STEPS],
                "dbfs_1": ["" for _ in DB_HL_STEPS],
                "dbfs_2": ["" for _ in DB_HL_STEPS],
                "gain_1": ["" for _ in DB_HL_STEPS],
                "gain_2": ["" for _ in DB_HL_STEPS],
            } for f in FREQUENCIES
        }

        # Peak values (shared)
        self.auto_peak = {"freq": None, "val": None}
        self.manual_peak = {"freq": None, "val": None}

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True)
    
        # === Spectrum Analyzer Tab ===
        self.spectrum_frame = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.spectrum_frame, text="Spectrum Analyzer")

        ttk.Label(self.spectrum_frame, text="Input device:").grid(row=0, column=0, sticky=tk.W)
        self.device_menu = ttk.OptionMenu(self.spectrum_frame, self.selected_device, self.selected_device.get(), *self.device_list)
        self.device_menu.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(self.spectrum_frame, text="Start", command=self.start_stream).grid(row=0, column=2)
        ttk.Button(self.spectrum_frame, text="Stop", command=self.stop_stream).grid(row=0, column=3)
        ttk.Button(self.spectrum_frame, text="Save Image", command=self.save_image).grid(row=0, column=4)
        
        ttk.Label(self.spectrum_frame, text="Min freq (Hz):").grid(row=1, column=0, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.freq_min, width=8).grid(row=1, column=1)
        ttk.Label(self.spectrum_frame, text="Max freq (Hz):").grid(row=1, column=2, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.freq_max, width=8).grid(row=1, column=3)
        ttk.Label(self.spectrum_frame, text="Y min (dBFS):").grid(row=1, column=4, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.ymin, width=8).grid(row=1, column=5)
        ttk.Label(self.spectrum_frame, text="Y max (dBFS):").grid(row=1, column=6, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.ymax, width=8).grid(row=1, column=7)

        ttk.Label(self.spectrum_frame, text="Manual peak (Hz):").grid(row=2, column=0, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.manual_freq, width=8).grid(row=2, column=1)
        ttk.Label(self.spectrum_frame, text="Smoothing [0.0–0.999]:").grid(row=2, column=2, sticky=tk.E)
        ttk.Entry(self.spectrum_frame, textvariable=self.smooth_factor, width=8).grid(row=2, column=3)

        self.fig, self.ax = plt.subplots(figsize=(13, 5), dpi=100)
        self.line, = self.ax.plot([], [], color="orange")
        self.auto_marker = self.ax.axvline(0, color="red", linestyle="--")
        self.manual_marker = self.ax.axvline(0, color="yellow", linestyle=":")
        self.auto_text = self.ax.text(0, 0, "", color="red", fontsize=10, ha="left", va="bottom")
        self.manual_text = self.ax.text(0, 0, "", color="yellow", fontsize=10, ha="left", va="bottom")
        self.ax.set_facecolor("#2e2e2e")
        self.fig.patch.set_facecolor("#2e2e2e")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")
        self.ax.set_xlabel("Frequency (Hz)", color="white")
        self.ax.set_ylabel("Amplitude (dBFS)", color="white")
        self.ax.title.set_color("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spectrum_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=8, pady=(10, 5))

        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.spectrum_frame, textvariable=self.status).grid(row=4, column=0, columnspan=8, pady=(5, 10))

        # === Calibration Assistant Tab ===
        self.calib_frame = ttk.Frame(self.tabs, padding=18)
        self.tabs.add(self.calib_frame, text="Calibration Assistant")
        
        # Top controls: spacing and standard button font
        ttk.Label(self.calib_frame, text="Calibration frequency (Hz):", font=("Segoe UI", 12)).grid(
            row=0, column=0, sticky=tk.E, padx=(2,8), pady=(4,12))
        freq_selector = ttk.Combobox(
            self.calib_frame,
            textvariable=self.calib_freq,
            values=FREQUENCIES,
            width=7,
            state="readonly",
            font=("Segoe UI", 12)
        )
        freq_selector.grid(row=0, column=1, sticky=tk.W, padx=(2,16), pady=(4,12))
        freq_selector.bind("<<ComboboxSelected>>", lambda e: self.update_calib_table_and_plot())
        
        ttk.Button(self.calib_frame, text="Save", command=self.save_calibration).grid(row=0, column=2, padx=(8,8), pady=(4,12))
        ttk.Button(self.calib_frame, text="Load", command=self.load_calibration).grid(row=0, column=3, padx=(8,8), pady=(4,12))
        ttk.Button(self.calib_frame, text="Load Example", command=self.load_default_dbfs_ref).grid(row=0, column=4, padx=(8,8), pady=(4,12))
        ttk.Button(self.calib_frame, text="Clear", command=self.clear_calibration_data).grid(row=0, column=5, padx=(8,8), pady=(4,12))
        
        # Calibration plot with extra separation from table
        self.calib_fig, self.calib_ax = plt.subplots(figsize=(6.5, 4.4), dpi=100)
        self.calib_canvas = FigureCanvasTkAgg(self.calib_fig, master=self.calib_frame)
        self.calib_canvas.get_tk_widget().grid(row=1, column=0, rowspan=len(DB_HL_STEPS)+3, padx=(0,25), pady=(14,8))
        
        # Table headers: bigger font, less crowded
        headers = ["dB HL", "dBFS", "Gain 1", "Exp 1", "Gain 2", "Exp 2"]
        for i, h in enumerate(headers):
            ttk.Label(self.calib_frame, text=h, anchor="center", font=("Segoe UI", 12, "bold")).grid(
                row=1, column=1+i, padx=4, pady=(2,8)
            )
        
        self.calib_entries = []
        for row, db_hl in enumerate(DB_HL_STEPS):
            row_entries = []
            # 0: dB HL (Label)
            entry_dbhl = ttk.Label(self.calib_frame, text=f"{db_hl}", width=7, anchor="center", font=("Segoe UI", 12))
            entry_dbhl.grid(row=2+row, column=1, padx=3, pady=2)
            row_entries.append(entry_dbhl)
            # 1: dBFS ref (Entry)
            e_dbfs = ttk.Entry(self.calib_frame, width=7, justify="center", font=("Segoe UI", 12))
            e_dbfs.grid(row=2+row, column=2, padx=3, pady=2)
            e_dbfs.bind("<FocusOut>", lambda e, idx=row: self.on_calib_cell_edit(idx, "dbfs_ref"))
            row_entries.append(e_dbfs)
            # 2: Gain 1 (Entry)
            e_gain1 = ttk.Entry(self.calib_frame, width=7, justify="center", font=("Segoe UI", 12))
            e_gain1.grid(row=2+row, column=3, padx=3, pady=2)
            e_gain1.bind("<FocusOut>", lambda e, idx=row: self.on_calib_cell_edit(idx, "gain_1"))
            row_entries.append(e_gain1)
            # 3: Expected 1 (Label, readonly)
            e_exp1 = ttk.Label(self.calib_frame, text="", width=7, anchor="center", background="#444", foreground="white", font=("Segoe UI", 12, "bold"))
            e_exp1.grid(row=2+row, column=4, padx=3, pady=2)
            row_entries.append(e_exp1)
            # 4: Gain 2 (Entry)
            e_gain2 = ttk.Entry(self.calib_frame, width=7, justify="center", font=("Segoe UI", 12))
            e_gain2.grid(row=2+row, column=5, padx=3, pady=2)
            e_gain2.bind("<FocusOut>", lambda e, idx=row: self.on_calib_cell_edit(idx, "gain_2"))
            row_entries.append(e_gain2)
            # 5: Expected 2 (Label, readonly)
            e_exp2 = ttk.Label(self.calib_frame, text="", width=7, anchor="center", background="#444", foreground="white", font=("Segoe UI", 12, "bold"))
            e_exp2.grid(row=2+row, column=6, padx=3, pady=2)
            row_entries.append(e_exp2)
            self.calib_entries.append(row_entries)
        
        # Regression results: large, bold and blue for visibility
        self.regression_text = tk.StringVar(value="")
        ttk.Label(
            self.calib_frame,
            textvariable=self.regression_text,
            font=("Segoe UI", 16, "bold"),
            foreground="#2255dd",  # Strong blue, visible on dark background
            anchor="w",
            justify="left"
        ).grid(
            row=2+len(DB_HL_STEPS), column=0, columnspan=7,
            pady=(32, 16), padx=(0, 0), sticky=tk.W
        )
        
        self.calib_auto_peak_text = tk.StringVar(value="Auto peak: N/A")
        self.calib_manual_peak_text = tk.StringVar(value="Manual peak: N/A")
        ttk.Label(self.calib_frame, textvariable=self.calib_auto_peak_text, font=("Segoe UI", 12, "bold")).grid(
            row=3+len(DB_HL_STEPS), column=1, columnspan=3, pady=(12,0), sticky=tk.W)
        ttk.Label(self.calib_frame, textvariable=self.calib_manual_peak_text, font=("Segoe UI", 12, "bold")).grid(
            row=3+len(DB_HL_STEPS), column=4, columnspan=3, pady=(12,0), sticky=tk.W)
        
        self.update_calib_table_and_plot()
        self.update_calib_peaks()

    # === Calibration methods ===
    def update_calib_table_and_plot(self):
        freq = self.calib_freq.get()
        data = self.calib_data[freq]
        reg_gain1 = self.get_log_regression(data["gain_1"], data["dbfs_ref"])
        reg_gain2 = self.get_log_regression(data["gain_2"], data["dbfs_ref"])
        for idx, row_entries in enumerate(self.calib_entries):
            # dB HL [col 0] (Label) -- no se actualiza
            # dBFS ref [col 1]
            row_entries[1].delete(0, tk.END)
            if data["dbfs_ref"][idx] != "":
                row_entries[1].insert(0, data["dbfs_ref"][idx])
            # Gain 1 [col 2]
            row_entries[2].delete(0, tk.END)
            if data["gain_1"][idx] != "":
                row_entries[2].insert(0, data["gain_1"][idx])
            # Expected 1 [col 3] (Label, solo .config)
            try:
                dbfs_target = float(data["dbfs_ref"][idx])
                gain_needed = self.invert_log_regression(reg_gain1, dbfs_target)
                if gain_needed is not None and np.isfinite(gain_needed) and gain_needed > 0:
                    row_entries[3].config(text=f"{gain_needed:.4f}")
                else:
                    row_entries[3].config(text="")
            except Exception:
                row_entries[3].config(text="")
            # Gain 2 [col 4]
            row_entries[4].delete(0, tk.END)
            if data["gain_2"][idx] != "":
                row_entries[4].insert(0, data["gain_2"][idx])
            # Expected 2 [col 5] (Label, solo .config)
            try:
                dbfs_target = float(data["dbfs_ref"][idx])
                gain_needed = self.invert_log_regression(reg_gain2, dbfs_target)
                if gain_needed is not None and np.isfinite(gain_needed) and gain_needed > 0:
                    row_entries[5].config(text=f"{gain_needed:.4f}")
                else:
                    row_entries[5].config(text="")
            except Exception:
                row_entries[5].config(text="")
        self.update_calib_plot(reg_gain1, reg_gain2)

    def get_log_regression(self, gain_list, dbfs_list):
        """Regresión logarítmica: dbfs = a * log10(gain) + b"""
        x, y = [], []
        for g, d in zip(gain_list, dbfs_list):
            try:
                g = float(g)
                d = float(d)
                if g > 0:
                    x.append(np.log10(g))
                    y.append(d)
            except:
                continue
        if len(x) > 1:
            coefs = np.polyfit(x, y, 1)
            # y = a*log10(gain) + b
            return coefs
        return None

    def invert_log_regression(self, coefs, dbfs_target):
         """Para y = a*log10(x) + b, despeja x = 10^((y-b)/a)"""
         if coefs is None:
             return None
         a, b = coefs
         if abs(a) < 1e-8:
             return None
         val = 10 ** ((dbfs_target - b) / a)
         return val if np.isfinite(val) and val > 0 else None
     
    def get_gain_to_dbhl_regression(self, gain_list, dbhl_list):
        x, y = [], []
        for g, dbhl in zip(gain_list, dbhl_list):
            try:
                g = float(g)
                dbhl = float(dbhl)
                if g > 0:
                    x.append(np.log10(g))
                    y.append(dbhl)
            except:
                continue
        if len(x) > 1:
            coefs = np.polyfit(x, y, 1)  # Y = m*X + n
            return coefs
        else:
            return None

    def get_regression(self, data, key):
        x_vals, y_vals = [], []
        for i, v in enumerate(data[key]):
            try:
                y = float(v)
                x = data["db_hl"][i]
                y_vals.append(y)
                x_vals.append(x)
            except:
                continue
        if len(x_vals) > 1:
            coefs = np.polyfit(x_vals, y_vals, 1)
            fit = np.poly1d(coefs)
            r2 = self.calc_r2(x_vals, y_vals, fit)
            return (fit, coefs, r2)
        else:
            return (None, None, None)

    def on_calib_cell_edit(self, idx, col):
        freq = self.calib_freq.get()
        data = self.calib_data[freq]
        entry_map = {
            "dbfs_ref": 1,
            "gain_1": 2,
            "dbfs_1": 3,
            "gain_2": 4,
            "dbfs_2": 5
        }
        if col not in entry_map:
            return
        e = self.calib_entries[idx][entry_map[col]]
        val = e.get()
        data[col][idx] = val  # Update the model data
    
        # Update all
        self.update_calib_table_and_plot()
        
    def update_expected_columns_only(self):
        freq = self.calib_freq.get()
        data = self.calib_data[freq]
        reg_gain1 = self.get_log_regression(data["gain_1"], data["dbfs_ref"])
        reg_gain2 = self.get_log_regression(data["gain_2"], data["dbfs_ref"])
        for idx, row_entries in enumerate(self.calib_entries):
            # Expected 1
            try:
                dbfs_target = float(data["dbfs_ref"][idx])
                gain_needed = self.invert_log_regression(reg_gain1, dbfs_target)
                if gain_needed is not None and np.isfinite(gain_needed) and gain_needed > 0:
                    row_entries[3].config(text=f"{gain_needed:.3f}")
                else:
                    row_entries[3].config(text="")
            except Exception:
                row_entries[3].config(text="")
            # Expected 2
            try:
                dbfs_target = float(data["dbfs_ref"][idx])
                gain_needed = self.invert_log_regression(reg_gain2, dbfs_target)
                if gain_needed is not None and np.isfinite(gain_needed) and gain_needed > 0:
                    row_entries[5].config(text=f"{gain_needed:.3f}")
                else:
                    row_entries[5].config(text="")
            except Exception:
                row_entries[5].config(text="")

    def update_calib_plot(self, reg_gain1, reg_gain2):
        freq = self.calib_freq.get()
        data = self.calib_data[freq]
        self.calib_ax.clear()
        self.calib_ax.set_facecolor("#2e2e2e")
        self.calib_fig.patch.set_facecolor("#2e2e2e")
        self.calib_ax.tick_params(colors="white")
        for spine in self.calib_ax.spines.values():
            spine.set_color("white")
        self.calib_ax.set_xlabel("dB HL", color="white")
        self.calib_ax.set_ylabel("dBFS", color="white")
        self.calib_ax.title.set_color("white")
        self.calib_ax.set_title(f"Calibration curves {freq} Hz")
    
        # Reference dBFS (medida)
        x_ref, y_ref = [], []
        for i, v in enumerate(data["dbfs_ref"]):
            try:
                y = float(v)
                x = data["db_hl"][i]
                x_ref.append(x)
                y_ref.append(y)
            except:
                continue
        if len(x_ref) > 0:
            self.calib_ax.plot(x_ref, y_ref, "o-", label="Reference dBFS", color="#2E86C1")
    
        eq_texts = []
        
        # Expected for Device 1
        if reg_gain1 is not None:
            a1, b1 = reg_gain1
            dbfs_expected1 = []
            x_real1, y_real1 = [], []
            for idx, x in enumerate(x_ref):
                try:
                    gain = float(data["gain_1"][DB_HL_STEPS.index(x)])
                    if gain > 0:
                        dbfs_calc = a1 * np.log10(gain) + b1
                        dbfs_expected1.append(dbfs_calc)
                        x_real1.append(x)
                        y_real1.append(dbfs_calc)
                    else:
                        dbfs_expected1.append(np.nan)
                except:
                    dbfs_expected1.append(np.nan)
            self.calib_ax.plot(x_ref, dbfs_expected1, "x--", label="Expected Device 1 dBFS", color="#28B463")
            if len(x_real1) > 0:
                self.calib_ax.plot(x_real1, y_real1, "o", label="Measured Device 1 dBFS", color="#E67E22")
            r2_1 = self.r2_log_fit(data["gain_1"], data["dbfs_ref"], reg_gain1)
            eq_texts.append(f"Device 1: dBFS = {a1:.3f}·log10(Gain) + {b1:.2f}   (R² = {r2_1:.3f})")
            coefs_1 = self.get_gain_to_dbhl_regression(data["gain_1"], data["db_hl"])
            if coefs_1 is not None:
                m1, n1 = coefs_1
                eq_texts.append(f"Device 1: dB HL = {m1:.3f}·log10(Gain) + {n1:.2f}")
    
        # Expected for Device 2
        if reg_gain2 is not None:
            a2, b2 = reg_gain2
            dbfs_expected2 = []
            x_real2, y_real2 = [], []
            for idx, x in enumerate(x_ref):
                try:
                    gain = float(data["gain_2"][DB_HL_STEPS.index(x)])
                    if gain > 0:
                        dbfs_calc = a2 * np.log10(gain) + b2
                        dbfs_expected2.append(dbfs_calc)
                        x_real2.append(x)
                        y_real2.append(dbfs_calc)
                    else:
                        dbfs_expected2.append(np.nan)
                except:
                    dbfs_expected2.append(np.nan)
            self.calib_ax.plot(x_ref, dbfs_expected2, "x--", label="Expected Device 2 dBFS", color="#E74C3C")
            if len(x_real2) > 0:
                self.calib_ax.plot(x_real2, y_real2, "o", label="Measured Device 2 dBFS", color="#FFB300")
            r2_2 = self.r2_log_fit(data["gain_2"], data["dbfs_ref"], reg_gain2)
            eq_texts.append(f"Device 2: dBFS = {a2:.3f}·log10(Gain) + {b2:.2f}   (R² = {r2_2:.3f})")
            coefs_2 = self.get_gain_to_dbhl_regression(data["gain_2"], data["db_hl"])
            if coefs_2 is not None:
                m2, n2 = coefs_2
                eq_texts.append(f"Device 2: dB HL = {m2:.3f}·log10(Gain) + {n2:.2f}")
        
        self.calib_ax.legend(loc="best", facecolor="#2e2e2e", edgecolor="white", labelcolor="white")
        self.calib_ax.grid(True, color="#888")
        self.calib_canvas.draw()
        self.regression_text.set("\n".join(eq_texts) if eq_texts else "")

    def calc_r2(self, x, y, fit):
        y_pred = fit(x)
        ss_res = np.sum((np.array(y) - y_pred) ** 2)
        ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    def r2_log_fit(self, gain_list, dbfs_list, coefs):
        """R² for log-fit: y = a * log10(gain) + b"""
        x, y = [], []
        for g, d in zip(gain_list, dbfs_list):
            try:
                g = float(g)
                d = float(d)
                if g > 0:
                    x.append(np.log10(g))
                    y.append(d)
            except:
                continue
        if len(x) < 2:
            return float('nan')
        a, b = coefs
        y_pred = a * np.array(x) + b
        ss_res = np.sum((np.array(y) - y_pred) ** 2)
        ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    def clear_calibration_data(self):
        freq = self.calib_freq.get()
        for col in ["dbfs_ref", "gain_1", "gain_2", "dbfs_1", "dbfs_2"]:
            self.calib_data[freq][col] = ["" for _ in DB_HL_STEPS]
        self.update_calib_table_and_plot()

    def save_calibration(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.calib_data, f, indent=2)
            messagebox.showinfo("Save Calibration", "Calibration data saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save calibration:\n{e}")

    def load_calibration(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for freq in FREQUENCIES:
                key_freq = str(freq)
                if key_freq in loaded:
                    for col in ["db_hl", "dbfs_ref", "dbfs_1", "dbfs_2", "gain_1", "gain_2"]:
                        if col in loaded[key_freq]:
                            self.calib_data[freq][col] = loaded[key_freq][col]
            self.update_calib_table_and_plot()
            messagebox.showinfo("Load Calibration", "Calibration data loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load calibration:\n{e}")
            
    def load_default_dbfs_ref(self):
        # Load example reference data
        for freq in FREQUENCIES:
            self.calib_data[freq]["dbfs_ref"] = DEFAULT_DBFS_REF[freq].copy()
        self.update_calib_table_and_plot()
        messagebox.showinfo("Default loaded", "Default dBFS ref values loaded successfully.")

    # === Spectrum analyzer methods ===
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
                self.auto_peak = {"freq": int(peak_freq), "val": float(peak_val)}
            else:
                peak_freq = 0
                peak_val = 0
                self.auto_marker.set_xdata([0])
                self.auto_text.set_text("")
                self.auto_peak = {"freq": None, "val": None}
            manual_freq = self.manual_freq.get()
            if manual_freq == 0:
                self.manual_marker.set_xdata([0])
                self.manual_text.set_text("")
                manual_val = 0
                self.manual_peak = {"freq": None, "val": None}
            else:
                idx = np.argmin(np.abs(self.freqs - manual_freq))
                manual_val = self.magnitude[idx]
                self.manual_marker.set_xdata([manual_freq])
                self.manual_text.set_position((manual_freq, manual_val))
                self.manual_text.set_text(f"{int(manual_freq)} Hz\n{manual_val:.1f} dB")
                self.manual_peak = {"freq": int(manual_freq), "val": float(manual_val)}
            self.status.set(
                f"Auto peak: {int(peak_freq)} Hz / {peak_val:.1f} dB   |   "
                f"Manual peak: {int(manual_freq)} Hz / {manual_val:.1f} dB"
            )
            self.update_calib_peaks()
            self.canvas.draw()
        self.root.after(self.update_interval, self.update_plot)

    def update_calib_peaks(self):
        # Synchronize calibration tab with current peaks
        if self.auto_peak["freq"] is not None and self.auto_peak["val"] is not None:
            txt = f"Auto peak: {self.auto_peak['freq']} Hz / {self.auto_peak['val']:.1f} dB"
        else:
            txt = "Auto peak: N/A"
        self.calib_auto_peak_text.set(txt)
        if self.manual_peak["freq"] is not None and self.manual_peak["val"] is not None:
            txt = f"Manual peak: {self.manual_peak['freq']} Hz / {self.manual_peak['val']:.1f} dB"
        else:
            txt = "Manual peak: N/A"
        self.calib_manual_peak_text.set(txt)

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
        for child in self.spectrum_frame.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.OptionMenu)):
                child.configure(state="disabled")

    def enable_controls(self):
        for child in self.spectrum_frame.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.OptionMenu)):
                child.configure(state="normal")

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