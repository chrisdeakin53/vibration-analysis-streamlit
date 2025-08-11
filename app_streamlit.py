import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew, norm
from docx import Document
from docx.shared import Inches

st.set_page_config(page_title="Vibration Analyzer", layout="wide")

# ---------------- Utility functions ----------------
def rms_raw(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.sqrt(np.mean(np.square(x))) if x.size else np.nan

def crest_factor(x):
    r = rms_raw(x)
    return np.nan if not np.isfinite(r) or r == 0 else np.nanmax(np.abs(x))/r

def rfft_mag_rect(x, fs):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or fs <= 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0], complex)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    mag = np.abs(X) * 2.0 / n
    mag[0] = np.abs(X[0]) / n
    return f, mag, X

def velocity_fft_integrate(a_g, fs, min_hz=10.0):
    a_ms2 = np.nan_to_num((np.asarray(a_g, float) - np.nanmean(a_g)) * 9.80665, nan=0.0)
    A = np.fft.rfft(a_ms2)
    f = np.fft.rfftfreq(len(a_ms2), d=1.0/fs)
    denom = 2j * np.pi * f
    denom[0] = np.inf
    V = A / denom
    V[0] = 0.0 + 0j
    if min_hz > 0:
        V[f < float(min_hz)] = 0.0 + 0j
    v_ms = np.fft.irfft(V, n=len(a_ms2))
    v_ms -= np.nanmean(v_ms)
    return v_ms

def shock_pulse(a_g, fs):
    rect = np.nan_to_num(np.abs(a_g), nan=0.0)
    if fs < 2100:
        return rect, fs
    decim = max(int(np.floor(fs / 2000.0)), 1)
    sp = resample_poly(rect, up=1, down=decim)
    fs_sp = fs / decim
    return sp, fs_sp

def read_csv_flexible(file):
    df = pd.read_csv(file)
    cols = [c.strip().lower() for c in df.columns]
    time_col = None
    accel_col = None
    for i, c in enumerate(cols):
        if "time" in c:
            time_col = df.columns[i]
        if "acc" in c or c in ["g", "amp", "amplitude", "dynamic (g)"]:
            accel_col = df.columns[i]
    if time_col is None:
        time_col = df.columns[0]
    if accel_col is None:
        accel_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if accel_col == time_col and len(df.columns) > 1:
            accel_col = df.columns[1]
    t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    a_g = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
    t = t_raw / 1000.0
    mask = np.isfinite(t) & np.isfinite(a_g)
    t = t[mask]
    a_g = a_g[mask]
    if t.size > 1:
        idx = np.argsort(t)
        t = t[idx]
        a_g = a_g[idx]
    return t, a_g

# ---------------- Streamlit UI ----------------
st.title("Vibration Analysis Tool")

uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

running_speed = st.number_input("Running speed (RPM)", min_value=0.0, value=0.0, step=1.0)
toggle_orders = st.checkbox("Show spectra in orders instead of Hz", value=False)
waveform_type_for_polar = st.selectbox("Polar plot waveform", ["Acceleration", "Shock pulse"])

if uploaded_files:
    doc = Document()
    all_metrics = []

    for file in uploaded_files:
        st.subheader(file.name)
        t, a_g = read_csv_flexible(file)
        if len(t) < 2:
            st.warning(f"Not enough data in {file.name}")
            continue
        dt = np.median(np.diff(t))
        fs = 1.0 / dt

        # Velocity and shock pulse
        v_ms = velocity_fft_integrate(a_g, fs, min_hz=10.0)
        v_mms = v_ms * 1000.0
        sp, fs_sp = shock_pulse(a_g, fs)

        # FFTs
        f_acc, A_mag, _ = rfft_mag_rect(a_g, fs)
        f_vel, V_mag_ms, _ = rfft_mag_rect(v_ms, fs)
        V_mag = V_mag_ms * 1000.0
        f_sp, SP_mag, _ = rfft_mag_rect(sp, fs_sp)

        # Convert to orders if needed
        if toggle_orders and running_speed > 0:
            speed_hz = running_speed / 60.0
            f_acc = f_acc / speed_hz
            f_vel = f_vel / speed_hz
            f_sp = f_sp / speed_hz
            x_label = "Orders"
        else:
            x_label = "Frequency (Hz)"

        # Plots side-by-side
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes[0,0].plot(t, a_g); axes[0,0].set_title("Acceleration waveform"); axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("G")
        axes[1,0].plot(t, v_mms); axes[1,0].set_title("Velocity waveform"); axes[1,0].set_xlabel("Time (s)"); axes[1,0].set_ylabel("mm/s")
        axes[2,0].plot(t[:len(sp)], sp); axes[2,0].set_title("Shock pulse waveform"); axes[2,0].set_xlabel("Time (s)"); axes[2,0].set_ylabel("G")
        axes[0,1].plot(f_acc, A_mag); axes[0,1].set_title("Acceleration spectrum"); axes[0,1].set_xlabel(x_label)
        axes[1,1].plot(f_vel, V_mag); axes[1,1].set_title("Velocity spectrum"); axes[1,1].set_xlabel(x_label)
        axes[1,1].set_xlim(10, 1000) if not toggle_orders else None
        axes[2,1].plot(f_sp, SP_mag); axes[2,1].set_title("Shock pulse spectrum"); axes[2,1].set_xlabel(x_label)
        plt.tight_layout()
        st.pyplot(fig)

        # Polar plot
        if running_speed > 0:
            if waveform_type_for_polar == "Acceleration":
                waveform = a_g
            else:
                waveform = sp
            samples_per_rev = int(round(fs / (running_speed / 60.0)))
            revs = len(waveform) / samples_per_rev
            theta = np.linspace(0, 2*np.pi, samples_per_rev, endpoint=False)
            polar_fig = plt.figure()
            axp = polar_fig.add_subplot(111, polar=True)
            axp.plot(theta, waveform[:samples_per_rev])
            axp.set_title(f"Polar plot: {waveform_type_for_polar}")
            st.pyplot(polar_fig)

        # Metrics
        metrics = {
            "Acceleration": dict(peak=float(np.nanmax(np.abs(a_g))), rms=float(rms_raw(a_g)), kurt=float(kurtosis(a_g, fisher=False)), skew=float(skew(a_g))),
            "Velocity": dict(peak=float(np.nanmax(np.abs(v_mms))), rms=float(rms_raw(v_mms)), kurt=float(kurtosis(v_mms, fisher=False)), skew=float(skew(v_mms))),
            "Shock": dict(peak=float(np.nanmax(np.abs(sp))), rms=float(rms_raw(sp)), kurt=float(kurtosis(sp, fisher=False)), skew=float(skew(sp)))
        }
        all_metrics.append((file.name, metrics))

        # Show table
        st.write(pd.DataFrame(metrics).T)

    # Comparison plots
    if len(all_metrics) > 1:
        names = [nm for nm, _ in all_metrics]
        def collect(sig, key):
            return [m[sig][key] for _, m in all_metrics]
        # Example: acceleration peak & RMS
        fig, ax = plt.subplots()
        idx = np.arange(len(names)); width = 0.35
        ax.bar(idx - width/2, collect("Acceleration", "peak"), width, label="Peak")
        ax.bar(idx + width/2, collect("Acceleration", "rms"), width, label="RMS")
        ax.set_xticks(idx); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        st.pyplot(fig)
