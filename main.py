"""
Patch microstrip GUI (PyQt5)
- calcula dimensões do patch (W, L, eps_eff, deltaL)
- estima posição do inset feed para casar a 50 ohm (R_in(y) = R_edge * cos^2(pi*y/L))
- estima S11 usando modelo RLC série (R = R_in @ feed, L/C from Q estimate)
- plota cortes E/H aproximados e S11 (dB)
Author: ChatGPT (exemplo didático)
References: see in-app comments and printed citations.
"""

import sys
import numpy as np
from math import pi
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFormLayout, QGroupBox, QComboBox, QSpinBox, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# --- Constants ---
c = 299792458.0
Z0 = 50.0  # reference impedance for matching

# --- Helper electromagnetic functions (formulas classical) ---
def calc_patch_dimensions(f0, eps_r, h):
    lam0 = c / f0
    W = (c / (2.0 * f0)) * np.sqrt(2.0 / (eps_r + 1.0))
    eps_eff = (eps_r + 1.0)/2.0 + (eps_r - 1.0)/2.0 * (1.0 + 12.0 * h / W)**(-0.5)
    F = (eps_eff + 0.3) * (W/h + 0.264) / ((eps_eff - 0.258) * (W/h + 0.8))
    deltaL = 0.412 * h * F
    L = c / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * deltaL
    return {'W': W, 'L': L, 'eps_eff': eps_eff, 'deltaL': deltaL, 'lambda0': lam0}

# R_in variation with inset feed position (y measured from edge towards center)
# R_in(y) = R_edge * cos^2(pi * y / L)
def Rin_from_y(y, L, R_edge):
    # y in meters (distance from edge along length)
    arg = (pi * y / L)
    return R_edge * (np.cos(arg)**2)

# invert for y given target R_target: y = (L/pi) * arccos( sqrt(R_target / R_edge) )
def y_from_Rtarget(R_target, L, R_edge):
    val = R_target / R_edge
    if val > 1.0:
        return None  # impossible to reach (R_target > R_edge)
    if val < 0.0:
        return None
    return (L / pi) * np.arccos(np.sqrt(val))

# simplified pattern model (slot-pair approximate) returning normalized field magnitude
def pattern_slot_pair(theta, phi, dims, f):
    k = 2.0 * pi * f / c
    W = dims['W']
    L_eff = dims['L'] + 2.0 * dims['deltaL']
    u = (k * W / 2.0) * np.sin(theta) * np.sin(phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        E_elem = np.where(np.abs(u) < 1e-12, 1.0, np.sin(u) / u)
    psi = (k * L_eff / 2.0) * np.cos(theta)
    AF = np.cos(psi)
    E = np.abs(E_elem * AF)
    return E / np.max(E)

# estimate fractional bandwidth (empirical approx from lecture material)
# We'll use a practical estimate: fractional BW ~ C1 * (h/lambda0) * (W/L) * F(eps_r)
# We use the coefficient and dependence from common slides (gives typical 1-5% for common patches).
def estimate_fractional_BW(h, lambda0, W, L, eps_r):
    # empirical formula (slide-based approximation). Good for order-of-magnitude.
    # fractional BW ~ 3.77 * (eps_r - 1)/eps_r^2 * (h/lambda0) * (W/L)
    # source: lecture notes / compiled formulas (empirical). See cited literature.
    factor = 3.77 * (eps_r - 1.0) / (eps_r**2)
    bw_frac = max(0.0005, factor * (h / lambda0) * (W / L))  # avoid zero
    return bw_frac

# Build series RLC from R (R_in at feed) and Q (from fractional BW)
# For series RLC: Q = omega0 * L / R  => L = Q*R/omega0 ; C = 1/(omega0^2 * L) = 1/(omega0 * Q * R)
def rlc_from_R_and_Q(R, f0, Q):
    w0 = 2.0 * pi * f0
    if R <= 0 or Q <= 0:
        return None
    L = Q * R / w0
    C = 1.0 / (w0 * Q * R)
    return {'R': R, 'L': L, 'C': C}

# compute Zin(f) for series RLC
def Zin_series_RLC(freqs, rlc):
    w = 2.0 * pi * freqs
    R = rlc['R']; L = rlc['L']; C = rlc['C']
    Zin = R + 1j * (w * L - 1.0/(w * C))
    return Zin

# S11 from Zin to Z0
def S11_from_Zin(Zin, Z0=50.0):
    return (Zin - Z0) / (Zin + Z0)

# --- PyQt GUI ---
class PatchDesigner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patch Microstrip Designer (2.4 GHz exemplo)")
        self.setMinimumSize(1000, 700)
        self.init_ui()

    def init_ui(self):
        # Input form
        form = QFormLayout()

        self.f0_edit = QLineEdit("2.45e9")  # Hz
        self.epsr_edit = QLineEdit("4.4")
        self.h_edit = QLineEdit("1.6e-3")  # m
        self.eff_edit = QLineEdit("0.85")
        self.Redge_edit = QLineEdit("243")  # ohm, empirical default (can be changed)
        self.Z0_edit = QLineEdit("50")
        form.addRow("Freq target f0 (Hz):", self.f0_edit)
        form.addRow("Dielectric εr:", self.epsr_edit)
        form.addRow("Substrate height h (m):", self.h_edit)
        form.addRow("Eficiência (0-1):", self.eff_edit)
        form.addRow("R_edge (Ω) [default emp. ~243Ω]:", self.Redge_edit)
        form.addRow("Linha referência Z0 (Ω):", self.Z0_edit)

        btn = QPushButton("Calcular & Plotar")
        btn.clicked.connect(self.on_calculate)

        # Results text area
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Matplotlib canvas
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12,4))
        plt.tight_layout()
        self.canvas = FigureCanvas(self.fig)

        left_layout = QVBoxLayout()
        left_group = QGroupBox("Parâmetros")
        vbox_form = QVBoxLayout()
        vbox_form.addLayout(form)
        vbox_form.addWidget(btn)
        left_group.setLayout(vbox_form)
        left_layout.addWidget(left_group)
        left_layout.addWidget(self.results_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.canvas, 2)

        self.setLayout(main_layout)

    def on_calculate(self):
        # read inputs safely
        try:
            f0 = float(self.f0_edit.text())
            epsr = float(self.epsr_edit.text())
            h = float(self.h_edit.text())
            eff = float(self.eff_edit.text())
            R_edge = float(self.Redge_edit.text())
            Z0_local = float(self.Z0_edit.text())
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Parâmetros inválidos: {e}")
            return

        dims = calc_patch_dimensions(f0, epsr, h)
        W = dims['W']; L = dims['L']; deltaL = dims['deltaL']; lam0 = dims['lambda0']
        L_eff = L + 2.0 * deltaL

        # compute R_target = desired 50 ohm (could be any Z0)
        R_target = Z0_local

        # compute feed inset y position from edge (meters)
        ypos = y_from_Rtarget(R_target, L, R_edge)
        y_text = ("Impossível com o R_edge atual (R_target > R_edge). "
                  "Aumente R_edge ou escolha edge feed.") if ypos is None else f"{ypos*1e3:.3f} mm from edge (inset)"

        # compute Rin at that y
        R_in_feed = Rin_from_y(ypos, L, R_edge) if ypos is not None else None

        # estimate BW fraction and Q
        bw_frac = estimate_fractional_BW(h, lam0, W, L, epsr)
        Q_est = 1.0 / bw_frac

        # build RLC
        if R_in_feed is None:
            R_use = R_edge  # fallback: assume edge feed
            note = "Usando R_edge como R (feed na borda)."
        else:
            R_use = R_in_feed
            note = "Usando R_in no ponto do inset feed."

        rlc = rlc_from_R_and_Q(R_use, f0, Q_est)
        if rlc is None:
            QMessageBox.warning(self, "Erro", "Não foi possível criar modelo RLC (valores inválidos).")
            return

        # frequency sweep for S11
        fspan = np.linspace(f0 * 0.95, f0 * 1.05, 801)
        Zin = Zin_series_RLC(fspan, rlc)
        S11 = S11_from_Zin(Zin, Z0_local)
        S11_dB = 20.0 * np.log10(np.abs(S11) + 1e-12)

        # pattern cuts
        thetas = np.linspace(0.0, np.pi, 400)
        E_phi90 = pattern_slot_pair(thetas, np.deg2rad(90.0), dims, f0)
        E_phi0 = pattern_slot_pair(thetas, 0.0, dims, f0)
        E_phi90_dB = 20.0 * np.log10(E_phi90 + 1e-9)
        E_phi0_dB = 20.0 * np.log10(E_phi0 + 1e-9)

        # estimated gain (approx)
        area = W * L_eff
        D_lin = 4.0 * pi * area / (lam0**2)
        G_est = D_lin * eff
        G_dBi = 10.0 * np.log10(G_est + 1e-12)

        # update plots
        self.axes[0].clear()
        self.axes[0].plot(np.degrees(thetas), E_phi90_dB, label='E-plane (phi=90°)')
        self.axes[0].plot(np.degrees(thetas), E_phi0_dB, label='H-plane (phi=0°)')
        self.axes[0].set_ylim(-40, 0)
        self.axes[0].set_xlabel('Theta (deg)')
        self.axes[0].set_ylabel('Magnitude (dB)')
        self.axes[0].set_title('Cortes aproximados (E & H)')
        self.axes[0].grid(True)
        self.axes[0].legend()

        self.axes[1].clear()
        self.axes[1].plot((fspan - f0)/1e6, S11_dB)
        self.axes[1].set_xlabel('Freq offset (MHz)')
        self.axes[1].set_ylabel('|S11| (dB)')
        self.axes[1].set_title('S11 aproximado (modelo RLC série)')
        self.axes[1].grid(True)
        # mark resonant freq
        self.axes[1].axvline(0.0, color='gray', linestyle='--')

        self.axes[2].clear()
        # show a small schematic: patch dims text / bar visual
        txt = (f"f0 = {f0/1e9:.3f} GHz\nW = {W*1e3:.3f} mm\nL = {L*1e3:.3f} mm\n"
               f"L_eff = {L_eff*1e3:.3f} mm\nε_eff = {dims['eps_eff']:.4f}\nΔL = {deltaL*1e3:.4f} mm\n\n"
               f"R_edge (assumed) = {R_edge:.1f} Ω\nR_in@feed ≈ {R_use:.2f} Ω\nInset pos: {y_text}\n\n"
               f"Est. ganho ≈ {G_dBi:.2f} dBi\nFrac. BW ≈ {bw_frac*100:.2f}% (Q≈{Q_est:.1f})\nModel RLC: R={rlc['R']:.1f}Ω, L={rlc['L']*1e9:.2f}nH, C={rlc['C']*1e12:.2f}pF\n\n"
               "Observações:\n- Modelo didático; valide em solver numérico (openEMS/CST/HFSS).\n- Fórmulas: R_in(y)=R_edge*cos^2(pi*y/L). BW empírica usada para Q.")
        self.axes[2].text(0.02, 0.02, txt, fontsize=9, va='bottom', ha='left', family='monospace')
        self.axes[2].axis('off')

        self.canvas.draw()

        # show results summary in label (also provide citations)
        res_html = (f"<b>Resumo:</b><br>"
                    f"Patch: W={W*1e3:.3f} mm, L={L*1e3:.3f} mm, L_eff={L_eff*1e3:.3f} mm, ε_eff={dims['eps_eff']:.3f}<br>"
                    f"Inset feed (from edge): {y_text}<br>"
                    f"R_in @ feed ≈ {R_use:.2f} Ω  •  Est. ganho ≈ {G_dBi:.2f} dBi • Frac. BW ≈ {bw_frac*100:.2f}%<br>"
                    f"<i>Referências rápidas:</i> relação R_in(y) (lecture notes / microwavetools), BW empírica (lecture slides).")
        self.results_label.setText(res_html)
        # print small citation pointers in console for traceability
        print("Citações (web): R_in(y)=R_edge*cos^2(pi y/L) — microwavetools / lecture notes. (see code comment)")
        print("Bw empírica (approx) — lecture notes / compiled formulas (used to get Q).")

def main():
    app = QApplication(sys.argv)
    win = PatchDesigner()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
