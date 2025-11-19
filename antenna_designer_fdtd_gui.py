import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
                             QLabel, QTabWidget, QSplitter, QProgressBar, QMessageBox,
                             QScrollArea)
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# --- IMPORTA O MOTOR FDTD (O ARQUIVO QUE VOCÊ CRIOU NO PASSO 2) ---
try:
    from fdtd_engine_numba import FDTDGrid, modulated_gaussian_pulse, calculate_s11, c0
except ImportError:
    print("ERRO: Não foi possível importar 'fdtd_engine_numba.py'.")
    print("Certifique-se de que ambos os arquivos estão na mesma pasta.")
    sys.exit(1)

# --- MODELO ANALÍTICO (Apenas para estimativa inicial de dimensões) ---
class PatchAntennaModel:
    def __init__(self, freq_ghz, er, h_mm):
        self.f0 = freq_ghz * 1e9
        self.er = er
        self.h = h_mm * 1e-3
        self.c = 3e8

    def calculate_dimensions(self):
        # Largura (W)
        self.W = (self.c / (2 * self.f0)) * np.sqrt(2 / (self.er + 1))
        # Epsilon Efetivo
        self.e_eff = (self.er + 1) / 2 + (self.er - 1) / 2 * (1 / np.sqrt(1 + 12 * self.h / self.W))
        # Delta L
        numer = (self.e_eff + 0.3) * (self.W / self.h + 0.264)
        denom = (self.e_eff - 0.258) * (self.W / self.h + 0.8)
        self.dL = 0.412 * self.h * (numer / denom)
        # Comprimento (L)
        self.L_eff = self.c / (2 * self.f0 * np.sqrt(self.e_eff))
        self.L = self.L_eff - 2 * self.dL
        return self.W, self.L, self.e_eff

    def get_radiation_pattern(self):
        # Gera padrão 3D analítico simples para visualização
        theta = np.linspace(0, np.pi/2, 45)
        phi = np.linspace(0, 2*np.pi, 90)
        THETA, PHI = np.meshgrid(theta, phi)
        R = np.abs(np.cos(THETA)) * 0.8 + 0.2
        X_grid = R * np.sin(THETA) * np.cos(PHI)
        Y_grid = R * np.sin(THETA) * np.sin(PHI)
        Z_grid = R * np.cos(THETA)
        return X_grid, Y_grid, Z_grid

# --- COMPONENTES DE PLOTAGEM ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# --- WORKER FDTD (THREAD SEPARADA) ---
class FDTDWorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

class FDTDWorker(QRunnable):
    def __init__(self, fdtd_params, antenna_dims_m):
        super().__init__()
        self.signals = FDTDWorkerSignals()
        self.fdtd_params = fdtd_params
        self.antenna_dims_m = antenna_dims_m

    def run(self):
        try:
            self.signals.log.emit("Iniciando configuração do Grid FDTD...")
            freq_ghz = self.fdtd_params['freq_ghz']
            er = self.fdtd_params['er']
            h_mm = self.fdtd_params['h_mm']
            W_m, L_m = self.antenna_dims_m['W_m'], self.antenna_dims_m['L_m']
            h_m = self.antenna_dims_m['h_m']

            # Resolução do Grid (Células por Lambda)
            lambda_min = c0 / (freq_ghz * 1e9 * np.sqrt(er))
            dx = lambda_min / 15.0 
            dy = dx
            dz = dx / 2.0 # Mais fino em Z

            # Margens e PML
            pml = 8
            margin_xy = 0.5 * lambda_min
            Nx = int(np.ceil((W_m + 2 * margin_xy) / dx)) + 2 * pml
            Ny = int(np.ceil((L_m + 2 * margin_xy) / dy)) + 2 * pml
            Nz = int(np.ceil((h_m + lambda_min) / dz)) + 2 * pml

            self.signals.log.emit(f"Grid size: {Nx}x{Ny}x{Nz} (Total cells: {Nx*Ny*Nz:,})")
            
            grid = FDTDGrid(Nx, Ny, Nz, dx, dy, dz, cpml_thickness=pml)

            # Posições
            center_x, center_y = Nx//2, Ny//2
            patch_w_idx = int(W_m / dx)
            patch_l_idx = int(L_m / dy)
            sub_h_idx = int(h_m / dz)
            
            z_ground = pml + 5
            z_patch = z_ground + sub_h_idx

            # Materiais
            # Substrato
            grid.update_material(0, Nx, 0, Ny, z_ground, z_patch, er)
            
            # Patch (PEC)
            x0 = center_x - patch_w_idx//2
            x1 = center_x + patch_w_idx//2
            y0 = center_y - patch_l_idx//2
            y1 = center_y + patch_l_idx//2
            
            grid.add_pec_patch(x0, x1, y0, y1, z_patch, z_patch+1)
            
            # Plano Terra (PEC - Infinito no plano XY da simulação)
            grid.add_pec_patch(0, Nx, 0, Ny, z_ground, z_ground+1)

            # Fonte (Feed Point) - Deslocado do centro para casar impedância
            feed_offset_y = int(patch_l_idx * 0.15) # Típico ponto de alimentação
            feed_x = center_x
            feed_y = center_y - feed_offset_y
            
            fc = freq_ghz * 1e9
            t_pulse = 1.0 / fc
            grid.add_source(feed_x, feed_y, z_patch, 
                           (modulated_gaussian_pulse, 3*t_pulse, t_pulse/2, fc), 'Ez')
            
            # Probes
            grid.add_probe('feed', feed_x, feed_y, z_patch)

            # Rodar
            n_steps = 1000 # Passos de simulação (aumente para maior precisão, ex: 2000-4000)
            self.signals.log.emit(f"Rodando {n_steps} passos de tempo...")
            
            grid.run(n_steps, progress_callback=self.signals.progress.emit)

            # Pós-processamento S11
            self.signals.log.emit("Calculando S11...")
            feed_data = np.array(grid.probes['feed']['data'])
            
            # Técnica simples de S11: FFT do sinal total (não é perfeito mas serve para demo)
            # Para S11 preciso, precisaríamos separar incidente de refletido,
            # mas isso exige rodar 2 simulações (uma sem a antena).
            # Aqui faremos a FFT direta para ver a ressonância (os "dips" no espectro).
            
            freqs_ghz, s11_db = calculate_s11(feed_data, feed_data, grid.dt, fc)
            # Nota: O calculate_s11 original espera (incidente, refletido). 
            # Passando (total, total) veremos o espectro de potência, onde os DIPS são as ressonâncias.

            results = {
                'freqs_ghz': freqs_ghz,
                's11_db': s11_db, # Na verdade aqui é Power Spectrum Density normalizada
                'W_mm': W_m * 1000,
                'L_mm': L_m * 1000,
                'e_eff': self.antenna_dims_m['e_eff']
            }
            self.signals.finished.emit(results)

        except Exception as e:
            self.signals.error.emit(str(e))
            import traceback
            traceback.print_exc()

# --- GUI PRINCIPAL ---
class AntennaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador FDTD Hardcore - Antena Patch")
        self.setGeometry(100, 100, 1100, 700)
        
        self.threadpool = QThreadPool()

        # Layout Principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)

        # --- Painel Esquerdo ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.input_freq = QLineEdit("2.4")
        self.input_er = QLineEdit("4.4")
        self.input_h = QLineEdit("1.6")
        
        form_layout.addRow("Frequência (GHz):", self.input_freq)
        form_layout.addRow("Substrato Er:", self.input_er)
        form_layout.addRow("Espessura h (mm):", self.input_h)
        
        self.btn_simulate = QPushButton("Rodar Simulação FDTD")
        self.btn_simulate.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        self.btn_simulate.clicked.connect(self.start_simulation)
        
        self.progress_bar = QProgressBar()
        self.lbl_status = QLabel("Pronto.")
        self.lbl_status.setWordWrap(True)

        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.btn_simulate)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.lbl_status)
        controls_layout.addStretch()
        
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

        # --- Painel Direito (Abas) ---
        self.tabs = QTabWidget()
        
        self.canvas_geo = MplCanvas(self, is_3d=False)
        self.tabs.addTab(self.canvas_geo, "Geometria Calculada")
        
        self.canvas_s11 = MplCanvas(self, is_3d=False)
        self.tabs.addTab(self.canvas_s11, "Espectro FDTD (Ressonância)")

        self.canvas_3d = MplCanvas(self, is_3d=True)
        self.tabs.addTab(self.canvas_3d, "Padrão 3D (Teórico)")

        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 800])
        layout.addWidget(splitter)

    def start_simulation(self):
        # Bloqueia botão
        self.btn_simulate.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Pega Inputs
        try:
            freq = float(self.input_freq.text())
            er = float(self.input_er.text())
            h = float(self.input_h.text())
        except ValueError:
            QMessageBox.critical(self, "Erro", "Valores inválidos.")
            self.btn_simulate.setEnabled(True)
            return

        # Calcula dimensões físicas (teóricas)
        model = PatchAntennaModel(freq, er, h)
        W, L, e_eff = model.calculate_dimensions()
        
        # Prepara dados para o Worker
        fdtd_params = {'freq_ghz': freq, 'er': er, 'h_mm': h}
        dims = {'W_m': W, 'L_m': L, 'h_m': h * 1e-3, 'e_eff': e_eff}

        # Instancia e roda Worker
        worker = FDTDWorker(fdtd_params, dims)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.log.connect(self.update_status)
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.simulation_error)
        
        self.threadpool.start(worker)
        
        # Atualiza geometria visual imediatamente
        self.plot_geometry(W, L)
        self.plot_3d_theoretical(model)

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def update_status(self, text):
        self.lbl_status.setText(text)

    def simulation_error(self, err):
        QMessageBox.critical(self, "Erro na Simulação", err)
        self.btn_simulate.setEnabled(True)
        self.lbl_status.setText("Erro.")

    def simulation_finished(self, results):
        self.btn_simulate.setEnabled(True)
        self.lbl_status.setText("Simulação Concluída.")
        
        # Plotar S11 (Espectro)
        self.canvas_s11.axes.cla()
        f = results['freqs_ghz']
        s11 = results['s11_db']
        
        # Plotar apenas banda de interesse
        center_f = float(self.input_freq.text())
        mask = (f > center_f * 0.5) & (f < center_f * 1.5)
        
        self.canvas_s11.axes.plot(f[mask], s11[mask], color='blue')
        self.canvas_s11.axes.set_title("Resposta em Frequência (FDTD)")
        self.canvas_s11.axes.set_xlabel("Frequência (GHz)")
        self.canvas_s11.axes.set_ylabel("Magnitude (dB)")
        self.canvas_s11.axes.grid(True)
        self.canvas_s11.draw()

    def plot_geometry(self, W, L):
        self.canvas_geo.axes.cla()
        margin = max(W, L) * 0.5
        
        rect_sub = plt.Rectangle((-W/2 - margin, -L/2 - margin), W + 2*margin, L + 2*margin, 
                                 color='green', alpha=0.3)
        rect_patch = plt.Rectangle((-W/2, -L/2), W, L, color='orange', alpha=0.8)
        
        self.canvas_geo.axes.add_patch(rect_sub)
        self.canvas_geo.axes.add_patch(rect_patch)
        self.canvas_geo.axes.set_xlim(-W - margin, W + margin)
        self.canvas_geo.axes.set_ylim(-L - margin, L + margin)
        self.canvas_geo.axes.set_aspect('equal')
        self.canvas_geo.axes.set_title(f"Dimensões: W={W*1000:.2f}mm, L={L*1000:.2f}mm")
        self.canvas_geo.draw()

    def plot_3d_theoretical(self, model):
        self.canvas_3d.axes.cla()
        X, Y, Z = model.get_radiation_pattern()
        self.canvas_3d.axes.plot_surface(X, Y, Z, cmap='inferno', alpha=0.8)
        self.canvas_3d.axes.set_title("Padrão Teórico (Modelo Cavidade)")
        self.canvas_3d.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AntennaApp()
    window.show()
    sys.exit(app.exec_())