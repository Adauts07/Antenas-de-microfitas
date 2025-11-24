import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
                             QLabel, QTabWidget, QSplitter, QProgressBar, QMessageBox,
                             QScrollArea, QCheckBox, QComboBox) # Adicionei QComboBox
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# --- IMPORTA O MOTOR FDTD ---
pasta_atual = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pasta_atual)
from fdtd_engine_numba import FDTDGrid, modulated_gaussian_pulse, calculate_s11, c0

# --- MODELO ANALÍTICO (Apenas para Patch) ---
class PatchAntennaModel:
    def __init__(self, freq_ghz, er, h_mm):
        self.f0 = freq_ghz * 1e9
        self.er = er
        self.h = h_mm * 1e-3
        self.c = 3e8

    def calculate_dimensions(self):
        self.W = (self.c / (2 * self.f0)) * np.sqrt(2 / (self.er + 1))
        self.e_eff = (self.er + 1) / 2 + (self.er - 1) / 2 * (1 / np.sqrt(1 + 12 * self.h / self.W))
        numer = (self.e_eff + 0.3) * (self.W / self.h + 0.264)
        denom = (self.e_eff - 0.258) * (self.W / self.h + 0.8)
        self.dL = 0.412 * self.h * (numer / denom)
        self.L_eff = self.c / (2 * self.f0 * np.sqrt(self.e_eff))
        self.L = self.L_eff - 2 * self.dL
        return self.W, self.L, self.e_eff

# --- COMPONENTES DE PLOTAGEM ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        self.fig.tight_layout()
        super(MplCanvas, self).__init__(self.fig)

# --- WORKER FDTD ---
class FDTDWorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    field_update = pyqtSignal(object)

class FDTDWorker(QRunnable):
    def __init__(self, fdtd_params, antenna_dims_m, antenna_type, enable_anim):
        super().__init__()
        self.signals = FDTDWorkerSignals()
        self.fdtd_params = fdtd_params
        self.antenna_dims_m = antenna_dims_m
        self.antenna_type = antenna_type # "Patch", "Dipole", "Yagi"
        self.enable_anim = enable_anim

    def run(self):
        try:
            self.signals.log.emit(f"Configurando Antena: {self.antenna_type}...")
            
            freq_ghz = self.fdtd_params['freq_ghz']
            er = self.fdtd_params['er']
            # h_mm só é usado no Patch, mas deixamos aqui
            
            fc = freq_ghz * 1e9
            lambda_0 = c0 / fc
            
            # --- SELEÇÃO DE GEOMETRIA ---
            
            grid = None
            feed_pos = None
            
            if self.antenna_type == "Microstrip Patch":
                # ==========================================
                # LÓGICA DO PATCH (Mantida do original)
                # ==========================================
                h_m = self.antenna_dims_m['h_m']
                W_m = self.antenna_dims_m['W_m']
                L_m = self.antenna_dims_m['L_m']
                
                lambda_diel = c0 / (freq_ghz * 1e9 * np.sqrt(er))
                dx = lambda_diel / 15.0 
                dy = dx
                dz = dx / 2.0 # Resolução Z maior por causa do substrato fino

                pml = 8
                margin_xy = 0.6 * lambda_diel
                Nx = int(np.ceil((W_m + 2 * margin_xy) / dx)) + 2 * pml
                Ny = int(np.ceil((L_m + 2 * margin_xy) / dy)) + 2 * pml
                Nz = int(np.ceil((h_m + lambda_diel) / dz)) + 2 * pml
                
                grid = FDTDGrid(Nx, Ny, Nz, dx, dy, dz, cpml_thickness=pml)
                
                center_x, center_y = Nx//2, Ny//2
                patch_w_idx = int(W_m / dx)
                patch_l_idx = int(L_m / dy)
                sub_h_idx = int(h_m / dz)
                z_ground = pml + 5
                z_patch = z_ground + sub_h_idx

                # Materiais
                grid.update_material(0, Nx, 0, Ny, z_ground, z_patch, er) # Substrato
                grid.add_pec_patch(center_x - patch_w_idx//2, center_x + patch_w_idx//2,
                                   center_y - patch_l_idx//2, center_y + patch_l_idx//2,
                                   z_patch, z_patch+1) # Patch
                grid.add_pec_patch(0, Nx, 0, Ny, z_ground, z_ground+1) # Terra

                # Fonte (Probe Feed)
                feed_offset_y = int(patch_l_idx * 0.15)
                feed_x, feed_y = center_x, center_y - feed_offset_y
                z_source = z_patch - 1
                
                grid.add_source(feed_x, feed_y, z_source, 
                               (modulated_gaussian_pulse, 3*(1/fc), (1/fc)/2, fc), 'Ez')
                grid.add_probe('feed', feed_x, feed_y, z_source)
                
                self.signals.log.emit(f"Grid Patch: {Nx}x{Ny}x{Nz}")

            elif self.antenna_type == "Dipole (Half-Wave)" or self.antenna_type == "Yagi-Uda":
                # ==========================================
                # LÓGICA DIPOLO / YAGI (No Ar)
                # ==========================================
                
                # Resolução baseada no ar
                dx = lambda_0 / 20.0 
                dy = dx
                dz = dx

                pml = 8
                margin = int(0.7 * lambda_0 / dx)
                
                # Comprimento do Dipolo (~0.48 lambda para ressonância)
                L_dipole_m = 0.47 * lambda_0 
                L_dipole_cells = int(L_dipole_m / dz)
                
                # Se for Yagi, precisa de mais espaço em Y
                y_space_factor = 1.5 if self.antenna_type == "Yagi-Uda" else 1.0
                
                Nx = margin * 2
                Ny = int(margin * 2 * y_space_factor)
                Nz = L_dipole_cells + margin * 2

                grid = FDTDGrid(Nx, Ny, Nz, dx, dy, dz, cpml_thickness=pml)
                
                cx, cy, cz = Nx//2, Ny//2, Nz//2
                gap = 1 # Gap de alimentação
                half_len = L_dipole_cells // 2

                # --- CONSTRÓI O DIPOLO (Elemento Ativo) ---
                # Haste Inferior
                grid.add_pec_patch(cx, cx+1, cy, cy+1, cz - half_len, cz - gap)
                # Haste Superior
                grid.add_pec_patch(cx, cx+1, cy, cy+1, cz + gap, cz + half_len)

                # Se for Yagi, adiciona elementos parasitas
                if self.antenna_type == "Yagi-Uda":
                    
                    # Refletor (Atrás, Maior)
                    dist_ref = int(0.2 * lambda_0 / dy)
                    L_ref = int(L_dipole_cells * 1.10)
                    y_ref = cy - dist_ref
                    grid.add_pec_patch(cx, cx+1, y_ref, y_ref+1, cz - L_ref//2, cz + L_ref//2)
                    
                    # Diretor (Na frente, Menor)
                    dist_dir = int(0.2 * lambda_0 / dy)
                    L_dir = int(L_dipole_cells * 0.90)
                    y_dir = cy + dist_dir
                    grid.add_pec_patch(cx, cx+1, y_dir, y_dir+1, cz - L_dir//2, cz + L_dir//2)
                    
                    self.signals.log.emit("Adicionado Refletor e Diretor (Yagi)")

                # Fonte (No Gap central)
                t_pulse = 1.0 / fc
                grid.add_source(cx, cy, cz, 
                               (modulated_gaussian_pulse, 3*t_pulse, t_pulse/2, fc), 'Ez')
                grid.add_probe('feed', cx+1, cy, cz) # Probe ao lado
                
                self.signals.log.emit(f"Grid Wire: {Nx}x{Ny}x{Nz}")

            # --- EXECUÇÃO COMUM ---
            vis_func = None
            vis_interval = 40 
            if self.enable_anim:
                vis_func = self.signals.field_update.emit

            n_steps = 1200
            grid.run(n_steps, 
                     progress_callback=self.signals.progress.emit,
                     vis_callback=vis_func,
                     vis_interval=vis_interval)

            # Pós-processamento
            feed_data = np.array(grid.probes['feed']['data'])
            freqs_ghz, s11_db = calculate_s11(feed_data, feed_data, grid.dt, fc)

            results = {'freqs_ghz': freqs_ghz, 's11_db': s11_db}
            self.signals.finished.emit(results)

        except Exception as e:
            self.signals.error.emit(str(e))
            import traceback
            traceback.print_exc()

# --- GUI PRINCIPAL ---
class AntennaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador FDTD Multiantena")
        self.setGeometry(100, 100, 1200, 800)
        self.threadpool = QThreadPool()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)

        # --- Painel Controle ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        
        # SELETOR DE TIPO DE ANTENA
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Microstrip Patch", "Dipole (Half-Wave)", "Yagi-Uda"])
        self.combo_type.currentIndexChanged.connect(self.on_type_changed)
        
        form = QFormLayout()
        form.addRow("<b>Tipo de Antena:</b>", self.combo_type)
        
        self.input_freq = QLineEdit("2.4")
        self.input_er = QLineEdit("4.4")
        self.input_h = QLineEdit("1.6")
        
        form.addRow("Freq (GHz):", self.input_freq)
        self.lbl_er = QLabel("Substrato Er:")
        self.lbl_h = QLabel("Altura h (mm):")
        
        form.addRow(self.lbl_er, self.input_er)
        form.addRow(self.lbl_h, self.input_h)
        
        self.check_anim = QCheckBox("Animação em Tempo Real")
        self.check_anim.setChecked(True)
        
        self.btn_run = QPushButton("RODAR SIMULAÇÃO")
        self.btn_run.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; height: 45px;")
        self.btn_run.clicked.connect(self.start_sim)
        
        self.progress = QProgressBar()
        self.lbl_log = QLabel("Selecione uma antena e clique em Rodar.")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setStyleSheet("font-size: 11px; color: #333; border: 1px solid #ccc; padding: 5px;")

        controls_layout.addLayout(form)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.check_anim)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.btn_run)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.progress)
        controls_layout.addWidget(self.lbl_log)
        controls_layout.addStretch()
        
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

        # --- Painel Visualização ---
        self.tabs = QTabWidget()
        self.canvas_anim = MplCanvas(self)
        self.tabs.addTab(self.canvas_anim, "Campo Elétrico (Visualização)")
        self.canvas_s11 = MplCanvas(self)
        self.tabs.addTab(self.canvas_s11, "Resposta S11 (Frequência)")

        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def on_type_changed(self):
        # Desabilita inputs que não fazem sentido para Dipolo/Yagi
        tipo = self.combo_type.currentText()
        is_patch = "Patch" in tipo
        self.input_er.setEnabled(is_patch)
        self.input_h.setEnabled(is_patch)
        if not is_patch:
            self.lbl_log.setText("Nota: Dipolos e Yagis são simulados no ar livre. Er e h serão ignorados.")

    def start_sim(self):
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.tabs.setCurrentIndex(0)
        
        self.canvas_anim.axes.cla()
        self.canvas_anim.axes.set_title("Inicializando...")
        self.canvas_anim.draw()

        try:
            ant_type = self.combo_type.currentText()
            f = float(self.input_freq.text())
            
            # Valores padrão caso estejam desabilitados
            er = float(self.input_er.text()) if self.input_er.text() else 1.0
            h = float(self.input_h.text()) if self.input_h.text() else 1.0
            
            # Calcula dimensões se for Patch
            W, L = 0, 0
            if "Patch" in ant_type:
                model = PatchAntennaModel(f, er, h)
                W, L, _ = model.calculate_dimensions()
            
            params = {'freq_ghz': f, 'er': er, 'h_mm': h}
            dims = {'W_m': W, 'L_m': L, 'h_m': h*1e-3}
            
            worker = FDTDWorker(params, dims, ant_type, self.check_anim.isChecked())
            worker.signals.progress.connect(self.progress.setValue)
            worker.signals.log.connect(self.lbl_log.setText)
            worker.signals.field_update.connect(self.update_anim)
            worker.signals.finished.connect(self.sim_done)
            worker.signals.error.connect(self.sim_error)
            
            self.threadpool.start(worker)
            
        except Exception as e:
            self.sim_error(str(e))

    def update_anim(self, field_data):
        max_val = np.max(np.abs(field_data))
        if max_val == 0: return
        try:
            self.canvas_anim.axes.cla()
            self.canvas_anim.axes.imshow(field_data, cmap='RdBu', aspect='auto', origin='lower') # vmin=-max_val/2, vmax=max_val/2
            self.canvas_anim.axes.set_title(f"Campo Elétrico Ez (Max: {max_val:.1e})")
            self.canvas_anim.draw()
            QApplication.processEvents() 
        except: pass

    def sim_done(self, res):
        self.btn_run.setEnabled(True)
        self.lbl_log.setText("Finalizado!")
        
        self.canvas_s11.axes.cla()
        f = res['freqs_ghz']
        s11 = res['s11_db']
        target_f = float(self.input_freq.text())
        
        mask = (f > target_f*0.5) & (f < target_f*1.5)
        self.canvas_s11.axes.plot(f[mask], s11[mask], 'b-', linewidth=2)
        self.canvas_s11.axes.axvline(x=target_f, color='r', linestyle='--', alpha=0.5, label='Alvo')
        self.canvas_s11.axes.set_title("Perda de Retorno (S11)")
        self.canvas_s11.axes.set_xlabel("Frequência (GHz)")
        self.canvas_s11.axes.set_ylabel("Magnitude (dB)")
        self.canvas_s11.axes.legend()
        self.canvas_s11.axes.grid(True, which='both', alpha=0.3)
        self.canvas_s11.draw()
        self.tabs.setCurrentIndex(1)

    def sim_error(self, msg):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Erro", msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = AntennaApp()
    win.show()
    sys.exit(app.exec_())