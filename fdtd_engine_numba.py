import numpy as np
import time
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman

# Tenta importar Numba. Se não der, cria um decorator "falso" que não faz nada.
try:
    from numba import njit, prange
    print(">> Numba detectado. Aceleração ativada para FDTD.")
    USE_NUMBA = True
except ImportError:
    print(">> Numba não encontrado. FDTD rodará em modo lento (apenas para grids muito pequenos).")
    def njit(*args, **kwargs):
        return lambda f: f
    prange = range # fallback for parallel
    USE_NUMBA = False

eps0 = 8.854187817e-12
mu0 = 4*np.pi*1e-7
c0 = 1/np.sqrt(eps0*mu0)

# --- KERNEL NÚMERICO (ESTÁTICO E RÁPIDO COM NUMBA) ---
# Adicionado prange para loops paralelos se Numba estiver disponível
@njit(fastmath=True, parallel=True)
def _update_h_fields(Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, dt_mu0_inv):
    Nx, Ny, Nz = Hx.shape

    # Hx: loop em y e z
    for i in prange(Nx):
        for j in prange(Ny - 1):
            for k in prange(Nz - 1):
                Hx[i, j, k] -= dt_mu0_inv * (
                    (Ez[i, j + 1, k] - Ez[i, j, k]) / dy -
                    (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                )
    
    # Hy: loop em x e z
    for i in prange(Nx - 1):
        for j in prange(Ny):
            for k in prange(Nz - 1):
                Hy[i, j, k] -= dt_mu0_inv * (
                    (Ex[i, j, k + 1] - Ex[i, j, k]) / dz -
                    (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
                )
    
    # Hz: loop em x e y
    for i in prange(Nx - 1):
        for j in prange(Ny - 1):
            for k in prange(Nz):
                Hz[i, j, k] -= dt_mu0_inv * (
                    (Ey[i + 1, j, k] - Ey[i, j, k]) / dx -
                    (Ex[i, j + 1, k] - Ex[i, j, k]) / dy
                )

@njit(fastmath=True, parallel=True)
def _update_e_fields(Ex, Ey, Ez, Hx, Hy, Hz, inv_eps_dt, dx, dy, dz):
    Nx, Ny, Nz = Ex.shape

    # Ex
    for i in prange(1, Nx - 1):
        for j in prange(1, Ny - 1):
            for k in prange(1, Nz - 1):
                Ex[i, j, k] += inv_eps_dt[i, j, k] * (
                    (Hz[i, j, k] - Hz[i, j - 1, k]) / dy -
                    (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
                )

    # Ey
    for i in prange(1, Nx - 1):
        for j in prange(1, Ny - 1):
            for k in prange(1, Nz - 1):
                Ey[i, j, k] += inv_eps_dt[i, j, k] * (
                    (Hx[i, j, k] - Hx[i, j, k - 1]) / dz -
                    (Hz[i, j, k] - Hz[i - 1, j, k]) / dx
                )

    # Ez
    for i in prange(1, Nx - 1):
        for j in prange(1, Ny - 1):
            for k in prange(1, Nz - 1):
                Ez[i, j, k] += inv_eps_dt[i, j, k] * (
                    (Hy[i, j, k] - Hy[i - 1, j, k]) / dx -
                    (Hx[i, j, k] - Hx[i, j - 1, k]) / dy
                )

@njit(fastmath=True, parallel=False) 
def _apply_pec_kernel(Ex, Ey, Ez, pec_mask):
    # Numba as vezes falha com indexação booleana avançada em 3D (Ex[mask]=0).
    # Substituindo por loops explícitos, que o Numba otimiza muito bem.
    Nx, Ny, Nz = Ex.shape
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if pec_mask[i, j, k]:
                    Ex[i, j, k] = 0.0
                    Ey[i, j, k] = 0.0
                    Ez[i, j, k] = 0.0
@njit(fastmath=True, parallel=False) # CPML simples também não
def _apply_simple_cpml_damping(Ex, Ey, Ez, cpml_thickness, alpha):
    Nx, Ny, Nz = Ex.shape
    m = cpml_thickness

    if m > 0:
        # X-boundaries
        Ex[:m, :, :] *= alpha; Ex[-m:, :, :] *= alpha
        Ey[:m, :, :] *= alpha; Ey[-m:, :, :] *= alpha
        Ez[:m, :, :] *= alpha; Ez[-m:, :, :] *= alpha
        
        # Y-boundaries
        Ex[:, :m, :] *= alpha; Ex[:, -m:, :] *= alpha
        Ey[:, :m, :] *= alpha; Ey[:, -m:, :] *= alpha
        Ez[:, :m, :] *= alpha; Ez[:, -m:, :] *= alpha
        
        # Z-boundaries
        Ex[:, :, :m] *= alpha; Ex[:, :, -m:] *= alpha
        Ey[:, :, :m] *= alpha; Ey[:, :, -m:] *= alpha
        Ez[:, :, :m] *= alpha; Ez[:, :, -m:] *= alpha

# --- FONTES ---
def gaussian_pulse(t, t0, spread):
    return np.exp(-((t - t0) / spread)**2)

def modulated_gaussian_pulse(t, t0, spread, fc):
    """Gaussian pulse modulated by a sine wave for broadband source."""
    return gaussian_pulse(t, t0, spread) * np.sin(2 * np.pi * fc * t)

# --- CLASSE DO GRID FDTD ---
class FDTDGrid:
    def __init__(self, Nx, Ny, Nz, dx, dy, dz, dt=None, cpml_thickness=10):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        if dt is None:
            # Courant stability condition (conservative)
            self.dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) # Fator 0.9 para mais segurança
        else:
            self.dt = dt

        # Fields on Yee grid (staggered)
        self.Ex = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ey = np.zeros_like(self.Ex)
        self.Ez = np.zeros_like(self.Ex)
        self.Hx = np.zeros_like(self.Ex)
        self.Hy = np.zeros_like(self.Ex)
        self.Hz = np.zeros_like(self.Ex)

        # Material arrays (scalar relative permittivity)
        self.eps_r = np.ones((Nx, Ny, Nz), dtype=np.float64)
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r) # Pré-cálculo para velocidade

        # PEC mask (True where metal)
        self.pec = np.zeros((Nx, Ny, Nz), dtype=bool)

        # CPML buffers (damping factor for simple CPML)
        self.cpml_thickness = cpml_thickness
        self.cpml_alpha = 0.98 # Damping factor for simple layer

        # Probes and sources
        self.probes = {}
        self.sources = []

    def update_material(self, x0, x1, y0, y1, z0, z1, eps_r_val):
        """Define uma região com constante dielétrica."""
        # Note: A grade FDTD é 0-indexed, então (x0:x1) significa células de x0 até x1-1
        self.eps_r[x0:x1, y0:y1, z0:z1] = eps_r_val
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r) # Recalcular

    def add_pec_patch(self, x0, x1, y0, y1, z0, z1):
        """Define uma região PEC (metal)."""
        self.pec[x0:x1, y0:y1, z0:z1] = True
        # Forçar epsilon_r muito alto dentro do PEC para evitar problemas numéricos
        # com inv_eps_dt, embora o campo E seja zero.
        self.eps_r[x0:x1, y0:y1, z0:z1] = 1e9 # Efeito de condutor perfeito
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r)

    def add_probe(self, name, x, y, z):
        """Adiciona uma probe para registrar o Ez em um ponto."""
        self.probes[name] = {'pos':(x,y,z), 'data':[]}

    def add_source(self, x, y, z, source_func_args, field_component='Ez'):
        """
        Adiciona uma fonte de campo em um ponto.
        source_func_args: tupla (function, t0, spread, fc)
        """
        self.sources.append({
            'pos': (x,y,z), 
            'func_args': source_func_args, 
            'component': field_component
        })

    def step(self, n):
        # Update H fields
        _update_h_fields(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, 
                         self.dx, self.dy, self.dz, self.dt / mu0)

        # Update E fields
        _update_e_fields(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, 
                         self.inv_eps_dt, self.dx, self.dy, self.dz)

        # Apply PEC
        _apply_pec_kernel(self.Ex, self.Ey, self.Ez, self.pec)

        # Apply simple absorbing layer (CPML-lite)
        _apply_simple_cpml_damping(self.Ex, self.Ey, self.Ez, 
                                   self.cpml_thickness, self.cpml_alpha)

        # Apply sources
        for src in self.sources:
            x,y,z = src['pos']
            func, t0, spread, fc = src['func_args']
            val = func(n*self.dt, t0, spread, fc)
            if src['component'] == 'Ez':
                self.Ez[x,y,z] += val
            elif src['component'] == 'Ex':
                self.Ex[x,y,z] += val
            elif src['component'] == 'Ey':
                self.Ey[x,y,z] += val

        # Record probes
        for name, p in self.probes.items():
            x,y,z = p['pos']
            # Para simplificar, vamos registrar apenas Ez nas probes por enquanto
            p['data'].append(self.Ez[x,y,z])
            
    def run(self, n_steps, progress_callback=None):
        t0_start = time.time()
        for n in range(n_steps):
            self.step(n)
            if progress_callback and (n % 50 == 0 or n == n_steps - 1): # Report progress periodically
                progress_callback(int((n + 1) / n_steps * 100))
        t1_end = time.time()
        print(f"FDTD run complete: {n_steps} steps in {t1_end-t0_start:.2f} s")

# --- ANÁLISE PÓS-PROCESSAMENTO (S11) ---
def calculate_s11(probe_incident_data, probe_reflected_data, dt, fc, bandwidth_factor=2):
    """
    Calcula o S11 a partir de dados de campo elétrico incidente e refletido.
    Assume que a source é um pulso gaussiano modulado.
    """
    if len(probe_incident_data) == 0 or len(probe_reflected_data) == 0:
        return np.array([]), np.array([])

    N = len(probe_incident_data)
    
    # Aplicar janela Blackman para reduzir vazamento espectral
    window = blackman(N)
    E_inc = np.array(probe_incident_data) * window
    E_ref = np.array(probe_reflected_data) * window

    # Transformada de Fourier
    freqs = fftfreq(N, d=dt)
    E_inc_fft = fft(E_inc)
    E_ref_fft = fft(E_ref)

    # Filtrar para frequências positivas e dentro da banda relevante
    positive_freqs_idx = np.where(freqs >= 0)
    freqs = freqs[positive_freqs_idx]
    E_inc_fft = E_inc_fft[positive_freqs_idx]
    E_ref_fft = E_ref_fft[positive_freqs_idx]

    # Calcular S11: E_ref / E_inc
    # Evitar divisão por zero e ruído fora da banda do pulso
    s11_mag = np.zeros_like(freqs, dtype=np.float64)
    
    # Definir uma máscara para a banda de interesse (em torno de fc)
    min_freq = fc - fc * bandwidth_factor / 2
    max_freq = fc + fc * bandwidth_factor / 2
    
    # Ajustar para não ter frequências negativas
    min_freq = max(0, min_freq) 
    
    valid_indices = np.where((np.abs(E_inc_fft) > 1e-10) & (freqs >= min_freq) & (freqs <= max_freq))
    
    if len(valid_indices[0]) > 0:
        s11_mag[valid_indices] = np.abs(E_ref_fft[valid_indices] / E_inc_fft[valid_indices])
    
    # Converter para dB
    s11_db = 20 * np.log10(s11_mag + 1e-12) # Adiciona pequeno valor para evitar log(0)
    
    return freqs / 1e9, s11_db # Retorna frequências em GHz