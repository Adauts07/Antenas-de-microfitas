import numpy as np
import time
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman

# Tenta importar Numba
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f
    prange = range
    USE_NUMBA = False

eps0 = 8.854187817e-12
mu0 = 4*np.pi*1e-7
c0 = 1/np.sqrt(eps0*mu0)

# --- KERNELS NUMÉRICOS (Numba) ---
@njit(fastmath=True, parallel=True)
def _update_h_fields(Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, dt_mu0_inv):
    Nx, Ny, Nz = Hx.shape
    # Hx
    for i in prange(Nx):
        for j in prange(Ny - 1):
            for k in prange(Nz - 1):
                Hx[i, j, k] -= dt_mu0_inv * (
                    (Ez[i, j + 1, k] - Ez[i, j, k]) / dy -
                    (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                )
    # Hy
    for i in prange(Nx - 1):
        for j in prange(Ny):
            for k in prange(Nz - 1):
                Hy[i, j, k] -= dt_mu0_inv * (
                    (Ex[i, j, k + 1] - Ex[i, j, k]) / dz -
                    (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
                )
    # Hz
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
    # Correção para evitar erro de índice booleano no Numba
    Nx, Ny, Nz = Ex.shape
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if pec_mask[i, j, k]:
                    Ex[i, j, k] = 0.0
                    Ey[i, j, k] = 0.0
                    Ez[i, j, k] = 0.0

@njit(fastmath=True, parallel=False)
def _apply_simple_cpml_damping(Ex, Ey, Ez, cpml_thickness, alpha):
    Nx, Ny, Nz = Ex.shape
    m = cpml_thickness
    if m > 0:
        Ex[:m, :, :] *= alpha; Ex[-m:, :, :] *= alpha
        Ey[:m, :, :] *= alpha; Ey[-m:, :, :] *= alpha
        Ez[:m, :, :] *= alpha; Ez[-m:, :, :] *= alpha
        
        Ex[:, :m, :] *= alpha; Ex[:, -m:, :] *= alpha
        Ey[:, :m, :] *= alpha; Ey[:, -m:, :] *= alpha
        Ez[:, :m, :] *= alpha; Ez[:, -m:, :] *= alpha
        
        Ex[:, :, :m] *= alpha; Ex[:, :, -m:] *= alpha
        Ey[:, :, :m] *= alpha; Ey[:, :, -m:] *= alpha
        Ez[:, :, :m] *= alpha; Ez[:, :, -m:] *= alpha

# --- FONTES ---
def modulated_gaussian_pulse(t, t0, spread, fc):
    return np.exp(-((t - t0) / spread)**2) * np.sin(2 * np.pi * fc * t)

# --- CLASSE GRID ---
class FDTDGrid:
    def __init__(self, Nx, Ny, Nz, dx, dy, dz, dt=None, cpml_thickness=10):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        if dt is None:
            self.dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
        else:
            self.dt = dt

        self.Ex = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ey = np.zeros_like(self.Ex)
        self.Ez = np.zeros_like(self.Ex)
        self.Hx = np.zeros_like(self.Ex)
        self.Hy = np.zeros_like(self.Ex)
        self.Hz = np.zeros_like(self.Ex)
        self.eps_r = np.ones((Nx, Ny, Nz), dtype=np.float64)
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r)
        self.pec = np.zeros((Nx, Ny, Nz), dtype=bool)
        self.cpml_thickness = cpml_thickness
        self.cpml_alpha = 0.98 
        self.probes = {}
        self.sources = []

    def update_material(self, x0, x1, y0, y1, z0, z1, eps_r_val):
        self.eps_r[x0:x1, y0:y1, z0:z1] = eps_r_val
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r)

    def add_pec_patch(self, x0, x1, y0, y1, z0, z1):
        self.pec[x0:x1, y0:y1, z0:z1] = True
        self.eps_r[x0:x1, y0:y1, z0:z1] = 1e9
        self.inv_eps_dt = self.dt / (eps0 * self.eps_r)

    def add_probe(self, name, x, y, z):
        self.probes[name] = {'pos':(x,y,z), 'data':[]}

    def add_source(self, x, y, z, source_func_args, field_component='Ez'):
        self.sources.append({
            'pos': (x,y,z), 
            'func_args': source_func_args, 
            'component': field_component
        })

    def step(self, n):
        _update_h_fields(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, 
                         self.dx, self.dy, self.dz, self.dt / mu0)
        _update_e_fields(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, 
                         self.inv_eps_dt, self.dx, self.dy, self.dz)
        _apply_pec_kernel(self.Ex, self.Ey, self.Ez, self.pec)
        _apply_simple_cpml_damping(self.Ex, self.Ey, self.Ez, 
                                   self.cpml_thickness, self.cpml_alpha)

        for src in self.sources:
            x,y,z = src['pos']
            func, t0, spread, fc = src['func_args']
            val = func(n*self.dt, t0, spread, fc)
            if src['component'] == 'Ez': self.Ez[x,y,z] += val
            elif src['component'] == 'Ex': self.Ex[x,y,z] += val
            elif src['component'] == 'Ey': self.Ey[x,y,z] += val

        for name, p in self.probes.items():
            x,y,z = p['pos']
            p['data'].append(self.Ez[x,y,z])
            
    # ESTA É A FUNÇÃO CRÍTICA PARA A ANIMAÇÃO
    def run(self, n_steps, progress_callback=None, vis_callback=None, vis_interval=10):
        t0_start = time.time()
        for n in range(n_steps):
            self.step(n)
            
            # Barra de progresso
            if progress_callback and (n % 50 == 0 or n == n_steps - 1):
                progress_callback(int((n + 1) / n_steps * 100))
            
            # Envio de Imagem para Animação
            if vis_callback and (n % vis_interval == 0):
                # Pega o corte no meio do eixo Z
                field_slice = self.Ez[:, :, self.Nz // 2].T
                vis_callback(field_slice)

        t1_end = time.time()
        print(f"FDTD run complete: {n_steps} steps in {t1_end-t0_start:.2f} s")

def calculate_s11(probe_incident_data, probe_reflected_data, dt, fc, bandwidth_factor=2):
    if len(probe_incident_data) == 0: return np.array([]), np.array([])
    N = len(probe_incident_data)
    window = blackman(N)
    E_inc = np.array(probe_incident_data) * window
    E_ref = np.array(probe_reflected_data) * window
    freqs = fftfreq(N, d=dt)
    E_inc_fft = fft(E_inc)
    E_ref_fft = fft(E_ref)
    positive_freqs_idx = np.where(freqs >= 0)
    freqs = freqs[positive_freqs_idx]
    E_inc_fft = E_inc_fft[positive_freqs_idx]
    E_ref_fft = E_ref_fft[positive_freqs_idx]
    s11_mag = np.zeros_like(freqs, dtype=np.float64)
    min_freq = max(0, fc - fc * bandwidth_factor / 2)
    max_freq = fc + fc * bandwidth_factor / 2
    valid_indices = np.where((np.abs(E_inc_fft) > 1e-10) & (freqs >= min_freq) & (freqs <= max_freq))
    if len(valid_indices[0]) > 0:
        s11_mag[valid_indices] = np.abs(E_ref_fft[valid_indices] / E_inc_fft[valid_indices])
    s11_db = 20 * np.log10(s11_mag + 1e-12)
    return freqs / 1e9, s11_db