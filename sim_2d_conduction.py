"""
Simulação 2D de Condução Térmica (diferenças finitas explícitas/iterativas)

Arquivo: sim_2d_conduction.py
Descrição: protótipo simples que resolve o problema estacionário
            -∇·(k ∇T) = q em um domínio retangular com subdomínios
            Representa: die (fonte volumétrica), base e algumas aletas

Retorna imagem PNG em bytes que pode ser exibida no Streamlit.
"""
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Tuple, Dict


def _create_geometry(nx: int, ny: int, Lx: float, Ly: float, params: Dict):
    """Cria máscaras e campos geométricos.

    Retorna: (x, y, die_mask, base_mask, fins_mask, cell_area)
    """
    dx = Lx / nx
    dy = Ly / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    x, y = np.meshgrid(xc, yc, indexing='xy')

    # Die no centro
    die_w = params.get('die_width', 0.012)
    die_h = params.get('die_height', 0.012)
    cx = Lx / 2.0
    cy = Ly / 2.0
    die_mask = (np.abs(x - cx) <= die_w / 2) & (np.abs(y - cy) <= die_h / 2)

    # Base: região inferior com altura base_h
    base_h = params.get('base_height', 0.01)
    base_mask = y <= base_h

    # Aletas simples: desenhar n_fins retângulos na parte superior da base
    n_fins = params.get('n_fins', 8)
    fin_th = params.get('fin_thickness', 1e-3)
    fin_w = params.get('fin_width', 0.04)
    fin_h = params.get('fin_height', 0.03)
    fins_mask = np.zeros_like(die_mask, dtype=bool)
    # distribuir as aletas ao longo do eixo x centralizado
    spacing = params.get('base_width', 0.04) / max(n_fins, 1)
    start_x = (Lx - params.get('base_width', 0.04)) / 2.0
    for i in range(n_fins):
        fx0 = start_x + i * spacing + (spacing - fin_th) / 2.0
        fx1 = fx0 + fin_th
        fy0 = base_h
        fy1 = base_h + fin_h
        fins_mask |= (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)

    cell_area = dx * dy
    return x, y, die_mask, base_mask, fins_mask, cell_area


def run_2d_simulation(power_w: float = 95.0,
                      T_amb: float = 25.0,
                      nx: int = 120,
                      ny: int = 120,
                      Lx: float = 0.06,
                      Ly: float = 0.06,
                      params: Dict = None,
                      max_iter: int = 5000,
                      tol: float = 1e-5) -> bytes:
    """Roda a simulação 2D e retorna imagem PNG em bytes.

    Método: solução iterativa Gauss-Seidel para o problema estacionário
            ∇·(k ∇T) + q = 0 com condição de contorno convectiva aproximada
            (bordas expostas com fluxo = h*(T - T_amb) aplicado como fluxo).
    """
    if params is None:
        params = {}

    # Propriedades
    k_air = params.get('k_air', 0.026)
    k_al = params.get('k_al', 200.0)
    k_die = params.get('k_die', 120.0)
    die_thickness = params.get('die_thickness', 0.5e-3)

    h = params.get('h', 80.0)

    x, y, die_mask, base_mask, fins_mask, cell_area = _create_geometry(nx, ny, Lx, Ly, params)

    # campo k por célula
    k = np.full((ny, nx), k_air)
    k[base_mask] = k_al
    k[fins_mask] = k_al
    k[die_mask] = k_die

    # Fonte volumétrica q (W/m3): apenas nas células do die
    q = np.zeros_like(k)
    die_cells = np.count_nonzero(die_mask)
    if die_cells == 0:
        raise ValueError('Die mask não contém células; verifique geometria e malha.')
    die_area = die_mask.sum() * cell_area
    die_volume = die_area * die_thickness
    q_value = power_w / die_volume
    q[die_mask] = q_value

    # Inicialização da temperatura
    T = np.full_like(k, T_amb, dtype=float)

    dx = Lx / nx
    dy = Ly / ny
    dx2 = dx * dx
    dy2 = dy * dy

    # Precomputar índices para vizinhos
    ny_i, nx_j = T.shape

    # Iteração Gauss-Seidel
    for it in range(max_iter):
        T_old = T.copy()

        # atualizar cada célula interior
        for i in range(ny_i):
            for j in range(nx_j):
                # coeficientes com média harmônica/geométrica simples para k entre células
                kc = k[i, j]
                # vizinhos e k médio
                # left
                if j - 1 >= 0:
                    kl = 0.5 * (kc + k[i, j - 1])
                    Tl = T[i, j - 1]
                else:
                    # fronteira esquerda: aplicar condição convectiva como vizinho fictício
                    kl = kc
                    Tl = T_amb - (h * dx / kl) * (T[i, j] - T_amb)

                # right
                if j + 1 < nx_j:
                    kr = 0.5 * (kc + k[i, j + 1])
                    Tr = T[i, j + 1]
                else:
                    kr = kc
                    Tr = T_amb - (h * dx / kr) * (T[i, j] - T_amb)

                # down
                if i - 1 >= 0:
                    kd = 0.5 * (kc + k[i - 1, j])
                    Td = T[i - 1, j]
                else:
                    kd = kc
                    Td = T_amb - (h * dy / kd) * (T[i, j] - T_amb)

                # up
                if i + 1 < ny_i:
                    ku = 0.5 * (kc + k[i + 1, j])
                    Tu = T[i + 1, j]
                else:
                    ku = kc
                    Tu = T_amb - (h * dy / ku) * (T[i, j] - T_amb)

                A = (kl / dx2) + (kr / dx2) + (kd / dy2) + (ku / dy2)
                B = (kl * Tl + kr * Tr) / dx2 + (kd * Td + ku * Tu) / dy2 + q[i, j]

                # atualizar T[i,j]
                T[i, j] = B / A

        # critério de convergência
        diff = np.max(np.abs(T - T_old))
        if diff < tol:
            # print(f'Convergiu em {it} iterações; diff={diff:.3e}')
            break

    # Criar figura
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(x, y, T, shading='auto', cmap='inferno')
    ax.set_aspect('equal')
    ax.set_title('Mapa de Temperatura (°C) - Simulação 2D')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('T (°C)')

    # destacar die e aletas
    ax.contour(x, y, die_mask.astype(float), levels=[0.5], colors='cyan', linewidths=0.8)
    ax.contour(x, y, fins_mask.astype(float), levels=[0.5], colors='white', linewidths=0.6)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


if __name__ == '__main__':
    # Exemplo rápido
    img = run_2d_simulation()
    with open('sim_2d_result.png', 'wb') as f:
        f.write(img)
    print('Simulação concluída. Resultado salvo em sim_2d_result.png')
