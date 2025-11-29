"""
Simulação 2D - Vista Lateral (Seção Transversal)
Gera uma imagem PNG com a distribuição de temperatura em corte lateral
e estima o coeficiente convectivo `h` a partir de correlações simples de convecção forçada.

Método: resolveu o campo estacionário 
-∇·(k ∇T) = q usando iteração Gauss-Seidel similar ao `sim_2d_conduction.py`.

Retorna bytes PNG prontos para exibir no Streamlit.
"""
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Dict, Tuple
import math

# Propriedades do ar típicas a ~25°C
K_AIR = 0.026  # W/mK
NU_AIR = 15.89e-6  # m2/s (cinemática)
PR_AIR = 0.71


def estimate_h_from_velocity(U: float, L_char: float) -> Tuple[float, Dict]:
    """Estimativa simples do coeficiente convectivo h usando correlações de placa plana.

    Args:
        U: velocidade do ar (m/s)
        L_char: comprimento característico (m) — por exemplo altura da aleta

    Returns:
        h (W/m2K), details dict
    """
    if U <= 0:
        # fluxo natural (ordem de grandeza)
        h = 5.0
        details = {'method': 'natural (fallback)', 'Re': 0.0, 'Nu': None}
        return h, details

    Re = U * L_char / NU_AIR
    Pr = PR_AIR

    # Escolher correlação: placa plana (envoltório): laminar até Re~5e5
    if Re < 5e5:
        Nu = 0.664 * (Re ** 0.5) * (Pr ** (1.0/3.0))
        method = 'laminar flat plate (local->avg approx)'
    else:
        Nu = 0.037 * (Re ** 0.8) * (Pr ** (1.0/3.0))
        method = 'turbulent flat plate (empirical)'

    # Evitar Nu muito baixo
    Nu = max(Nu, 0.1)
    h = Nu * K_AIR / L_char

    details = {'method': method, 'Re': Re, 'Nu': Nu}
    return h, details


def _create_vertical_geometry(nx: int, ny: int, Lx: float, Ly: float, params: Dict):
    """Cria máscaras para uma vista lateral: camadas empilhadas verticalmente (y).

    Domínio: x horizontal (lateral), y vertical (altura). Die e passivas no centro em x;
    região de ar à direita/embaixo/à esquerda conforme `air_margin`.
    """
    dx = Lx / nx
    dy = Ly / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    x, y = np.meshgrid(xc, yc, indexing='xy')

    # Parâmetros geométricos
    die_w = params.get('die_width', 0.012)
    die_th = params.get('die_thickness', 0.0005)
    tim_th = params.get('tim_thickness', 100e-6)
    spreader_th = params.get('spreader_thickness', 2e-3)
    base_th = params.get('base_thickness', 3e-3)
    fin_h = params.get('fin_height', 25e-3)
    fin_th = params.get('fin_thickness', 1e-3)
    n_fins = params.get('n_fins', 10)
    base_width = params.get('base_width', 0.04)

    # Centralizar em x
    cx = Lx / 2.0
    # construir camadas verticalmente a partir do die (y increasing upward from bottom 0)
    # vamos posicionar o die perto do topo do conjunto de camadas (para visual)

    # coordenada y do topo do die (a partir do fundo y=0)
    y_die_top = base_th + spreader_th + tim_th + die_th
    # definir máscaras por faixa y
    y_bottom = 0.0
    die_mask = (np.abs(x - cx) <= die_w / 2.0) & (y >= (y_die_top - die_th)) & (y <= y_die_top)

    tim_mask = (np.abs(x - cx) <= base_width / 2.0) & (y >= (y_die_top - die_th - tim_th)) & (y < (y_die_top - die_th))

    spreader_mask = (np.abs(x - cx) <= base_width / 2.0) & (y >= (y_die_top - die_th - tim_th - spreader_th)) & (y < (y_die_top - die_th - tim_th))

    base_mask = (np.abs(x - cx) <= base_width / 2.0) & (y >= 0) & (y < base_th)

    # Aletas: partir do topo da base (y = base_th) e crescer para cima
    fins_mask = np.zeros_like(die_mask, dtype=bool)
    # distribuir aletas ao longo da largura do base
    spacing = base_width / max(n_fins, 1)
    start_x = cx - base_width / 2.0
    for i in range(n_fins):
        fx0 = start_x + i * spacing + (spacing - fin_th) / 2.0
        fx1 = fx0 + fin_th
        fy0 = base_th
        fy1 = base_th + fin_h
        fins_mask |= (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)

    # Air mask: tudo que não é sólido é ar
    solid_mask = die_mask | tim_mask | spreader_mask | base_mask | fins_mask
    air_mask = ~solid_mask

    cell_area = dx * dy
    return x, y, die_mask, tim_mask, spreader_mask, base_mask, fins_mask, air_mask, cell_area


def run_side_view_simulation(power_w: float = 95.0,
                             T_amb: float = 25.0,
                             nx: int = 120,
                             ny: int = 160,
                             Lx: float = 0.04,
                             Ly: float = 0.06,
                             params: Dict = None,
                             airflow: Dict = None,
                             max_iter: int = 8000,
                             tol: float = 1e-5) -> Tuple[bytes, Dict]:
    """Roda a simulação de vista lateral e retorna imagem PNG e um resumo (dict).

    Args:
        params: dicionário com propriedades geométricas e materiais (k values etc.)
        airflow: dicionário com chaves: {'mode':'velocity'|'h_manual', 'U':..., 'h':...}

    Returns:
        (img_bytes, summary)
    """
    if params is None:
        params = {}
    if airflow is None:
        airflow = {'mode': 'velocity', 'U': 2.0}

    # Propriedades materiais
    k_air = params.get('k_air', K_AIR)
    k_al = params.get('k_al', 200.0)
    k_cu = params.get('k_cu', 390.0)
    k_die = params.get('k_die', 120.0)

    # Geometrias
    x, y, die_mask, tim_mask, spreader_mask, base_mask, fins_mask, air_mask, cell_area = _create_vertical_geometry(nx, ny, Lx, Ly, params)

    # campo k
    k_field = np.full_like(x, k_air, dtype=float)
    k_field[base_mask] = k_al
    k_field[fins_mask] = k_al
    k_field[spreader_mask] = k_cu
    k_field[tim_mask] = params.get('k_tim', 5.0)
    k_field[die_mask] = k_die

    # Fonte volumétrica: apenas células do die
    q = np.zeros_like(k_field, dtype=float)
    die_cells = die_mask.sum()
    if die_cells == 0:
        raise ValueError('Die mask vazia; verifique parâmetros de geometria.')
    # Volume do die aproximado: die_area (in-plane) * die_thickness (depth out of plane)
    die_area_inplane = die_mask.sum() * (Lx / nx) * (Ly / ny)
    die_volume = die_area_inplane * params.get('die_thickness', 0.5e-3)
    q_value = power_w / die_volume
    q[die_mask] = q_value

    # Determinar h a partir do airflow
    if airflow.get('mode', 'velocity') == 'h_manual':
        h = float(airflow.get('h', 10.0))
        h_details = {'method': 'manual', 'h': h}
    else:
        U = float(airflow.get('U', 2.0))
        # comprimento característico = altura da aleta ou do conjunto
        L_char = params.get('fin_height', 25e-3)
        h, h_det = estimate_h_from_velocity(U, L_char)
        h_details = {'method': 'correlation', 'U': U, **h_det, 'h': h}

    # Inicializar T
    T = np.full_like(k_field, T_amb, dtype=float)

    dx = Lx / nx
    dy = Ly / ny
    dx2 = dx * dx
    dy2 = dy * dy

    ny_i, nx_j = T.shape

    # Iteração Gauss-Seidel com condição convectiva nas células de ar que estão na borda externa
    for it in range(max_iter):
        T_old = T.copy()

        for i in range(ny_i):
            for j in range(nx_j):
                kc = k_field[i, j]

                # vizinhos
                # left
                if j - 1 >= 0:
                    kl = 0.5 * (kc + k_field[i, j - 1])
                    Tl = T[i, j - 1]
                else:
                    kl = kc
                    Tl = T_amb

                # right
                if j + 1 < nx_j:
                    kr = 0.5 * (kc + k_field[i, j + 1])
                    Tr = T[i, j + 1]
                else:
                    # borda externa direita -> convecção para o ar exterior
                    kr = kc
                    # aproximar vizinho fictício usando h (fluxo = h*(T - Tamb))
                    Tr = T_amb - (h * dx / kr) * (T[i, j] - T_amb)

                # down
                if i - 1 >= 0:
                    kd = 0.5 * (kc + k_field[i - 1, j])
                    Td = T[i - 1, j]
                else:
                    kd = kc
                    Td = T_amb

                # up
                if i + 1 < ny_i:
                    ku = 0.5 * (kc + k_field[i + 1, j])
                    Tu = T[i + 1, j]
                else:
                    ku = kc
                    Tu = T_amb - (h * dy / ku) * (T[i, j] - T_amb)

                A = (kl / dx2) + (kr / dx2) + (kd / dy2) + (ku / dy2)
                B = (kl * Tl + kr * Tr) / dx2 + (kd * Td + ku * Tu) / dy2 + q[i, j]

                T[i, j] = B / A

        diff = np.max(np.abs(T - T_old))
        if diff < tol:
            break

    # Preparar figura
    fig, ax = plt.subplots(figsize=(5, 8))
    pcm = ax.pcolormesh(x, y, T, shading='auto', cmap='inferno')
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('T (°C)')

    # Contornos para as regiões sólidas
    ax.contour(x, y, die_mask.astype(float), levels=[0.5], colors='cyan', linewidths=1.0)
    ax.contour(x, y, tim_mask.astype(float), levels=[0.5], colors='lime', linewidths=0.8)
    ax.contour(x, y, spreader_mask.astype(float), levels=[0.5], colors='yellow', linewidths=0.8)
    ax.contour(x, y, base_mask.astype(float), levels=[0.5], colors='white', linewidths=0.8)
    ax.contour(x, y, fins_mask.astype(float), levels=[0.5], colors='gray', linewidths=0.6)

    # Desenhar setas de fluxo de ar (simplificado) à direita do dissipador
    # setas horizontais com magnitude proporcional a U (se disponível)
    if airflow.get('mode', 'velocity') != 'h_manual':
        U = airflow.get('U', 2.0)
        # escolher um campo de velocidade para desenho
        # vetor uniforme para a área do ar
        sx = x[::max(1, ny_i//20), -1]  # pontos à direita
        sy = np.linspace(0.05*Ly, 0.95*Ly, len(sx))
        # construir grid de setas simples
        Xq, Yq = np.meshgrid(np.linspace(x.min(), x.max(), 6), np.linspace(0.05*Ly, 0.95*Ly, 12))
        uq = np.full_like(Xq, U)
        vq = np.zeros_like(Xq)
        ax.quiver(Xq, Yq, uq, vq, color='white', alpha=0.6, width=0.003)
        ax.text(0.02*Lx, 0.97*Ly, f'Velocidade do ar U={U:.2f} m/s', color='white')

    # Anotações de camadas
    ax.text(0.02*Lx, 0.90*Ly, 'Die', color='cyan')
    ax.text(0.02*Lx, 0.84*Ly, 'TIM', color='lime')
    ax.text(0.02*Lx, 0.78*Ly, 'Spreader', color='yellow')
    ax.text(0.02*Lx, 0.72*Ly, 'Base / Fins', color='white')

    ax.set_title('Vista Lateral - Mapa de Temperatura (°C)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # legenda para h
    ax.text(0.02*Lx, 0.03*Ly, f'Est. h = {h:.1f} W/m²K ({h_details["method"]})', color='white')

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    summary = {
        'h': h,
        'h_details': h_details,
        'T_max': float(T.max()),
        'T_min': float(T.min()),
        'T_center_die': float(T[die_mask].mean()) if die_mask.sum() > 0 else None
    }

    return buf.read(), summary


if __name__ == '__main__':
    # Exemplo de execução rápida
    params = {
        'die_width': 0.012,
        'die_thickness': 0.5e-3,
        'tim_thickness': 100e-6,
        'spreader_thickness': 2e-3,
        'base_thickness': 3e-3,
        'fin_height': 25e-3,
        'fin_thickness': 1e-3,
        'n_fins': 15,
        'base_width': 0.04,
        'k_tim': 5.0
    }
    img, summary = run_side_view_simulation(power_w=95.0, T_amb=25.0, params=params, airflow={'mode': 'velocity', 'U': 2.0})
    with open('sim_side_view.png', 'wb') as f:
        f.write(img)
    print('Simulação side-view salva em sim_side_view.png')
    print(summary)
