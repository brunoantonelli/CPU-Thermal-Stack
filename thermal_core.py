"""
CPU Thermal Stack Designer - Núcleo de Cálculos
Projeto de Fenômenos de Transporte - Condução 1D Estacionária
Autores: Monique Rosa de Moraes, Bruno Antonelli de Oliveira
"""

import math
import numpy as np

def r_cond(thickness_m, k_w_mk, area_m2):
    """
    Calcula resistência térmica de condução
    R = L / (k * A)
    
    Args:
        thickness_m: espessura em metros
        k_w_mk: condutividade térmica (W/m·K)
        area_m2: área em m²
    
    Returns:
        Resistência térmica (K/W)
    """
    return thickness_m / (k_w_mk * area_m2)

def r_conv(h_w_m2k, area_m2):
    """
    Calcula resistência térmica de convecção
    R = 1 / (h * A)
    
    Args:
        h_w_m2k: coeficiente de convecção (W/m²·K)
        area_m2: área em m²
    
    Returns:
        Resistência térmica (K/W)
    """
    return 1.0 / (h_w_m2k * area_m2)

def fin_efficiency_rectangular(h, k_fin, thickness, width, height):
    """
    Calcula eficiência de aleta retangular (ponta adiabática)
    
    Args:
        h: coeficiente de convecção (W/m²·K)
        k_fin: condutividade da aleta (W/m·K)
        thickness: espessura da aleta (m)
        width: largura da aleta (m)
        height: altura da aleta (m)
    
    Returns:
        tuple: (eficiência, área_seção, perímetro)
    """
    # Área da seção transversal e perímetro
    A_c = thickness * width
    P = 2 * (thickness + width)
    
    # Parâmetro m da aleta
    if A_c == 0 or k_fin == 0:
        return 1.0, A_c, P
    
    m = math.sqrt(h * P / (k_fin * A_c))
    
    # Eficiência (ponta adiabática)
    if m * height == 0:
        eta_f = 1.0
    else:
        eta_f = math.tanh(m * height) / (m * height)
    
    return eta_f, A_c, P

def heatsink_thermal_resistance(h, k_fin, fin_thickness, fin_width, fin_height, 
                               n_fins, base_length, base_width):
    """
    Calcula resistência térmica total do dissipador com aletas
    
    Args:
        h: coeficiente de convecção (W/m²·K)
        k_fin: condutividade das aletas (W/m·K)
        fin_thickness: espessura das aletas (m)
        fin_width: largura das aletas (m)
        fin_height: altura das aletas (m)
        n_fins: número de aletas
        base_length: comprimento da base (m)
        base_width: largura da base (m)
    
    Returns:
        tuple: (resistência_térmica, detalhes_dict)
    """
    # Área total da base
    A_base_total = base_length * base_width
    
    # Área ocupada pelas aletas na base
    A_fins_footprint = n_fins * fin_thickness * base_width
    
    # Área da base exposta (entre as aletas)
    A_base_exposed = max(A_base_total - A_fins_footprint, 0.0)
    
    # Eficiência das aletas
    eta_f, A_c, P = fin_efficiency_rectangular(h, k_fin, fin_thickness, fin_width, fin_height)
    
    # Área de convecção de uma aleta (2 faces + ponta)
    A_fin_single = 2 * fin_width * fin_height + fin_thickness * fin_width
    A_fins_total = n_fins * A_fin_single
    
    # Área efetiva total para convecção
    A_eff = A_base_exposed + eta_f * A_fins_total
    
    # Resistência térmica de convecção
    R_conv = 1.0 / (h * A_eff) if A_eff > 0 else float('inf')
    
    # Detalhes para análise
    details = {
        'A_base_total': A_base_total,
        'A_base_exposed': A_base_exposed,
        'A_fin_single': A_fin_single,
        'A_fins_total': A_fins_total,
        'eta_f': eta_f,
        'A_eff': A_eff,
        'n_fins': n_fins
    }
    
    return R_conv, details

def calculate_cpu_temperatures(power_w, T_ambient, die_area, die_thickness, die_k,
                              layers, heatsink_params):
    """
    Calcula temperaturas na pilha térmica da CPU
    
    Args:
        power_w: potência dissipada (W)
        T_ambient: temperatura ambiente (°C)
        die_area: área do die (m²)
        die_thickness: espessura do die (m)
        die_k: condutividade do die (W/m·K)
        layers: lista de dicts com camadas {'name', 'thickness', 'k', 'area'}
        heatsink_params: dict com parâmetros do dissipador
    
    Returns:
        dict com resultados completos
    """
    
    # Resistência térmica do dissipador (convecção + aletas)
    R_heatsink, hs_details = heatsink_thermal_resistance(**heatsink_params)
    
    # Resistências das camadas (de cima para baixo na pilha)
    R_total = R_heatsink
    R_breakdown = [('Dissipador + Convecção', R_heatsink)]
    
    # Somar resistências das camadas (TIM, spreader, base, etc.)
    for layer in reversed(layers):  # reversed porque vamos do dissipador para o die
        R_layer = r_cond(layer['thickness'], layer['k'], layer['area'])
        R_total += R_layer
        R_breakdown.append((layer['name'], R_layer))
    
    # Temperatura na superfície do die
    T_die_surface = T_ambient + power_w * R_total
    
    # Geração interna no die (modelo de placa com geração uniforme)
    q_dot = power_w / (die_area * die_thickness)  # W/m³
    delta_T_generation = q_dot * die_thickness**2 / (8.0 * die_k)
    
    # Temperatura de junção (centro do die)
    T_junction = T_die_surface + delta_T_generation
    
    # Resultados organizados
    results = {
        'T_ambient': T_ambient,
        'T_die_surface': T_die_surface,
        'T_junction': T_junction,
        'R_total': R_total,
        'R_breakdown': list(reversed(R_breakdown)),  # ordem do die para o ambiente
        'delta_T_generation': delta_T_generation,
        'heatsink_details': hs_details,
        'power': power_w,
        'q_dot': q_dot
    }
    
    return results

# Função de teste rápido
def test_basic_calculation():
    """Teste rápido para verificar se os cálculos estão funcionando"""
    
    print("🧪 Testando cálculos básicos...")
    
    # Parâmetros de teste - CPU típica de 95W
    power = 95  # W
    T_amb = 25  # °C
    
    # Die (chip)
    die_area = 12e-3 * 12e-3  # 12x12 mm
    die_thickness = 0.5e-3    # 0.5 mm
    die_k = 120               # W/m·K (silício)
    
    # Camadas da pilha térmica
    layers = [
        {
            'name': 'TIM (Pasta Térmica)', 
            'thickness': 100e-6,  # 100 μm
            'k': 5.0,             # W/m·K
            'area': 40e-3 * 40e-3 # 40x40 mm
        },
        {
            'name': 'Heat Spreader (Cobre)', 
            'thickness': 2e-3,    # 2 mm
            'k': 390,             # W/m·K
            'area': 40e-3 * 40e-3 # 40x40 mm
        },
        {
            'name': 'Base Dissipador (Alumínio)', 
            'thickness': 3e-3,    # 3 mm
            'k': 200,             # W/m·K
            'area': 40e-3 * 40e-3 # 40x40 mm
        }
    ]
    
    # Parâmetros do dissipador
    heatsink_params = {
        'h': 80,                    # W/m²·K (ventilador médio)
        'k_fin': 200,               # W/m·K (alumínio)
        'fin_thickness': 1e-3,      # 1 mm
        'fin_width': 40e-3,         # 40 mm
        'fin_height': 30e-3,        # 30 mm
        'n_fins': 20,               # 20 aletas
        'base_length': 40e-3,       # 40 mm
        'base_width': 40e-3         # 40 mm
    }
    
    # Executar cálculo
    result = calculate_cpu_temperatures(power, T_amb, die_area, die_thickness, 
                                       die_k, layers, heatsink_params)
    
    # Mostrar resultados
    print(f"\n📊 Resultados:")
    print(f"   Potência: {result['power']} W")
    print(f"   Temperatura ambiente: {result['T_ambient']} °C")
    print(f"   Temperatura de junção: {result['T_junction']:.1f} °C")
    print(f"   Temperatura superfície do die: {result['T_die_surface']:.1f} °C")
    print(f"   Resistência térmica total: {result['R_total']:.3f} K/W")
    print(f"   Eficiência das aletas: {result['heatsink_details']['eta_f']:.2f}")
    
    print(f"\n🔍 Breakdown das resistências:")
    for name, resistance in result['R_breakdown']:
        percentage = 100 * resistance / result['R_total']
        print(f"   {name}: {resistance:.4f} K/W ({percentage:.1f}%)")
    
    # Verificação de sanidade
    if 60 <= result['T_junction'] <= 90:
        print(f"\n✅ Resultado plausível! Tj = {result['T_junction']:.1f}°C está na faixa esperada.")
    else:
        print(f"\n⚠️  Resultado fora do esperado. Verificar parâmetros.")
    
    return result

if __name__ == "__main__":
    test_basic_calculation()
