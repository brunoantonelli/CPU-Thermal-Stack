"""
CPU Thermal Stack Designer - Núcleo de Cálculos
Projeto de Fenômenos de Transporte - Condução 1D Estacionária
Autores: Monique Rosa de Moraes, Bruno Antonelli de Oliveira
"""

import math
import numpy as np
from materials import MATERIALS, FLUIDS, PUMP_PRESETS, RADIATOR_PRESETS, CPU_BLOCK_PRESETS, MINOR_LOSS_COEFFICIENTS
from scipy.interpolate import interp1d # Adicione esta linha

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

def convert_pressure(value, from_unit, to_unit):
    """Converte valores de pressão entre diferentes unidades."""
    # Constantes de conversão para Pascal (Pa)
    PA_PER_BAR = 100000.0
    PA_PER_MMH2O = 9.80665
    PA_PER_M_H2O = 9806.65

    if from_unit == to_unit:
        return value
    
    # Converter para Pa primeiro
    if from_unit == "bar":
        value_pa = value * PA_PER_BAR
    elif from_unit == "mmH₂O":
        value_pa = value * PA_PER_MMH2O
    elif from_unit == "m": # metro de coluna de água
        value_pa = value * PA_PER_M_H2O
    elif from_unit == "Pa":
        value_pa = value
    else:
        raise ValueError(f"Unidade de pressão '{from_unit}' não reconhecida.")
    
    # Converter de Pa para a unidade de destino
    if to_unit == "bar":
        return value_pa / PA_PER_BAR
    elif to_unit == "mmH₂O":
        return value_pa / PA_PER_MMH2O
    elif to_unit == "m":
        return value_pa / PA_PER_M_H2O
    elif to_unit == "Pa":
        return value_pa
    else:
        raise ValueError(f"Unidade de pressão '{to_unit}' não reconhecida.")

def convert_flow_rate(value, from_unit, to_unit):
    """Converte valores de vazão entre diferentes unidades."""
    # Constantes de conversão para m³/s
    M3S_PER_LPM = 1.0 / 60000.0

    if from_unit == to_unit:
        return value
    
    # Converter para m³/s primeiro
    if from_unit == "L/min":
        value_m3_s = value * M3S_PER_LPM
    elif from_unit == "m³/s":
        value_m3_s = value
    else:
        raise ValueError(f"Unidade de vazão '{from_unit}' não reconhecida.")
    
    # Converter de m³/s para a unidade de destino
    if to_unit == "L/min":
        return value_m3_s / M3S_PER_LPM
    elif to_unit == "m³/s":
        return value_m3_s
    else:
        raise ValueError(f"Unidade de vazão '{to_unit}' não reconhecida.")

# Importar MINOR_LOSS_COEFFICIENTS, FLUIDS, PUMP_PRESETS, RADIATOR_PRESETS, CPU_BLOCK_PRESETS
# do arquivo materials.py. Você precisará adicionar isso no topo do thermal_core.txt
# from materials import MINOR_LOSS_COEFFICIENTS, FLUIDS, PUMP_PRESETS, RADIATOR_PRESETS, CPU_BLOCK_PRESETS

def reynolds_number(density, velocity, hydraulic_diameter, viscosity):
    """Calcula o número de Reynolds."""
    if viscosity == 0: return 1e9 # Evitar divisão por zero, assumir turbulento
    return (density * velocity * hydraulic_diameter) / viscosity

def darcy_friction_factor(reynolds, relative_roughness=0.00001): # Para tubos lisos
    """Calcula o fator de atrito de Darcy."""
    if reynolds < 2300: # Escoamento laminar
        return 64 / reynolds
    else: # Escoamento turbulento (correlação de Haaland para Re > 4000)
        # Para simplificar, usaremos uma aproximação para tubos lisos ou Blasius para Re < 100k
        # Uma correlação mais geral seria a de Colebrook-White ou Haaland
        # Aqui, uma simplificação para turbulento em tubos lisos
        return (0.790 * math.log(reynolds) - 1.64)**-2 # Correlação de Haaland para tubos lisos
        # Ou Blasius para Re < 100k: return 0.316 / (reynolds**0.25)
        
def calculate_major_loss(friction_factor, length, hydraulic_diameter, velocity, density):
    """Calcula a perda de carga maior (por atrito)."""
    if hydraulic_diameter == 0: return 1e9
    return friction_factor * (length / hydraulic_diameter) * (0.5 * density * velocity**2)

def calculate_minor_loss(k_loss, velocity, density):
    """Calcula a perda de carga menor (por acessórios)."""
    return k_loss * (0.5 * density * velocity**2)

def calculate_circuit_pressure_drop(flow_rate_m3_s, fluid_props, cpu_block_params, radiator_params, tubing_params):
    """
    Calcula a queda de pressão total do circuito para uma dada vazão.
    Retorna a queda de pressão em Pa.
    """
    density = fluid_props['density']
    viscosity = fluid_props['viscosity']
    
    if flow_rate_m3_s <= 0: return 1e12 # Vazão zero, resistência infinita

    # --- Perdas na tubulação ---
    tubing_area = math.pi * (tubing_params['diameter_inner'] / 2)**2
    velocity_tubing = flow_rate_m3_s / tubing_area
    re_tubing = reynolds_number(density, velocity_tubing, tubing_params['diameter_inner'], viscosity)
    f_tubing = darcy_friction_factor(re_tubing)
    delta_p_tubing = calculate_major_loss(f_tubing, tubing_params['length'], tubing_params['diameter_inner'], velocity_tubing, density)

    # --- Perdas menores (curvas, conectores) ---
    # Simplificação: usa velocidade da tubulação para todas as perdas menores
    delta_p_minor_total = (tubing_params['num_bends'] * MINOR_LOSS_COEFFICIENTS['bend_90deg'] +
                           2 * MINOR_LOSS_COEFFICIENTS['fitting']) * (0.5 * density * velocity_tubing**2)
    
    # --- Perdas no bloco da CPU (microcanais) ---
    # Assumindo microcanais retangulares
    block_width = cpu_block_params['block_length_m'] # Usar como largura do bloco
    microchannel_area = cpu_block_params['microchannel_width_m'] * cpu_block_params['microchannel_height_m']
    total_microchannel_area = cpu_block_params['num_microchannels'] * microchannel_area
    
    if total_microchannel_area == 0: return 1e12
    
    velocity_block = flow_rate_m3_s / total_microchannel_area
    
    # Diâmetro hidráulico para canal retangular
    block_dh = 2 * microchannel_area / (cpu_block_params['microchannel_width_m'] + cpu_block_params['microchannel_height_m'])
    
    re_block = reynolds_number(density, velocity_block, block_dh, viscosity)
    f_block = darcy_friction_factor(re_block)
    
    # Comprimento efetivo dos microcanais (assumindo que é o comprimento do bloco)
    block_effective_length = cpu_block_params['block_length_m']
    delta_p_block = calculate_major_loss(f_block, block_effective_length, block_dh, velocity_block, density)
    
    # Adicionar perdas menores de entrada/saída do bloco
    delta_p_block += (MINOR_LOSS_COEFFICIENTS['entrance'] + MINOR_LOSS_COEFFICIENTS['exit']) * (0.5 * density * velocity_block**2)

    # --- Perdas no radiador ---
    # Simplificação: usar um fator de resistência do preset do radiador
    # Este fator precisa ser calibrado para retornar Pa para uma dada vazão
    # Por exemplo, um fator que representa K_L * (0.5 * rho * V_ref^2)
    # Para uma vazão, a queda de pressão no radiador pode ser modelada como C * (flow_rate_m3_s)^2
    radiator_resistance_coeff = RADIATOR_PRESETS[radiator_params['preset_name']].get('flow_resistance_coeff', 5e9) # Exemplo de coeficiente
    delta_p_radiator = radiator_resistance_coeff * (flow_rate_m3_s**2)
    
    # Adicionar perdas menores de entrada/saída do radiador
    delta_p_radiator += (MINOR_LOSS_COEFFICIENTS['entrance'] + MINOR_LOSS_COEFFICIENTS['exit']) * (0.5 * density * velocity_tubing**2) # Usar velocidade da tubulação como referência

    total_pressure_drop = delta_p_tubing + delta_p_minor_total + delta_p_block + delta_p_radiator
    return total_pressure_drop

def find_operating_point(pump_curve_data, circuit_resistance_func, max_flow_rate_guess=0.0002):
    """
    Encontra o ponto de operação (vazão e pressão) do watercooler.
    Usa interpolação para a curva da bomba e busca numérica.
    """
    pump_flow_rates_m3_s = np.array([p[0] for p in pump_curve_data])
    pump_pressures_pa = np.array([p[1] for p in pump_curve_data])
    
    # Cria uma função de interpolação para a curva da bomba
    pump_curve_interp = interp1d(pump_flow_rates_m3_s, pump_pressures_pa, kind='linear', fill_value="extrapolate")
    
    # Função objetivo: diferença entre a pressão da bomba e a pressão do circuito
    def objective_function(flow_rate):
        if flow_rate < 0: return 1e12 # Penaliza vazões negativas
        
        # Garante que a vazão esteja dentro do domínio da interpolação da bomba
        flow_rate = np.clip(flow_rate, pump_flow_rates_m3_s.min(), pump_flow_rates_m3_s.max())
        
        pump_pressure = pump_curve_interp(flow_rate)
        circuit_pressure = circuit_resistance_func(flow_rate)
        return pump_pressure - circuit_pressure
    
    # Busca pela vazão onde a função objetivo é zero (ponto de operação)
    # Podemos usar um método de busca de raiz, mas para simplificar, uma busca em um grid
    flow_rates_test = np.linspace(0, max_flow_rate_guess, 200) # Testar 200 pontos
    diffs = [objective_function(q) for q in flow_rates_test]
    
    # Encontrar o ponto onde a diferença é mínima (mais próxima de zero)
    idx = np.argmin(np.abs(diffs))
    operating_flow_rate = flow_rates_test[idx]
    operating_pressure = pump_curve_interp(operating_flow_rate) # Pressão no ponto de operação
    
    return operating_flow_rate, operating_pressure

def calculate_cpu_block_h(flow_rate_m3_s, fluid_props, cpu_block_params):
    """
    Calcula o coeficiente de convecção (h) dentro dos microcanais do bloco da CPU.
    """
    density = fluid_props['density']
    viscosity = fluid_props['viscosity']
    k_fluid = fluid_props['thermal_conductivity']
    cp_fluid = fluid_props['specific_heat']

    microchannel_area = cpu_block_params['microchannel_width_m'] * cpu_block_params['microchannel_height_m']
    total_microchannel_area = cpu_block_params['num_microchannels'] * microchannel_area
    
    if total_microchannel_area == 0: return 1e-9 # Evitar divisão por zero

    velocity_block = flow_rate_m3_s / total_microchannel_area
    
    block_dh = 2 * microchannel_area / (cpu_block_params['microchannel_width_m'] + cpu_block_params['microchannel_height_m'])
    
    re_block = reynolds_number(density, velocity_block, block_dh, viscosity)
    pr_fluid = (cp_fluid * viscosity) / k_fluid

    # Correlação de Nusselt para escoamento interno (simplificada para canais retangulares)
    # Para canais retangulares, Nu pode variar. Usaremos correlações comuns.
    if re_block < 2300: # Laminar (Nu para placas paralelas ou dutos retangulares)
        # Para placas paralelas com T constante na parede, Nu ~ 7.54
        # Para dutos retangulares, pode variar de 2.98 a 8.23 dependendo da razão de aspecto
        nu = 5.0 # Valor médio para canais retangulares
    else: # Turbulento (Dittus-Boelter para aquecimento do fluido)
        nu = 0.023 * (re_block**0.8) * (pr_fluid**0.4)
    
    if block_dh == 0: return 1e-9
    h_block = (nu * k_fluid) / block_dh
    return h_block

def calculate_radiator_thermal_resistance(radiator_params, fluid_props, flow_rate_m3_s, T_ambient):
    """
    Calcula a resistência térmica do radiador (do fluido para o ambiente).
    Este é um modelo simplificado.
    """
    # Um modelo mais robusto consideraria:
    # - h interno (fluido nos tubos do radiador)
    # - h externo (ar nas aletas do radiador, dependendo do fan_rpm e pressão estática)
    # - Resistência de condução das paredes do tubo e aletas
    # - Eficiência das aletas do radiador (lado ar)

    # Para a primeira versão, usamos um fator de resistência do preset do radiador
    # e ajustamos com base no RPM do fan.
    
    base_resistance_factor = RADIATOR_PRESETS[radiator_params['preset_name']]['thermal_resistance_factor']
    
    # Ajuste baseado no RPM do fan (exemplo: maior RPM -> menor resistência)
    # Assumimos que o fator de resistência é inversamente proporcional a alguma potência do RPM
    # RPM de referência para o fator base
    ref_fan_rpm = RADIATOR_PRESETS[radiator_params['preset_name']]['default_fan_rpm']
    
    if radiator_params['fan_rpm'] <= 0: return 1e12 # Sem fans, resistência infinita
    
    # Fator de escala do RPM (exemplo: raiz quadrada, pode ser ajustado)
    rpm_scale_factor = (radiator_params['fan_rpm'] / ref_fan_rpm)**0.5
    
    R_radiator_thermal = base_resistance_factor / rpm_scale_factor
    
    return R_radiator_thermal

def calculate_watercooler_temperatures(power_w, T_ambient, die_area, die_thickness, die_k,
                                      layers, fluid_props, pump_params, radiator_params,
                                      cpu_block_params, tubing_params, verbose=False):
    """
    Calcula temperaturas na pilha térmica da CPU com watercooler.
    """
    trace = [] if verbose else None

    # 1. Ajustar a curva da bomba para o RPM atual
    # A curva de dados do preset é para um RPM de referência.
    # Podemos escalar a pressão da curva com (RPM_atual / RPM_referencia)^2
    # Para simplificar, usaremos a curva do preset diretamente para o RPM padrão da bomba.
    # Se o RPM da bomba for um input, a curva deve ser ajustada.
    # Por enquanto, vamos assumir que pump_params['curve_data'] já reflete o RPM.
    
    # 2. Calcular a curva de resistência do circuito
    def circuit_resistance_func(q):
        return calculate_circuit_pressure_drop(q, fluid_props, cpu_block_params, radiator_params, tubing_params)
    
    # 3. Encontrar o ponto de operação (vazão e pressão)
    operating_flow_rate_m3_s, operating_pressure_pa = find_operating_point(
        pump_params['curve_data'], circuit_resistance_func
    )
    
    if trace is not None:
        trace.append(f"Vazão de operação: {convert_flow_rate(operating_flow_rate_m3_s, 'm³/s', 'L/min'):.2f} L/min")
        trace.append(f"Queda de pressão total: {convert_pressure(operating_pressure_pa, 'Pa', 'bar'):.2f} bar")

    # 4. Calcular o coeficiente de convecção (h) no bloco da CPU
    h_cpu_block = calculate_cpu_block_h(operating_flow_rate_m3_s, fluid_props, cpu_block_params)
    
    # 5. Calcular a resistência térmica do bloco da CPU (convecção do fluido para a parede do bloco)
    # Área de convecção dos microcanais
    microchannel_width = cpu_block_params['microchannel_width_m']
    microchannel_height = cpu_block_params['microchannel_height_m']
    block_length = cpu_block_params['block_length_m']
    num_microchannels = cpu_block_params['num_microchannels']

    # Área de superfície de convecção total nos microcanais
    A_convec_block = num_microchannels * (2 * microchannel_height * block_length + microchannel_width * block_length)
    
    R_conv_cpu_block = r_conv(h_cpu_block, A_convec_block)
    
    if trace is not None:
        trace.append(f"h_cpu_block: {h_cpu_block:.1f} W/m²K")
        trace.append(f"A_convec_block: {A_convec_block:.4e} m²")
        trace.append(f"R_conv_cpu_block: {R_conv_cpu_block:.4e} K/W")

    # 6. Calcular a resistência térmica do radiador (do fluido para o ambiente)
    R_radiator_thermal = calculate_radiator_thermal_resistance(radiator_params, fluid_props, operating_flow_rate_m3_s, T_ambient)
    
    if trace is not None:
        trace.append(f"R_radiator_thermal: {R_radiator_thermal:.4e} K/W")

    # 7. Balanço de energia para o fluido (para encontrar T_fluid_avg)
    # Q_cpu = power_w
    # Q_radiator = (T_fluid_avg - T_ambient) / R_radiator_thermal
    # Em regime estacionário, Q_cpu = Q_radiator
    T_fluid_avg = T_ambient + power_w * R_radiator_thermal
    
    if trace is not None:
        trace.append(f"T_fluid_avg = T_ambient + P * R_radiator = {T_ambient} + {power_w} * {R_radiator_thermal:.4e} = {T_fluid_avg:.1f} °C")

    # 8. Resistências das camadas (do die até o fluido no bloco)
    R_total_internal_stack = 0
    R_breakdown = []
    
    # Adicionar a resistência de condução do material do bloco da CPU
    # Simplificação: considerar uma espessura efetiva da base do bloco
    block_base_thickness = 3e-3 # Exemplo: 3mm de base do bloco
    block_base_area = die_area # Assumir que a área de condução é a área do die
    k_block_material = cpu_block_params['material_k']
    R_cond_block_base = r_cond(block_base_thickness, k_block_material, block_base_area)
    R_total_internal_stack += R_cond_block_base
    R_breakdown.append(('Bloco da CPU (Condução)', R_cond_block_base))
    
    if trace is not None:
        trace.append(f"R_cond_block_base: {R_cond_block_base:.4e} K/W")

    # Somar resistências das camadas (TIM, spreader)
    for layer in reversed(layers): # layers já deve vir sem a base do dissipador a ar
        R_layer = r_cond(layer['thickness'], layer['k'], layer['area'])
        R_total_internal_stack += R_layer
        R_breakdown.append((layer['name'], R_layer))
        if trace is not None:
            trace.append(f"{layer['name']}: R = {R_layer:.4e} K/W")

    # Adicionar a resistência de convecção do bloco da CPU para o fluido
    R_total_internal_stack += R_conv_cpu_block
    R_breakdown.append(('Bloco da CPU (Convecção para Fluido)', R_conv_cpu_block))

    # 9. Temperatura da superfície do die
    T_die_surface = T_fluid_avg + power_w * R_total_internal_stack
    if trace is not None:
        trace.append(f"T_die_surface = T_fluid_avg + P * R_internal_stack = {T_fluid_avg:.1f} + {power_w} * {R_total_internal_stack:.4e} = {T_die_surface:.1f} °C")

    # 10. Geração interna no die
    q_dot = power_w / (die_area * die_thickness)
    delta_T_generation = q_dot * die_thickness**2 / (8.0 * die_k)
    if trace is not None:
        trace.append(f"q_dot = {q_dot:.4e} W/m³")
        trace.append(f"ΔT (geração) = {delta_T_generation:.4e} K")
    
    # 11. Temperatura de junção (centro do die)
    T_junction = T_die_surface + delta_T_generation
    
    # 12. Resistência térmica total do sistema (do die para o ambiente)
    R_total_system = R_total_internal_stack + R_radiator_thermal
    
    results = {
        'T_ambient': T_ambient,
        'T_die_surface': T_die_surface,
        'T_junction': T_junction,
        'R_total': R_total_system,
        'R_breakdown': list(reversed(R_breakdown)) + [('Radiador + Ambiente', R_radiator_thermal)], # Ordem do die para o ambiente
        'delta_T_generation': delta_T_generation,
        'heatsink_details': {}, # Watercooler não tem heatsink_details como air cooler
        'power': power_w,
        'q_dot': q_dot,
        'flow_rate_m3_s': operating_flow_rate_m3_s,
        'flow_rate_lpm': convert_flow_rate(operating_flow_rate_m3_s, 'm³/s', 'L/min'),
        'total_pressure_drop_pa': operating_pressure_pa,
        'total_pressure_drop_bar': convert_pressure(operating_pressure_pa, 'Pa', 'bar'),
        'T_fluid_avg': T_fluid_avg,
        'h_cpu_block': h_cpu_block
    }
    
    if trace is not None:
        results['trace'] = trace
    return results

def fin_efficiency_rectangular(h, k_fin, thickness, width, height, trace=None):
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
        if trace is not None:
            trace.append(f"A_c={A_c:.4e} m² ou k_fin={k_fin} inválido → eta_f=1.0")
        return 1.0, A_c, P

    m = math.sqrt(h * P / (k_fin * A_c))

    # Eficiência (ponta adiabática)
    if m * height == 0:
        eta_f = 1.0
    else:
        eta_f = math.tanh(m * height) / (m * height)

    if trace is not None:
        trace.append(f"A_c={A_c:.4e} m², P={P:.4e} m; m={m:.4e} 1/m; m*h={m*height:.4e}; eta_f={eta_f:.4f}")

    return eta_f, A_c, P

def heatsink_thermal_resistance(h, k_fin, fin_thickness, fin_width, fin_height, 
                               n_fins, base_length, base_width, trace=None):
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
    eta_f, A_c, P = fin_efficiency_rectangular(h, k_fin, fin_thickness, fin_width, fin_height, trace=trace)
    
    # Área de convecção de uma aleta (2 faces + ponta)
    A_fin_single = 2 * fin_width * fin_height + fin_thickness * fin_width
    A_fins_total = n_fins * A_fin_single
    
    # Área efetiva total para convecção
    A_eff = A_base_exposed + eta_f * A_fins_total

    # Resistência térmica de convecção
    R_conv = 1.0 / (h * A_eff) if A_eff > 0 else float('inf')

    if trace is not None:
        trace.append(f"A_base_total={A_base_total:.4e} m²; A_base_exposed={A_base_exposed:.4e} m²")
        trace.append(f"A_fin_single={A_fin_single:.4e} m²; A_fins_total={A_fins_total:.4e} m²; A_eff={A_eff:.4e} m²")
        trace.append(f"h={h} W/m²K → R_conv=1/(h*A_eff)={R_conv:.4e} K/W")
    
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

# Renomeie esta função
def calculate_air_cooler_temperatures(power_w, T_ambient, die_area, die_thickness, die_k,
                              layers, heatsink_params, verbose=False):
    """
    Calcula temperaturas na pilha térmica da CPU com air cooler.
    
    
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
    
    # Preparar trace opcional
    trace = [] if verbose else None

    # Resistência térmica do dissipador (convecção + aletas)
    R_heatsink, hs_details = heatsink_thermal_resistance(trace=trace, **heatsink_params)

    if trace is not None:
        trace.append(f"Resistência do dissipador (convecção + aletas): R_heatsink={R_heatsink:.4e} K/W")
    
    # Resistências das camadas (de cima para baixo na pilha)
    R_total = R_heatsink
    R_breakdown = [('Dissipador + Convecção', R_heatsink)]
    
    # Somar resistências das camadas (TIM, spreader, base, etc.)
    for layer in reversed(layers):  # reversed porque vamos do dissipador para o die
        R_layer = r_cond(layer['thickness'], layer['k'], layer['area'])
        R_total += R_layer
        R_breakdown.append((layer['name'], R_layer))
        if trace is not None:
            trace.append(f"{layer['name']}: R = L/(k*A) = {layer['thickness']:.4e}/({layer['k']}*{layer['area']:.4e}) = {R_layer:.4e} K/W")
    
    # Temperatura na superfície do die
    T_die_surface = T_ambient + power_w * R_total
    if trace is not None:
        trace.append(f"T_die_surface = T_amb + P*R_total = {T_ambient} + {power_w}*{R_total:.4e} = {T_die_surface:.4f} °C")
    
    # Geração interna no die (modelo de placa com geração uniforme)
    q_dot = power_w / (die_area * die_thickness)  # W/m³
    delta_T_generation = q_dot * die_thickness**2 / (8.0 * die_k)
    if trace is not None:
        trace.append(f"q_dot = P/(A*L) = {power_w}/({die_area:.4e}*{die_thickness:.4e}) = {q_dot:.4e} W/m³")
        trace.append(f"ΔT (geração) = q_dot*L²/(8*k) = {q_dot:.4e}*{die_thickness:.4e}²/(8*{die_k}) = {delta_T_generation:.4e} K")
    
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
    
    if trace is not None:
        results['trace'] = trace

    return results

def calculate_thermal_performance(cooling_type, power_w, T_ambient, die_area, die_thickness, die_k,
                                 layers, heatsink_params=None, fluid_props=None, pump_params=None,
                                 radiator_params=None, cpu_block_params=None, tubing_params=None, verbose=False):
    """
    Função wrapper para calcular o desempenho térmico com base no tipo de resfriamento.
    """
    if cooling_type == "Air Cooler":
        if heatsink_params is None:
            raise ValueError("Parâmetros do heatsink são necessários para Air Cooler.")
        return calculate_air_cooler_temperatures(power_w, T_ambient, die_area, die_thickness, die_k,
                                                layers, heatsink_params, verbose)
    elif cooling_type == "Watercooler":
        if any(p is None for p in [fluid_props, pump_params, radiator_params, cpu_block_params, tubing_params]):
            raise ValueError("Todos os parâmetros do watercooler são necessários para Watercooler.")
        return calculate_watercooler_temperatures(power_w, T_ambient, die_area, die_thickness, die_k,
                                                 layers, fluid_props, pump_params, radiator_params,
                                                 cpu_block_params, tubing_params, verbose)
    else:
        raise ValueError(f"Tipo de resfriamento '{cooling_type}' não reconhecido.")


# Função de teste rápido para Air Cooler (atualizada)
def test_air_cooler_calculation(): # Renomeada para clareza
    """Teste rápido para verificar se os cálculos do Air Cooler estão funcionando"""
    
    print("🧪 Testando cálculos básicos do Air Cooler...")
    
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
    
    # Executar cálculo usando a nova função de despacho
    result = calculate_thermal_performance(
        "Air Cooler", power, T_amb, die_area, die_thickness, 
        die_k, layers, heatsink_params=heatsink_params, verbose=True
    )
    
    # Mostrar resultados
    print(f"\n📊 Resultados (Air Cooler):")
    print(f"   Potência: {result['power']} W")
    print(f"   Temperatura ambiente: {result['T_ambient']} °C")
    print(f"   Temperatura de junção: {result['T_junction']:.1f} °C")
    print(f"   Temperatura superfície do die: {result['T_die_surface']:.1f} °C")
    print(f"   Resistência térmica total: {result['R_total']:.3f} K/W")
    print(f"   Eficiência das aletas: {result['heatsink_details']['eta_f']:.2f}")
    
    print(f"\n🔍 Breakdown das resistências (Air Cooler):")
    for name, resistance in result['R_breakdown']:
        percentage = 100 * resistance / result['R_total']
        print(f"   {name}: {resistance:.4f} K/W ({percentage:.1f}%)")
    
    # Verificação de sanidade
    if 60 <= result['T_junction'] <= 90:
        print(f"\n✅ Resultado plausível para Air Cooler! Tj = {result['T_junction']:.1f}°C está na faixa esperada.")
    else:
        print(f"\n⚠️  Resultado Air Cooler fora do esperado. Verificar parâmetros.")
    
    return result

# Função de teste rápido para Watercooler
def test_watercooler_calculation():
    """Teste rápido para verificar se os cálculos do Watercooler estão funcionando"""
    
    print("\n\n🧪 Testando cálculos básicos do Watercooler...")
    
    # Parâmetros de teste - CPU típica de 95W
    power = 95  # W
    T_amb = 25  # °C
    
    # Die (chip)
    die_area = 12e-3 * 12e-3  # 12x12 mm
    die_thickness = 0.5e-3    # 0.5 mm
    die_k = 120               # W/m·K (silício)
    
    # Camadas da pilha térmica (TIM, Spreader - sem a base do dissipador a ar)
    layers_wc = [
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
        }
    ]

    # Parâmetros do Watercooler
    fluid_props = FLUIDS['Água Destilada']
    pump_params = PUMP_PRESETS['Bomba D5 (Padrão)']
    radiator_params = RADIATOR_PRESETS['Radiador 360mm (Performance)']
    cpu_block_params = CPU_BLOCK_PRESETS['Microcanais Padrão']
    cpu_block_params['material_k'] = MATERIALS['heatsink']['Cobre'] # Material do cold plate
    tubing_params = {
        'length': 1.0, # 1 metro
        'diameter_inner': 0.01, # 10 mm
        'num_bends': 4
    }
    
    # Executar cálculo do watercooler
    wc_result = calculate_thermal_performance(
        "Watercooler", power, T_amb, die_area, die_thickness, die_k,
        layers_wc, fluid_props=fluid_props, pump_params=pump_params,
        radiator_params=radiator_params, cpu_block_params=cpu_block_params,
        tubing_params=tubing_params, verbose=True
    )
    
    print(f"\n📊 Resultados (Watercooler):")
    print(f"   Potência: {wc_result['power']} W")
    print(f"   Temperatura ambiente: {wc_result['T_ambient']} °C")
    print(f"   Temperatura de junção: {wc_result['T_junction']:.1f} °C")
    print(f"   Temperatura superfície do die: {wc_result['T_die_surface']:.1f} °C")
    print(f"   Resistência térmica total: {wc_result['R_total']:.3f} K/W")
    print(f"   Vazão do fluido: {wc_result['flow_rate_lpm']:.2f} L/min")
    print(f"   Queda de pressão total: {wc_result['total_pressure_drop_bar']:.3f} bar")
    print(f"   Temperatura média do fluido: {wc_result['T_fluid_avg']:.1f} °C")
    print(f"   h no bloco da CPU: {wc_result['h_cpu_block']:.1f} W/m²K")

    print(f"\n🔍 Breakdown das resistências (Watercooler):")
    for name, resistance in wc_result['R_breakdown']:
        percentage = 100 * resistance / wc_result['R_total']
        print(f"   {name}: {resistance:.4f} K/W ({percentage:.1f}%)")

    if 60 <= wc_result['T_junction'] <= 90:
        print(f"\n✅ Resultado plausível para Watercooler! Tj = {wc_result['T_junction']:.1f}°C.")
    else:
        print(f"\n⚠️  Resultado Watercooler fora do esperado. Verificar parâmetros.")
    
    return wc_result


if __name__ == "__main__":
    air_cooler_result = test_air_cooler_calculation()
    watercooler_result = test_watercooler_calculation()
