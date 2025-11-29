"""
CPU Thermal Stack Designer - Interface Web
Streamlit App
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importar nossos módulos
from thermal_core import calculate_cpu_temperatures
from materials import MATERIALS, CONVECTION, CPU_PRESETS, HEATSINK_PRESETS
from sim_2d_conduction import run_2d_simulation
from sim_side_view import run_side_view_simulation

# Configuração da página
st.set_page_config(
    page_title="CPU Thermal Designer", 
    page_icon="🔥",
    layout="wide"
)

# Título principal
st.title("🔥 CPU Thermal Stack Designer")
st.markdown("**Simulador de Pilha Térmica usando Condução 1D Estacionária**")
st.markdown("*Projeto de Fenômenos de Transporte - Monique Moraes & Bruno Oliveira*")

# Sidebar para inputs
st.sidebar.header("⚙️ Configuração do Sistema")

# === SEÇÃO 1: CPU ===
st.sidebar.subheader("💻 Processador")
cpu_preset = st.sidebar.selectbox("Preset de CPU:", list(CPU_PRESETS.keys()))
preset_data = CPU_PRESETS[cpu_preset]

power = st.sidebar.slider("Potência (W):", 30, 300, preset_data['power'])
T_ambient = st.sidebar.slider("Temperatura ambiente (°C):", 15, 45, 25)

# === SEÇÃO 2: MATERIAIS ===
st.sidebar.subheader("🧪 Materiais")

# Die material
die_material = st.sidebar.selectbox("Material do Die:", list(MATERIALS['die'].keys()))
die_k = MATERIALS['die'][die_material]

# TIM
tim_material = st.sidebar.selectbox("Interface Térmica (TIM):", list(MATERIALS['tim'].keys()))
tim_k = MATERIALS['tim'][tim_material]
tim_thickness = st.sidebar.selectbox("Espessura TIM:", 
    ["50 μm (fina)", "100 μm (normal)", "200 μm (grossa)"])
tim_thickness_val = {'50 μm (fina)': 50e-6, '100 μm (normal)': 100e-6, '200 μm (grossa)': 200e-6}[tim_thickness]

# Spreader
spreader_material = st.sidebar.selectbox("Heat Spreader:", list(MATERIALS['spreader'].keys()))
spreader_k = MATERIALS['spreader'][spreader_material]

# Heatsink
heatsink_material = st.sidebar.selectbox("Material do Dissipador:", list(MATERIALS['heatsink'].keys()))
heatsink_k = MATERIALS['heatsink'][heatsink_material]

# === SEÇÃO 3: RESFRIAMENTO ===
st.sidebar.subheader("❄️ Resfriamento")
cooling_method = st.sidebar.selectbox("Método de Resfriamento:", list(CONVECTION.keys()))
h = CONVECTION[cooling_method]

# === SEÇÃO 4: GEOMETRIA ===
st.sidebar.subheader("📐 Geometria do Dissipador")

col1, col2 = st.sidebar.columns(2)
with col1:
    n_fins = st.slider("Nº Aletas:", 5, 50, 20)
    fin_height = st.slider("Altura (mm):", 10, 80, 30)
    
with col2:
    fin_thickness = st.slider("Espessura (mm):", 0.3, 3.0, 1.0)
    base_size = st.slider("Base (mm):", 30, 80, 40)

# === CÁLCULOS ===

# Parâmetros do die
die_area = preset_data['die_area']
die_thickness = preset_data['die_thickness']

# Camadas da pilha térmica
layers = [
    {
        'name': f'TIM ({tim_material})', 
        'thickness': tim_thickness_val, 
        'k': tim_k, 
        'area': (base_size * 1e-3) ** 2
    },
    {
        'name': f'Spreader ({spreader_material})', 
        'thickness': 2e-3, 
        'k': spreader_k, 
        'area': (base_size * 1e-3) ** 2
    },
    {
        'name': f'Base ({heatsink_material})', 
        'thickness': 3e-3, 
        'k': heatsink_k, 
        'area': (base_size * 1e-3) ** 2
    }
]

# Parâmetros do dissipador
heatsink_params = {
    'h': h,
    'k_fin': heatsink_k,
    'fin_thickness': fin_thickness * 1e-3,
    'fin_width': base_size * 1e-3,
    'fin_height': fin_height * 1e-3,
    'n_fins': n_fins,
    'base_length': base_size * 1e-3,
    'base_width': base_size * 1e-3
}

# Executar cálculo
result = calculate_cpu_temperatures(power, T_ambient, die_area, die_thickness, 
                                   die_k, layers, heatsink_params, verbose=True)

# === RESULTADOS PRINCIPAIS ===
st.header("📊 Resultados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    tj_color = "normal"
    if result['T_junction'] > 85:
        tj_color = "inverse"
    elif result['T_junction'] > 75:
        tj_color = "off"
    
    st.metric("🌡️ Temperatura de Junção", f"{result['T_junction']:.1f} °C")

with col2:
    st.metric("🔥 Superfície do Die", f"{result['T_die_surface']:.1f} °C")

with col3:
    st.metric("⚡ Resistência Total", f"{result['R_total']:.3f} K/W")

with col4:
    st.metric("🎯 Eficiência das Aletas", f"{result['heatsink_details']['eta_f']:.1%}")

# Status da temperatura
if result['T_junction'] <= 70:
    st.success(f"✅ Temperatura excelente! Tj = {result['T_junction']:.1f}°C")
elif result['T_junction'] <= 85:
    st.warning(f"⚠️ Temperatura aceitável. Tj = {result['T_junction']:.1f}°C")
else:
    st.error(f"🚨 Temperatura muito alta! Tj = {result['T_junction']:.1f}°C - Risco de throttling!")

# === GRÁFICO DE RESISTÊNCIAS ===
st.subheader("🔍 Análise de Resistências Térmicas")

# Preparar dados
resistance_data = pd.DataFrame(result['R_breakdown'], columns=['Componente', 'Resistência (K/W)'])
resistance_data['Percentual'] = 100 * resistance_data['Resistência (K/W)'] / result['R_total']

# Criar gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras
bars = ax1.barh(resistance_data['Componente'], resistance_data['Resistência (K/W)'], 
                color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
ax1.set_xlabel('Resistência Térmica (K/W)')
ax1.set_title('Contribuição de cada camada')

# Adicionar valores nas barras
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}\n({resistance_data.iloc[i]["Percentual"]:.1f}%)',
            ha='left', va='center', fontsize=9)

# Gráfico de pizza
ax2.pie(resistance_data['Percentual'], labels=resistance_data['Componente'], 
        autopct='%1.1f%%', startangle=90, colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
ax2.set_title('Distribuição das Resistências')

plt.tight_layout()
st.pyplot(fig)

# === DETALHES DO DISSIPADOR ===
st.subheader("🔧 Detalhes do Dissipador")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Área Base Total", f"{result['heatsink_details']['A_base_total']*1e6:.0f} mm²")
    st.metric("Área Base Exposta", f"{result['heatsink_details']['A_base_exposed']*1e6:.0f} mm²")

with col2:
    st.metric("Área das Aletas", f"{result['heatsink_details']['A_fins_total']*1e6:.0f} mm²")
    st.metric("Área Efetiva Total", f"{result['heatsink_details']['A_eff']*1e6:.0f} mm²")

with col3:
    st.metric("Número de Aletas", f"{n_fins}")
    st.metric("Eficiência das Aletas", f"{result['heatsink_details']['eta_f']:.1%}")

# === INFORMAÇÕES TÉCNICAS ===
with st.expander("📋 Informações Técnicas"):
    st.write("**Parâmetros do Die:**")
    st.write(f"- Área: {die_area*1e6:.1f} mm²")
    st.write(f"- Espessura: {die_thickness*1e3:.1f} mm") 
    st.write(f"- Condutividade: {die_k} W/m·K")
    st.write(f"- Geração volumétrica: {result['q_dot']/1e6:.1f} MW/m³")
    
    st.write("**Configuração da Pilha:**")
    for layer in layers:
        st.write(f"- {layer['name']}: {layer['thickness']*1e6:.0f} μm, k = {layer['k']} W/m·K")

# === PASSO-A-PASSO DOS CÁLCULOS ===
with st.expander("🧾 Cálculos (Passo a passo)"):
    trace = result.get('trace', None)
    if trace:
        # Mostrar todo o trace como bloco de código para fácil cópia
        st.code("\n".join(trace), language='text')
    else:
        st.write("Nenhum detalhe passo-a-passo disponível.")

# === SIMULAÇÃO 2D (PROTÓTIPO) ===
st.header("🧪 Simulação 2D - Condução (Protótipo)")
with st.expander("Configurar simulação 2D"):
    sim_nx = st.slider('Resolução X (nx)', 40, 240, 120)
    sim_ny = st.slider('Resolução Y (ny)', 40, 240, 120)
    sim_Lx = st.number_input('Largura do domínio (m)', value=0.06)
    sim_Ly = st.number_input('Altura do domínio (m)', value=0.06)
    sim_fin_height = st.number_input('Altura das aletas (m)', value=fin_height * 1e-3)

if st.button('▶️ Rodar simulação 2D'):
    params = {
        'n_fins': n_fins,
        'fin_thickness': fin_thickness * 1e-3,
        'fin_height': sim_fin_height,
        'base_width': base_size * 1e-3,
        'base_height': 0.01,
        'k_al': heatsink_k,
        'k_die': die_k,
        'die_width': np.sqrt(die_area),
        'die_height': np.sqrt(die_area),
        'die_thickness': die_thickness,
        'h': h
    }
    with st.spinner('Rodando simulação 2D (pode demorar alguns segundos)...'):
        try:
            img_bytes = run_2d_simulation(power_w=power, T_amb=T_ambient,
                                         nx=sim_nx, ny=sim_ny, Lx=sim_Lx, Ly=sim_Ly,
                                         params=params)
            st.image(img_bytes, caption='Mapa de Temperatura (2D)', use_column_width=True)
        except Exception as e:
            st.error(f'Erro ao rodar simulação: {e}')

# === VISTA LATERAL (SEÇÃO TRANSVERSAL) ===
st.header("🔎 Vista Lateral - Seção Transversal (simplificada)")
with st.expander("Configurar vista lateral e modelo de convecção"):
    side_nx = st.slider('Resolução Lateral (nx)', 40, 240, 120)
    side_ny = st.slider('Resolução Vertical (ny)', 80, 320, 160)
    side_Lx = st.number_input('Largura do domínio lateral (m)', value=base_size * 1e-3)
    side_Ly = st.number_input('Altura do domínio (m)', value=fin_height * 1e-3 + 0.02)
    airflow_mode = st.selectbox('Modelo de convecção:', ['velocity', 'h_manual'], index=0)
    if airflow_mode == 'velocity':
        air_U = st.slider('Velocidade do ar U (m/s):', 0.0, 10.0, 2.0, 0.1)
    else:
        air_h_manual = st.number_input('Coeficiente convectivo h (W/m²K):', value=20.0)

if st.button('▶️ Rodar vista lateral'):
    # preparar parâmetros para sim_side_view
    side_params = {
        'die_width': np.sqrt(die_area),
        'die_thickness': die_thickness,
        'tim_thickness': tim_thickness_val,
        'spreader_thickness': 2e-3,
        'base_thickness': 3e-3,
        'fin_height': fin_height * 1e-3,
        'fin_thickness': fin_thickness * 1e-3,
        'n_fins': n_fins,
        'base_width': base_size * 1e-3,
        'k_tim': tim_k,
        'k_die': die_k,
        'k_al': heatsink_k,
        'k_cu': MATERIALS['spreader'].get(spreader_material, 390)
    }

    airflow = {'mode': airflow_mode}
    if airflow_mode == 'velocity':
        airflow['U'] = air_U
    else:
        airflow['h'] = air_h_manual

    with st.spinner('Rodando vista lateral (pode demorar)...'):
        try:
            img_side, side_summary = run_side_view_simulation(power_w=power, T_amb=T_ambient,
                                                             nx=side_nx, ny=side_ny,
                                                             Lx=side_Lx, Ly=side_Ly,
                                                             params=side_params,
                                                             airflow=airflow)
            st.image(img_side, caption='Vista Lateral - Mapa de Temperatura', use_column_width=True)

            # Mostrar resumo
            st.subheader('Resumo - Vista Lateral')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('h estimado (W/m²K)', f"{side_summary['h']:.1f}")
            with col2:
                st.metric('T máx (°C)', f"{side_summary['T_max']:.1f}")
            with col3:
                tcenter = side_summary.get('T_center_die', None)
                st.metric('T média no die (°C)', f"{tcenter:.1f}" if tcenter is not None else 'N/A')

            with st.expander('Detalhes da estimativa de convecção'):
                st.write(side_summary.get('h_details', {}))

            st.info('Modelo simplificado: correlações 1D/2D — para análise detalhada de mecânica dos fluidos use um solver CFD (ex.: OpenFOAM).')

        except Exception as e:
            st.error(f'Erro na vista lateral: {e}')

# === COMPARAÇÃO RÁPIDA ===
st.subheader("⚖️ Comparação Rápida")

if st.button("🔄 Comparar com Configuração Básica"):
    # Configuração básica de referência
    basic_layers = [
        {'name': 'TIM Básica', 'thickness': 100e-6, 'k': 3.0, 'area': (40e-3)**2},
        {'name': 'Spreader Al', 'thickness': 2e-3, 'k': 200, 'area': (40e-3)**2},
        {'name': 'Base Al', 'thickness': 3e-3, 'k': 200, 'area': (40e-3)**2}
    ]
    
    basic_heatsink = {
        'h': 45, 'k_fin': 200, 'fin_thickness': 1e-3,
        'fin_width': 40e-3, 'fin_height': 25e-3, 'n_fins': 15,
        'base_length': 40e-3, 'base_width': 40e-3
    }
    
    basic_result = calculate_cpu_temperatures(power, T_ambient, die_area, die_thickness, 
                                            die_k, basic_layers, basic_heatsink)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Configuração Básica", f"{basic_result['T_junction']:.1f} °C")
    with col2:
        st.metric("Sua Configuração", f"{result['T_junction']:.1f} °C")
    with col3:
        delta = result['T_junction'] - basic_result['T_junction']
        st.metric("Diferença", f"{delta:+.1f} °C")
        
    if delta < 0:
        st.success(f"🎉 Sua configuração é {abs(delta):.1f}°C melhor!")
    else:
        st.info(f"💡 A configuração básica é {delta:.1f}°C melhor.")