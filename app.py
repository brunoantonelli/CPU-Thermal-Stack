"""
CPU Thermal Stack Designer - Interface Web
Streamlit App
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importar nossos módulos
from thermal_core import calculate_thermal_performance, convert_pressure, convert_flow_rate
from materials import MATERIALS, CONVECTION, CPU_PRESETS, HEATSINK_PRESETS, FLUIDS, PUMP_PRESETS, RADIATOR_PRESETS, CPU_BLOCK_PRESETS
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

# --- INICIALIZAÇÃO DE VARIÁVEIS PARA AIR COOLER ---
# Estas variáveis precisam de um valor padrão caso o Watercooler seja selecionado,
# pois são usadas fora do bloco 'if cooling_type == "Air Cooler"'.
# Se o Air Cooler for selecionado, elas serão sobrescritas pelos sliders.
h = 45.0 # Coeficiente de convecção padrão (W/m²K)
heatsink_material = "Alumínio" # Material padrão do dissipador
heatsink_k = MATERIALS['heatsink'][heatsink_material] # Condutividade padrão
n_fins = 20 # Número de aletas padrão
fin_height = 30 # Altura da aleta padrão (mm)
fin_thickness = 1.0 # Espessura da aleta padrão (mm)
base_size = 40 # Tamanho da base padrão (mm) para TIM/Spreader e para simulações 2D/lateral

# === SEÇÃO 3: RESFRIAMENTO ===
st.sidebar.subheader("❄️ Resfriamento")
cooling_type = st.sidebar.radio("Tipo de Resfriamento:", ["Air Cooler", "Watercooler"])

# --- Configuração Air Cooler ---
if cooling_type == "Air Cooler":
    st.sidebar.markdown("---")
    cooling_method = st.sidebar.selectbox("Método de Resfriamento:", list(CONVECTION.keys()))
    h = CONVECTION[cooling_method] # Atualiza 'h' com o valor selecionado

    # Heatsink
    heatsink_material = st.sidebar.selectbox("Material do Dissipador:", list(MATERIALS['heatsink'].keys()))
    heatsink_k = MATERIALS['heatsink'][heatsink_material] # Atualiza 'heatsink_k' com o valor selecionado

    st.sidebar.subheader("📐 Geometria do Dissipador")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        n_fins = st.slider("Nº Aletas:", 5, 50, 20, key="air_n_fins") # Atualiza 'n_fins'
        fin_height = st.slider("Altura (mm):", 10, 80, 30, key="air_fin_height") # Atualiza 'fin_height'
    with col2:
        fin_thickness = st.slider("Espessura (mm):", 0.3, 3.0, 1.0, key="air_fin_thickness") # Atualiza 'fin_thickness'
        base_size = st.slider("Base (mm):", 30, 80, 40, key="air_base_size") # Atualiza 'base_size'

# --- Configuração Watercooler ---
else: # cooling_type == "Watercooler"
    st.sidebar.markdown("---")

    # Fluido
    fluid_type = st.sidebar.selectbox("Fluido de Refrigeração:", list(FLUIDS.keys()))
    fluid_props = FLUIDS[fluid_type]

    # Bomba
    st.sidebar.markdown("#### Bomba")
    pump_preset_name = st.sidebar.selectbox("Preset da Bomba:", list(PUMP_PRESETS.keys()))
    pump_preset_data = PUMP_PRESETS[pump_preset_name]
    pump_rpm = st.sidebar.slider("RPM da Bomba:", 1000, 3500, pump_preset_data['default_rpm'])

    # Radiador
    st.sidebar.markdown("#### Radiador")
    radiator_preset_name = st.sidebar.selectbox("Preset do Radiador:", list(RADIATOR_PRESETS.keys()))
    radiator_preset_data = RADIATOR_PRESETS[radiator_preset_name]
    radiator_fan_rpm = st.sidebar.slider("RPM dos Ventiladores do Radiador:", 800, 3000, radiator_preset_data['default_fan_rpm'])

    # Bloco da CPU (Cold Plate)
    st.sidebar.markdown("#### Bloco da CPU")
    cpu_block_material = st.sidebar.selectbox("Material do Bloco:", list(MATERIALS['heatsink'].keys()))
    cpu_block_preset_name = st.sidebar.selectbox("Preset de Microcanais:", list(CPU_BLOCK_PRESETS.keys()))
    cpu_block_preset_data = CPU_BLOCK_PRESETS[cpu_block_preset_name]
    # Permitir ajuste fino dos microcanais
    num_microchannels = st.sidebar.slider("Nº de Microcanais:", 10, 100, cpu_block_preset_data['num_microchannels'])
    microchannel_width = st.sidebar.slider("Largura Microcanal (mm):", 0.1, 1.0, cpu_block_preset_data['microchannel_width_m'] * 1e3, step=0.05) * 1e-3
    microchannel_height = st.sidebar.slider("Altura Microcanal (mm):", 0.5, 3.0, cpu_block_preset_data['microchannel_height_m'] * 1e3, step=0.1) * 1e-3
    block_length = st.sidebar.slider("Comprimento do Bloco (mm):", 20.0, 60.0, cpu_block_preset_data['block_length_m'] * 1e3, step=1.0) * 1e-3

    # Tubulação
    st.sidebar.markdown("#### Tubulação")
    tubing_length = st.sidebar.slider("Comprimento Total da Tubulação (m):", 0.5, 5.0, 1.0)
    tubing_diameter_inner = st.sidebar.slider("Diâmetro Interno da Tubulação (mm):", 5, 15, 10) * 1e-3
    num_bends = st.sidebar.slider("Nº de Curvas de 90°:", 0, 12, 4)

    # Parâmetros para o watercooler
    watercooler_params = {
        'fluid_props': fluid_props,
        'pump_params': {
            'preset_name': pump_preset_name,
            'rpm': pump_rpm,
            'curve_data': pump_preset_data['curve_data']
        },
        'radiator_params': {
            'preset_name': radiator_preset_name,
            'size_mm': radiator_preset_data['size_mm'],
            'fan_rpm': radiator_fan_rpm
        },
        'cpu_block_params': {
            'material_k': MATERIALS['heatsink'][cpu_block_material],
            'num_microchannels': num_microchannels,
            'microchannel_width_m': microchannel_width,
            'microchannel_height_m': microchannel_height,
            'block_length_m': block_length
        },
        'tubing_params': {
            'length': tubing_length,
            'diameter_inner': tubing_diameter_inner,
            'num_bends': num_bends
        }
    }


# === CÁLCULOS ===

# Parâmetros do die
die_area = preset_data['die_area']
die_thickness = preset_data['die_thickness']
die_k = MATERIALS['die'][die_material]

# Camadas da pilha térmica (TIM, Spreader)
# 'base_size' agora está sempre definida devido à inicialização padrão.
layers_stack = [
    {
        'name': f'TIM ({tim_material})',
        'thickness': tim_thickness_val,
        'k': tim_k,
        'area': (base_size * 1e-3) ** 2 # Usar base_size como referência para área
    },
    {
        'name': f'Spreader ({spreader_material})',
        'thickness': 2e-3,
        'k': spreader_k,
        'area': (base_size * 1e-3) ** 2 # Usar base_size como referência para área
    }
]

# Parâmetros do dissipador a ar (se for o caso)
heatsink_params = None
if cooling_type == "Air Cooler":
    # Adicionar a base do dissipador a ar às camadas
    layers_stack.append({
        'name': f'Base ({heatsink_material})',
        'thickness': 3e-3,
        'k': heatsink_k,
        'area': (base_size * 1e-3) ** 2
    })
    heatsink_params = {
        'h': h,
        'k_fin': heatsink_k,
        'fin_thickness': fin_thickness * 1e-3,
        'fin_width': base_size * 1e-3, # Largura da aleta igual à base
        'fin_height': fin_height * 1e-3,
        'n_fins': n_fins,
        'base_length': base_size * 1e-3,
        'base_width': base_size * 1e-3
    }

# Executar cálculo
if cooling_type == "Air Cooler":
    result = calculate_thermal_performance(
        "Air Cooler", power, T_ambient, die_area, die_thickness, die_k,
        layers_stack, heatsink_params=heatsink_params, verbose=True
    )
else: # Watercooler
    result = calculate_thermal_performance(
        "Watercooler", power, T_ambient, die_area, die_thickness, die_k,
        layers_stack, **watercooler_params, verbose=True
    )

# === RESULTADOS PRINCIPAIS ===
st.header("📊 Resultados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌡️ Temperatura de Junção", f"{result['T_junction']:.1f} °C")

with col2:
    st.metric("🔥 Superfície do Die", f"{result['T_die_surface']:.1f} °C")

with col3:
    st.metric("⚡ Resistência Total", f"{result['R_total']:.3f} K/W")

with col4:
    if cooling_type == "Air Cooler":
        st.metric("🎯 Eficiência das Aletas", f"{result['heatsink_details']['eta_f']:.1%}")
    else: # Watercooler
        st.metric("💧 Vazão do Fluido", f"{result['flow_rate_lpm']:.2f} L/min")

# Adicionar mais métricas para watercooler
if cooling_type == "Watercooler":
    col_wc1, col_wc2, col_wc3 = st.columns(3)
    with col_wc1:
        st.metric("🌡️ Temp. Fluido (Média)", f"{result['T_fluid_avg']:.1f} °C")
    with col_wc2:
        st.metric("�� Queda de Pressão Total", f"{result['total_pressure_drop_bar']:.3f} bar")
    with col_wc3:
        st.metric("💦 h no Bloco da CPU", f"{result['h_cpu_block']:.1f} W/m²K")

# Status da temperatura (usando suas referências)
if result['T_junction'] <= 70:
    st.success(f"✅ Temperatura excelente! Tj = {result['T_junction']:.1f}°C")
elif result['T_junction'] <= 80:
    st.info(f"👍 Temperatura ótima para jogos/uso normal. Tj = {result['T_junction']:.1f}°C")
elif result['T_junction'] <= 85:
    st.warning(f"⚠️ Temperatura aceitável sob carga pesada. Tj = {result['T_junction']:.1f}°C")
elif result['T_junction'] <= 95:
    st.error(f"🚨 Temperatura alta! Tj = {result['T_junction']:.1f}°C - Risco de throttling em estresse.")
else:
    st.error(f"🔥 Temperatura crítica! Tj = {result['T_junction']:.1f}°C - Risco de danos!")

# Alertas de pressão para watercooler (usando suas referências)
if cooling_type == "Watercooler":
    pressure_bar = result['total_pressure_drop_bar']
    if pressure_bar > 0.7:
        st.warning(f"⚠️ Pressão do circuito ({pressure_bar:.2f} bar) acima do recomendado para tubos rígidos (>0.7 bar).")
    elif pressure_bar < 0.2 and pressure_bar > 0: # Evitar alerta para pressão zero se a vazão for zero
        st.info(f"💡 Pressão do circuito ({pressure_bar:.2f} bar) abaixo do recomendado para tubos rígidos (<0.2 bar), pode indicar baixa restrição.")

# === GRÁFICO DE RESISTÊNCIAS ===
st.subheader("🔍 Análise de Resistências Térmicas")

# Preparar dados
resistance_data = pd.DataFrame(result['R_breakdown'], columns=['Componente', 'Resistência (K/W)'])
resistance_data['Percentual'] = 100 * resistance_data['Resistência (K/W)'] / result['R_total']

# Criar gráfico
# Aumentar o tamanho da figura para dar mais espaço
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8)) 

# Definir uma paleta de cores consistente
# Adicione mais cores se houver mais componentes do que as listadas
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#a2d5c6', '#ffc107', '#8d6e63', '#7986cb']

# --- Gráfico de Barras ---
bars = ax1.barh(resistance_data['Componente'], resistance_data['Resistência (K/W)'],
                color=colors[:len(resistance_data)]) # Usar cores conforme o número de componentes
ax1.set_xlabel('Resistência Térmica (K/W)', fontsize=12)
ax1.set_title('Contribuição de cada camada', fontsize=14)
ax1.tick_params(axis='y', length=0) # Esconde os 'risquinhos' do tick do Y
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)


# Ajustar os rótulos dos valores nas barras de forma inteligente
max_resistance_val = resistance_data['Resistência (K/W)'].max()
for i, bar in enumerate(bars):
    width = bar.get_width()
    percentage = resistance_data.iloc[i]["Percentual"]
    label_text = f'{width:.3f}\n({percentage:.1f}%)'

    # Heurística: se a barra for razoavelmente grande, coloca o texto dentro
    # Caso contrário, coloca fora à direita com um pequeno offset
    if width > (max_resistance_val * 0.1): # 10% da maior barra
        ax1.text(width * 0.95, bar.get_y() + bar.get_height()/2,
                 label_text, ha='right', va='center', color='black', fontsize=9)
    else:
        ax1.text(width + (max_resistance_val * 0.02), bar.get_y() + bar.get_height()/2, # Offset proporcional
                 label_text, ha='left', va='center', color='black', fontsize=9)

# Ajustar o limite do eixo X para dar espaço para os rótulos externos das barras menores
ax1.set_xlim(0, max_resistance_val * 1.15) # 15% de espaço extra


# --- Gráfico de Rosca (Donut Chart) ---
# Função para formatar as porcentagens, mostrando apenas as maiores dentro do gráfico
# O limiar de 1.0% pode ser ajustado conforme necessário
def autopct_format(pct):
    return ('%1.1f%%' % pct) if pct > 1.0 else ''

wedges, texts, autotexts = ax2.pie(
    resistance_data['Percentual'],
    autopct=autopct_format,
    startangle=90,
    counterclock=False, # Para ir no sentido horário
    pctdistance=0.85, # Distância das porcentagens grandes do centro
    colors=colors[:len(resistance_data)],
    wedgeprops=dict(width=0.4, edgecolor='w'), # Largura do anel e borda branca para o donut
    textprops={'fontsize': 9, 'color': 'black'} # Cor padrão para os textos internos
)

ax2.set_title('Distribuição das Resistências', fontsize=14)
ax2.axis('equal') # Garante que o círculo seja desenhado corretamente

# Ajustar a cor do texto para o slice grande para ser visível
for autotext in autotexts:
    if autotext.get_text(): # Se houver texto (ou seja, pct > 1.0)
        autotext.set_color('black') # Cor para contrastar com o slice

# --- Lógica para labels externos e empilhados para as fatias pequenas ---
small_slices_info = []
for i, p in enumerate(resistance_data['Percentual']):
    if p <= 1.0: # Usar o mesmo limiar que autopct_format
        ang = (wedges[i].theta2 + wedges[i].theta1) / 2 # Ângulo central da fatia
        y_arrow = np.sin(np.deg2rad(ang))
        x_arrow = np.cos(np.deg2rad(ang))
        
        small_slices_info.append({
            'component': resistance_data["Componente"].iloc[i],
            'percentage': p,
            'x_arrow': x_arrow,
            'y_arrow': y_arrow,
            'angle': ang # Armazenar o ângulo para ordenação
        })

# Separar labels por lado (esquerda/direita) e ordenar para empilhamento
left_side_labels = [s for s in small_slices_info if s['x_arrow'] < 0]
right_side_labels = [s for s in small_slices_info if s['x_arrow'] >= 0]

# Ordenar cada lado por posição vertical (y_arrow) para empilhamento de cima para baixo
left_side_labels.sort(key=lambda x: x['y_arrow'], reverse=True)
right_side_labels.sort(key=lambda x: x['y_arrow'], reverse=True)

# Parâmetros de empilhamento
y_start_right = 0.9 # Posição Y inicial para labels do lado direito
y_start_left = 0.9 # Posição Y inicial para labels do lado esquerdo
y_offset_step = 0.12 # Espaçamento vertical entre labels empilhados

# Função para desenhar anotações empilhadas
def draw_stacked_annotations(labels_list, start_x_pos, start_y_pos, horizontal_alignment):
    current_y = start_y_pos
    for i, slice_data in enumerate(labels_list):
        comp_name = slice_data['component']
        pct_value = slice_data['percentage']
        x_arrow = slice_data['x_arrow']
        y_arrow = slice_data['y_arrow']

        # Posição do texto empilhado
        xytext_x = start_x_pos
        xytext_y = current_y

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="none", lw=0.72, alpha=0.8) # Fundo branco para o texto
        kw = dict(arrowprops=dict(arrowstyle="-", color='gray', connectionstyle="arc3,rad=0.2"),
                  bbox=bbox_props, zorder=0, va="center", ha=horizontal_alignment)

        ax2.annotate(f'{comp_name}\n({pct_value:.1f}%)',
                     xy=(x_arrow, y_arrow), # Ponto na borda da rosca
                     xytext=(xytext_x, xytext_y), # Ponto onde o texto será colocado
                     **kw, color='black', fontsize=8)
        
        current_y -= y_offset_step # Move para baixo para o próximo label

# Desenhar anotações para o lado direito
draw_stacked_annotations(right_side_labels, 1.3, y_start_right, "left") # x=1.3 é fora da rosca, à direita

# Desenhar anotações para o lado esquerdo
draw_stacked_annotations(left_side_labels, -1.3, y_start_left, "right") # x=-1.3 é fora da rosca, à esquerda


# --- Legenda ---
# Mover a legenda para o canto inferior direito
ax2.legend(wedges, resistance_data['Componente'],
           title="Componentes",
           loc="lower right", # Alterado para canto inferior direito
           fontsize=10,
           title_fontsize=12)

# Ajustar layout para evitar sobreposição
# plt.tight_layout() tentará ajustar tudo.
# Se os labels ainda estiverem cortados, pode ser necessário aumentar o figsize
# ou ajustar o parâmetro rect em tight_layout.
plt.tight_layout()

st.pyplot(fig)

# === DETALHES DO DISSIPADOR ===
if cooling_type == "Air Cooler":
    st.subheader("🔎 Detalhes do Dissipador")

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
else:
    st.subheader("💧 Detalhes do Watercooler")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fluido", fluid_type)
        st.metric("Bomba RPM", f"{watercooler_params['pump_params']['rpm']} RPM")
    with col2:
        st.metric("Radiador", radiator_preset_name)
        st.metric("Fans Radiador RPM", f"{watercooler_params['radiator_params']['fan_rpm']} RPM")
    with col3:
        st.metric("Bloco CPU Material", cpu_block_material)
        st.metric("Microcanais", f"{num_microchannels}x {microchannel_width*1e3:.1f}x{microchannel_height*1e3:.1f}mm")

# === INFORMAÇÕES TÉCNICAS ===
with st.expander("📋 Informações Técnicas"):
    st.write("**Parâmetros do Die:**")
    st.write(f"- Área: {die_area*1e6:.1f} mm²")
    st.write(f"- Espessura: {die_thickness*1e3:.1f} mm")
    st.write(f"- Condutividade: {die_k} W/m·K")
    st.write(f"- Geração volumétrica: {result['q_dot']/1e6:.1f} MW/m³")

    st.write("**Configuração da Pilha:**")
    for layer in layers_stack:
        st.write(f"- {layer['name']}: {layer['thickness']*1e6:.0f} μm, k = {layer['k']} W/m·K")

    if cooling_type == "Watercooler":
        st.write("**Parâmetros do Fluido:**")
        st.write(f"- Tipo: {fluid_type}")
        st.write(f"- Densidade: {fluid_props['density']:.0f} kg/m³")
        st.write(f"- Viscosidade: {fluid_props['viscosity']:.1e} Pa·s")
        st.write(f"- Calor Específico: {fluid_props['specific_heat']:.0f} J/kg·K")

# === PASSO-A-PASSO DOS CÁLCULOS ===
with st.expander("🧾 Cálculos (Passo a passo)"):
    trace = result.get('trace', None)
    # opção para forçar fundo escuro no bloco de trace (útil se o tema do Streamlit não propagar)
    force_dark_trace = st.checkbox('Forçar fundo escuro para o passo-a-passo', value=False)

    if trace:
        # Renderizar trace como tabela estilizada para melhor leitura
        # Se o usuário escolher forçar fundo escuro, aplicamos estilo inline para garantir contraste
        if force_dark_trace:
            div_style = "background:#071328;color:#dbeefc;padding:8px;border-radius:8px;"
            row_even = "background: rgba(255,255,255,0.01);"
            row_odd = "background: transparent;"
        else:
            div_style = ""
            row_even = "background:#fbfbfb;"
            row_odd = "background:#ffffff;"

        html = """
        <style>
        /* Caixa do trace: estilo neutro com detecção do tema do Streamlit via html[data-theme] */
        .trace-table { width:100%; border-collapse:collapse; font-family: monospace; font-size:13px; }
        .trace-table th { text-align:left; padding:8px 6px; border-bottom:1px solid rgba(0,0,0,0.08); }
        .trace-table td { padding:8px 6px; border-bottom:1px solid rgba(0,0,0,0.04); vertical-align:top; }
        .trace-box { max-height:360px; overflow:auto; padding:8px; border-radius:8px; }

        /* Tema claro (streamlit data-theme='light' ou padrão) */
        html[data-theme='light'] .trace-box,
        html:not([data-theme]) .trace-box { background: #ffffff; color:#0b1220; box-shadow: 0 1px 4px rgba(16,24,40,0.06); }
        html[data-theme='light'] .trace-table tr:nth-child(even), html:not([data-theme]) .trace-table tr:nth-child(even) { background:#fbfbfb; }

        /* Tema escuro (streamlit data-theme='dark') */
        html[data-theme='dark'] .trace-box { background: #071328; color: #dbeefc; box-shadow: none; border: 1px solid rgba(255,255,255,0.04); }
        html[data-theme='dark'] .trace-table th { border-bottom-color: rgba(255,255,255,0.06); }
        html[data-theme='dark'] .trace-table td { border-bottom-color: rgba(255,255,255,0.03); }
        html[data-theme='dark'] .trace-table tr:nth-child(even) { background: rgba(255,255,255,0.01); }
        </style>
        <div class='trace-box'>
          <table class='trace-table'>
            <thead><tr><th style='width:56px'>Passo</th><th>Descrição</th></tr></thead>
            <tbody>
        """

        for i, line in enumerate(trace):
            # escapar tags HTML somente por segurança
            safe_line = str(line).replace("<", "&lt;").replace(">", "&gt;")
            # escolher cor da linha conforme preferência do usuário
            row_bg = row_odd if i % 2 == 0 else row_even
            html += f"<tr style='{row_bg}'><td><strong>{i+1}</strong></td><td>{safe_line}</td></tr>"

        html += "</tbody></table></div>"

        # injetar o estilo e, se necessário, aplicar o estilo inline ao div
        if force_dark_trace:
            # substituir a abertura da div por uma com style inline (garante prioridade)
            html = html.replace("<div class='trace-box'>", f"<div class='trace-box' style='{div_style}'>")
        st.markdown(html, unsafe_allow_html=True)

        # Fornecer opção para baixar o trace bruto e ver raw
        raw_text = "\n".join(trace)
        st.download_button("📥 Baixar trace (texto)", data=raw_text, file_name="calculos_trace.txt", mime="text/plain")
        with st.expander('Ver raw (copiar)'):
            st.code(raw_text, language='text')
    else:
        st.write("Nenhum detalhe passo-a-passo disponível.")

# === SIMULAÇÃO 2D (PROTÓTIPO) ===
# Esta seção só deve aparecer para Air Cooler
if cooling_type == "Air Cooler":
    st.header("🧪 Simulação 2D - Condução (Protótipo)")
    with st.expander("Configurar simulação 2D"):
        sim_nx = st.slider('Resolução X (nx)', 40, 240, 120)
        sim_ny = st.slider('Resolução Y (ny)', 40, 240, 120)
        sim_Lx = st.number_input('Largura do domínio (m)', value=0.06)
        sim_Ly = st.number_input('Altura do domínio (m)', value=0.06)
        # fin_height já está definida globalmente ou pelo slider do Air Cooler
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
else:
    st.info("A simulação 2D de condução está disponível apenas para configurações de Air Cooler.")


# === VISTA LATERAL (SEÇÃO TRANSVERSAL) ===
# Esta seção só deve aparecer para Air Cooler
if cooling_type == "Air Cooler":
    st.header("🔎 Vista Lateral - Seção Transversal (simplificada)")
    with st.expander("Configurar vista lateral e modelo de convecção"):
        side_nx = st.slider('Resolução Lateral (nx)', 40, 240, 120)
        side_ny = st.slider('Resolução Vertical (ny)', 80, 320, 160)
        # base_size e fin_height já estão definidas globalmente ou pelo slider do Air Cooler
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
else:
    st.info("A simulação de vista lateral está disponível apenas para configurações de Air Cooler.")

# === COMPARAÇÃO RÁPIDA ===
st.subheader("⚖️ Comparação Rápida")
if st.button("🔄 Comparar com Configuração Básica"):
    # Configuração básica de referência (Air Cooler)
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

    basic_result = calculate_thermal_performance(
        "Air Cooler", power, T_ambient, die_area, die_thickness,
        die_k, basic_layers, heatsink_params=basic_heatsink
    )

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
