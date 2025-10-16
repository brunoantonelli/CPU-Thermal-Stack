"""
Biblioteca de Materiais e Presets
CPU Thermal Stack Designer
"""

# Condutividades térmicas típicas (W/m·K)
MATERIALS = {
    'die': {
        'Silício': 120,
        'GaN (Gallium Nitride)': 130,
        'SiC (Silicon Carbide)': 200,
        'GaAs': 55
    },
    
    'tim': {  # Thermal Interface Materials
        'Pasta térmica básica': 3.0,
        'Pasta térmica boa': 5.0,
        'Pasta térmica premium': 8.5,
        'Pasta térmica líquida': 12.0,
        'Metal líquido (Ga-In)': 25.0,
        'Solda térmica': 50.0,
        'Grafite pirolítico': 1500,
        'Pad térmico básico': 1.5,
        'Pad térmico premium': 6.0
    },
    
    'spreader': {
        'Cobre': 390,
        'Cobre niquelado': 350,
        'Alumínio': 200,
        'Prata': 420,
        'Grafite sintético': 1000
    },
    
    'heatsink': {
        'Alumínio': 200,
        'Alumínio anodizado': 180,
        'Cobre': 390,
        'Liga de alumínio 6061': 167,
        'Liga de alumínio 1050': 229
    }
}

# Coeficientes de convecção típicos (W/m²·K)
CONVECTION = {
    'Convecção natural': 8,
    'Ventilador muito baixo (600 RPM)': 25,
    'Ventilador baixo (800 RPM)': 45,
    'Ventilador médio (1200 RPM)': 80,
    'Ventilador alto (1800 RPM)': 120,
    'Ventilador muito alto (2500 RPM)': 180,
    'Refrigeração líquida (AIO)': 200,
    'Refrigeração líquida custom': 300
}

# Presets de CPU comuns
CPU_PRESETS = {
    'CPU Básica (65W)': {
        'power': 65,
        'die_area': 10e-3 * 10e-3,  # 10x10 mm
        'die_thickness': 0.4e-3,    # 0.4 mm
        'description': 'CPU de escritório, baixo consumo'
    },
    
    'CPU Desktop (95W)': {
        'power': 95,
        'die_area': 12e-3 * 12e-3,  # 12x12 mm
        'die_thickness': 0.5e-3,    # 0.5 mm
        'description': 'CPU mainstream para jogos'
    },
    
    'CPU High-end (125W)': {
        'power': 125,
        'die_area': 15e-3 * 15e-3,  # 15x15 mm
        'die_thickness': 0.7e-3,    # 0.7 mm
        'description': 'CPU enthusiast/workstation'
    },
    
    'CPU Server (200W)': {
        'power': 200,
        'die_area': 20e-3 * 20e-3,  # 20x20 mm
        'die_thickness': 1.0e-3,    # 1.0 mm
        'description': 'CPU servidor/HEDT'
    },
    
    'GPU Mobile (50W)': {
        'power': 50,
        'die_area': 8e-3 * 8e-3,    # 8x8 mm
        'die_thickness': 0.3e-3,    # 0.3 mm
        'description': 'GPU integrada/mobile'
    }
}

# Configurações típicas de dissipadores
HEATSINK_PRESETS = {
    'Stock Cooler': {
        'fin_height': 25e-3,        # 25 mm
        'n_fins': 15,
        'fin_thickness': 1.2e-3,    # 1.2 mm
        'base_size': 35e-3,         # 35x35 mm
        'description': 'Dissipador que vem com a CPU'
    },
    
    'Tower Cooler Básico': {
        'fin_height': 40e-3,        # 40 mm
        'n_fins': 25,
        'fin_thickness': 1.0e-3,    # 1.0 mm
        'base_size': 40e-3,         # 40x40 mm
        'description': 'Dissipador torre entrada'
    },
    
    'Tower Cooler Premium': {
        'fin_height': 60e-3,        # 60 mm
        'n_fins': 35,
        'fin_thickness': 0.8e-3,    # 0.8 mm
        'base_size': 50e-3,         # 50x50 mm
        'description': 'Dissipador torre high-end'
    },
    
    'Low Profile': {
        'fin_height': 15e-3,        # 15 mm
        'n_fins': 20,
        'fin_thickness': 0.6e-3,    # 0.6 mm
        'base_size': 45e-3,         # 45x45 mm
        'description': 'Para gabinetes compactos'
    }
}

# Espessuras típicas (metros)
TYPICAL_THICKNESS = {
    'tim_thin': 50e-6,      # 50 μm - aplicação fina
    'tim_normal': 100e-6,   # 100 μm - aplicação normal
    'tim_thick': 200e-6,    # 200 μm - aplicação grossa
    'spreader_thin': 1e-3,  # 1 mm
    'spreader_normal': 2e-3, # 2 mm
    'spreader_thick': 3e-3, # 3 mm
    'base_normal': 3e-3,    # 3 mm
    'base_thick': 5e-3      # 5 mm
}

def get_material_info(category, material_name):
    """Retorna informações sobre um material"""
    if category in MATERIALS and material_name in MATERIALS[category]:
        return {
            'name': material_name,
            'k': MATERIALS[category][material_name],
            'category': category
        }
    return None

def list_materials(category=None):
    """Lista materiais disponíveis"""
    if category:
        return list(MATERIALS.get(category, {}).keys())
    else:
        all_materials = {}
        for cat, materials in MATERIALS.items():
            all_materials[cat] = list(materials.keys())
        return all_materials

def get_preset_config(preset_name, config_type='cpu'):
    """Retorna configuração de preset"""
    if config_type == 'cpu' and preset_name in CPU_PRESETS:
        return CPU_PRESETS[preset_name]
    elif config_type == 'heatsink' and preset_name in HEATSINK_PRESETS:
        return HEATSINK_PRESETS[preset_name]
    return None

# Teste da biblioteca
if __name__ == "__main__":
    print("📚 Testando biblioteca de materiais...")
    
    print("\n🔧 Materiais TIM disponíveis:")
    for tim in MATERIALS['tim']:
        k = MATERIALS['tim'][tim]
        print(f"   {tim}: {k} W/m·K")
    
    print("\n🌡️ Opções de convecção:")
    for conv in CONVECTION:
        h = CONVECTION[conv]
        print(f"   {conv}: {h} W/m²·K")
    
    print("\n💻 Presets de CPU:")
    for cpu in CPU_PRESETS:
        preset = CPU_PRESETS[cpu]
        print(f"   {cpu}: {preset['power']}W - {preset['description']}")