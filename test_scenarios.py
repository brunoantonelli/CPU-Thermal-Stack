"""
Testes de Cenários Comparativos
"""

import sys
sys.path.append('.')

from thermal_core import calculate_cpu_temperatures
from materials import *

def compare_tim_materials():
    """Compara diferentes materiais de interface térmica"""
    
    print("🔬 Comparando Materiais TIM (Interface Térmica)")
    print("=" * 60)
    
    # Configuração base
    power = 95
    T_ambient = 25
    die_area = 12e-3 * 12e-3
    die_thickness = 0.5e-3
    die_k = 120
    
    # Dissipador fixo
    heatsink_params = {
        'h': 80,
        'k_fin': 200,
        'fin_thickness': 1e-3,
        'fin_width': 40e-3,
        'fin_height': 30e-3,
        'n_fins': 20,
        'base_length': 40e-3,
        'base_width': 40e-3
    }
    
    # Testar diferentes TIMs
    tims_to_test = [
        'Pasta térmica básica',
        'Pasta térmica boa', 
        'Pasta térmica premium',
        'Metal líquido (Ga-In)'
    ]
    
    results = []
    
    for tim_name in tims_to_test:
        tim_k = MATERIALS['tim'][tim_name]
        
        layers = [
            {'name': 'TIM', 'thickness': 100e-6, 'k': tim_k, 'area': 40e-3 * 40e-3},
            {'name': 'Spreader', 'thickness': 2e-3, 'k': 390, 'area': 40e-3 * 40e-3},
            {'name': 'Base', 'thickness': 3e-3, 'k': 200, 'area': 40e-3 * 40e-3}
        ]
        
        result = calculate_cpu_temperatures(power, T_ambient, die_area, die_thickness, 
                                           die_k, layers, heatsink_params)
        
        results.append({
            'tim': tim_name,
            'k': tim_k,
            'T_junction': result['T_junction'],
            'R_total': result['R_total']
        })
    
    # Mostrar resultados
    print(f"{'Material TIM':<25} {'k (W/m·K)':<12} {'Tj (°C)':<10} {'R_total (K/W)':<15}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['tim']:<25} {r['k']:<12.1f} {r['T_junction']:<10.1f} {r['R_total']:<15.3f}")
    
    # Análise
    best = min(results, key=lambda x: x['T_junction'])
    worst = max(results, key=lambda x: x['T_junction'])
    
    print(f"\n📊 Análise:")
    print(f"   Melhor: {best['tim']} → {best['T_junction']:.1f}°C")
    print(f"   Pior: {worst['tim']} → {worst['T_junction']:.1f}°C")
    print(f"   Diferença: {worst['T_junction'] - best['T_junction']:.1f}°C")
    
    return results

def compare_cooling_methods():
    """Compara diferentes métodos de resfriamento"""
    
    print("\n\n❄️ Comparando Métodos de Resfriamento")
    print("=" * 60)
    
    # Configuração base
    power = 125  # CPU mais potente
    T_ambient = 25
    die_area = 15e-3 * 15e-3
    die_thickness = 0.7e-3
    die_k = 120
    
    # Camadas fixas
    layers = [
        {'name': 'TIM', 'thickness': 100e-6, 'k': 8.5, 'area': 40e-3 * 40e-3},
        {'name': 'Spreader', 'thickness': 2e-3, 'k': 390, 'area': 40e-3 * 40e-3},
        {'name': 'Base', 'thickness': 3e-3, 'k': 200, 'area': 40e-3 * 40e-3}
    ]
    
    # Diferentes métodos de resfriamento
    cooling_methods = [
        ('Convecção natural', 8, 15e-3, 15),
        ('Ventilador baixo', 45, 30e-3, 20),
        ('Ventilador médio', 80, 40e-3, 25),
        ('Ventilador alto', 120, 50e-3, 30),
        ('Refrigeração líquida', 200, 60e-3, 35)
    ]
    
    results = []
    
    for method_name, h, fin_height, n_fins in cooling_methods:
        heatsink_params = {
            'h': h,
            'k_fin': 200,
            'fin_thickness': 1e-3,
            'fin_width': 50e-3,
            'fin_height': fin_height,
            'n_fins': n_fins,
            'base_length': 50e-3,
            'base_width': 50e-3
        }
        
        result = calculate_cpu_temperatures(power, T_ambient, die_area, die_thickness, 
                                           die_k, layers, heatsink_params)
        
        results.append({
            'method': method_name,
            'h': h,
            'T_junction': result['T_junction'],
            'eta_f': result['heatsink_details']['eta_f']
        })
    
    # Mostrar resultados
    print(f"{'Método':<20} {'h (W/m²·K)':<12} {'Tj (°C)':<10} {'η_aletas':<10}")
    print("-" * 55)
    
    for r in results:
        print(f"{r['method']:<20} {r['h']:<12} {r['T_junction']:<10.1f} {r['eta_f']:<10.2f}")
    
    # Análise
    print(f"\n📊 Análise:")
    for i, r in enumerate(results[1:], 1):
        prev = results[i-1]
        temp_reduction = prev['T_junction'] - r['T_junction']
        print(f"   {prev['method']} → {r['method']}: -{temp_reduction:.1f}°C")
    
    return results

def analyze_fin_geometry():
    """Analisa impacto da geometria das aletas"""
    
    print("\n\n🔧 Análise de Geometria das Aletas")
    print("=" * 60)
    
    # Configuração base
    power = 95
    T_ambient = 25
    die_area = 12e-3 * 12e-3
    die_thickness = 0.5e-3
    die_k = 120
    
    layers = [
        {'name': 'TIM', 'thickness': 100e-6, 'k': 5.0, 'area': 40e-3 * 40e-3},
        {'name': 'Spreader', 'thickness': 2e-3, 'k': 390, 'area': 40e-3 * 40e-3},
        {'name': 'Base', 'thickness': 3e-3, 'k': 200, 'area': 40e-3 * 40e-3}
    ]
    
    # Diferentes configurações de aletas
    fin_configs = [
        ("Poucas/Grossas", 10, 2e-3, 40e-3),
        ("Médias", 20, 1e-3, 30e-3), 
        ("Muitas/Finas", 40, 0.5e-3, 25e-3),
        ("Muito Altas", 15, 1e-3, 60e-3)
    ]
    
    results = []
    
    for config_name, n_fins, thickness, height in fin_configs:
        heatsink_params = {
            'h': 80,
            'k_fin': 200,
            'fin_thickness': thickness,
            'fin_width': 40e-3,
            'fin_height': height,
            'n_fins': n_fins,
            'base_length': 40e-3,
            'base_width': 40e-3
        }
        
        result = calculate_cpu_temperatures(power, T_ambient, die_area, die_thickness, 
                                           die_k, layers, heatsink_params)
        
        results.append({
            'config': config_name,
            'n_fins': n_fins,
            'thickness_mm': thickness * 1000,
            'height_mm': height * 1000,
            'T_junction': result['T_junction'],
            'eta_f': result['heatsink_details']['eta_f'],
            'A_eff': result['heatsink_details']['A_eff'] * 1e6  # mm²
        })
    
    # Mostrar resultados
    print(f"{'Configuração':<15} {'N':<3} {'t(mm)':<7} {'h(mm)':<7} {'Tj(°C)':<8} {'η':<6} {'A_eff(mm²)':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['config']:<15} {r['n_fins']:<3} {r['thickness_mm']:<7.1f} {r['height_mm']:<7.1f} "
              f"{r['T_junction']:<8.1f} {r['eta_f']:<6.2f} {r['A_eff']:<12.0f}")
    
    return results

if __name__ == "__main__":
    # Executar todos os testes
    compare_tim_materials()
    compare_cooling_methods() 
    analyze_fin_geometry()
    
    print("\n\n🎯 Conclusões Gerais:")
    print("1. TIM premium reduz temperatura, mas o ganho é limitado")
    print("2. Ventilação é o fator mais crítico")
    print("3. Geometria das aletas deve balancear área vs eficiência")
    print("4. Aletas muito finas podem ter eficiência baixa")