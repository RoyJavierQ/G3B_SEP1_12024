# G3B_SEP1_12024
TAREA 1
Nombres: Kevin Tarqui - Roy Quinchaleo


# 3a)
import pandapower as pp  
import matplotlib.pyplot as plt  
import numpy as np  

# red
net = pp.create_empty_network(name="empty")  

# barras
b1 = pp.create_bus(net, vn_kv=500. , name="bus 1")  
b2 = pp.create_bus(net, vn_kv=500. , name="bus 2")  

# generador
pp.create_ext_grid(net, b1, vm_pu=1., name="external grid")  

# linea
test_type = {"r_ohm_per_km": 0.02, "x_ohm_per_km": 0.115, "c_nf_per_km": 19.1, "max_i_ka": 1, "type": "cs"}  
pp.create_std_type(net, name="test_type", data=test_type, element="line")  
pp.available_std_types(net, element="line")  

line1 = pp.create_line(net, b1, b2, std_type="test_type", length_km=500., name="line1")  
line2 = pp.create_line(net, b1, b2, std_type="test_type", length_km=500., name="line2")  

# Potencia Carga

s_nominal= 1200 #MVA  
p_nominal= 1080 #MW  
q_nominal=523.07 #MVAR  

# Carga

pp.create_load(net, bus=b2, p_mw=p_nominal, q_mvar=q_nominal, name="load")  

# Variación

p_nuevo= np.linspace(0.5 * p_nominal, 1.5 * p_nominal, 100)  
q_nuevo= q_nominal * (p_nuevo / p_nominal)  

# tensiones

tensiones= []  

# Ciclo for

for p, q in zip(p_nuevo, q_nuevo):
    
    net.load.at[0, 'p_mw'] = p
    net.load.at[0, 'q_mvar'] = q
    # Ejecutar simulación 
    pp.runpp(net)
    # se agrega tensión
    tensiones.append(net.res_bus.vm_pu[b2])

# Grafico

plt.figure(figsize=(10, 6))  
plt.plot(p_nuevo, tensiones, label='Tensión en la barra')  
plt.xlabel('Carga (MW)')  
plt.ylabel('Tensión (p.u.)')  
plt.title('Comportamiento de la Tensión para Variación de Carga')  
plt.legend()  
plt.grid(True)  
plt.show()  

# 3b)
# tensiones

tensiones_sin = []  
tensiones_con = []  

# Limites de tensiones

#Límites de tensión según normativa (ejemplo: 0.95 y 1.05 p.u.)  
tension_minima = 0.97  
tension_maxima = 1.03  

for p, q in zip(p_nuevo, q_nuevo):
    
    net.load.at[0, 'p_mw'] = p
    net.load.at[0, 'q_mvar'] = q
    
    pp.runpp(net)
    tensiones_sin.append(net.res_bus.vm_pu[b2])
    
    if net.res_bus.vm_pu[b2] < tension_minima:
        q_shunt = (tension_minima - net.res_bus.vm_pu[b2]) * p_nominal  
    elif net.res_bus.vm_pu[b2] > tension_maxima:
        q_shunt = (tension_maxima - net.res_bus.vm_pu[b2]) * p_nominal  
    else:
        q_shunt = 0
    
    if q_shunt != 0:
        pp.create_sgen(net, bus=b2, p_mw=0, q_mvar=q_shunt, name="shunt compensation")
    
    pp.runpp(net)
    tensiones_con.append(net.res_bus.vm_pu[b2])
    
    if q_shunt != 0:
        net.sgen.drop(net.sgen.index[-1], inplace=True)

# Graficar los resultados
plt.figure(figsize=(10, 6))  
plt.plot(p_nuevo, tensiones_sin, label='Tensión sin compensación')  
plt.plot(p_nuevo, tensiones_con, label='Tensión con compensación', linestyle='--')  
plt.axhline(y=tension_minima, color='r', linestyle=':', label='Límite mínimo')  
plt.axhline(y=tension_maxima, color='r', linestyle=':', label='Límite máximo')  
plt.xlabel('Carga (MW)')  
plt.ylabel('Tensión (p.u.)')  
plt.title('Comportamiento de la Tensión con y sin Compensación Shunt')  
plt.legend()  
plt.grid(True)  
plt.show()  

# 3c)

# Almacenar nuevos resultados
perdidas_sin = []  
perdidas_con = []  

# Añadir shunt al sistema
shunt = pp.create_sgen(net, bus=b2, p_mw=0, q_mvar=0, name="shunt")  

for p, q in zip(p_nuevo, q_nuevo):

    #Actualizar la carga 
    net.load.at[0, 'p_mw'] = p
    net.load.at[0, 'q_mvar'] = q
    
    pp.runpp(net)
    perdidas_sin.append(net.res_line.pl_mw.sum())
    
    #Calcular compensación 
    if net.res_bus.vm_pu[b2] < 0.97:
        q_shunt = (0.97 - net.res_bus.vm_pu[b2]) * (p / net.res_bus.vm_pu[b2])
    elif net.res_bus.vm_pu[b2] > 1.03:
        q_shunt = (1.03 - net.res_bus.vm_pu[b2]) * (p / net.res_bus.vm_pu[b2])
    else:
        q_shunt = 0
    
    #Aplicar compensación
    net.sgen.at[shunt, 'q_mvar'] = q_shunt
    
    #Simulación
    pp.runpp(net)
    perdidas_con.append(net.res_line.pl_mw.sum())

# Graficar los resultados
plt.figure(figsize=(10, 6))  
plt.plot(p_nuevo, perdidas_sin, label='Pérdidas sin compensación')  
plt.plot(p_nuevo, perdidas_con, label='Pérdidas con shunt', linestyle='--')  
plt.xlabel('Carga (MW)')  
plt.ylabel('Pérdidas en la línea (MW)')  
plt.title('Pérdidas en la línea con y sin Compensación shunt')  
plt.legend()  
plt.grid(True)  
plt.show()  

#-----------------------------------------------------------------

import pandapower as pp  
import matplotlib.pyplot as plt  
import pandapower.plotting as plot  

# Crear la red
net = pp.create_empty_network(name="empty")  

# Crear barras
b1 = pp.create_bus(net, vn_kv=110.0, name="bus 1")  
b2 = pp.create_bus(net, vn_kv=220.0, name="bus 2")  
b3 = pp.create_bus(net, vn_kv=220.0, name="bus 1a")  
b4 = pp.create_bus(net, vn_kv=220.0, name="bus 2a")  
b5 = pp.create_bus(net, vn_kv=220.0, name="bus 3a")  
b6 = pp.create_bus(net, vn_kv=220.0, name="bus 2b")  
b7 = pp.create_bus(net, vn_kv=220.0, name="bus 1b")  

# Crear generador externo
pp.create_ext_grid(net, b1, vm_pu=1.0, name="external grid")  

# Crear transformador
trafo1 = pp.create_transformer(net, b2, b1, name="110kV/220kV transformer", std_type="100 MVA 220/110 kV")  

# Crear líneas
line1 = pp.create_line(net, b2, b3, length_km=10, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 1")  
line2 = pp.create_line(net, b3, b4, length_km=15, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 2")  
line3 = pp.create_line(net, b4, b5, length_km=20, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 3")  
line4 = pp.create_line(net, b5, b6, length_km=15, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 4")  
line5 = pp.create_line(net, b6, b7, length_km=30, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 5")  
line6 = pp.create_line(net, b7, b2, length_km=10, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 6")  

# Crear cargas
loads = [
    pp.create_load(net, b3, p_mw=30, q_mvar=20, name="load 1a"),
    pp.create_load(net, b4, p_mw=52.5, q_mvar=35, name="load 2a"),
    pp.create_load(net, b5, p_mw=22.5, q_mvar=15, name="load 3a"),
    pp.create_load(net, b6, p_mw=90, q_mvar=60, name="load 2b"),
    pp.create_load(net, b7, p_mw=15, q_mvar=10, name="load 1b")
]  

# Crear switches
sw1 = pp.create_switch(net, b5, line4, et="l", type="LBS", closed=True)  
sw2 = pp.create_switch(net, b6, line4, et="l", type="LBS", closed=True)  

# Casos
casos = {
    'estado_normal': {'v_limit_sup': 1.05, 'v_limit_inf': 0.95},
    'estado_alerta': {'v_limit_sup': 1.07, 'v_limit_inf': 0.93}
}  

# Ejecutar flujo de potencia
pp.runpp(net)  

# Verificar tensiones
barras_fuera_norma = net.res_bus[(net.res_bus.vm_pu > casos['estado_normal']['v_limit_sup']) | (net.res_bus.vm_pu < casos['estado_normal']['v_limit_inf'])]  
lineas_saturadas = net.res_line[net.res_line.loading_percent > 100]  

# Resultados
print("Condiciones iniciales:")  
if not barras_fuera_norma.empty:
    print("Barras fuera de norma:")
    print(barras_fuera_norma)
else:
    print("No hay buses fuera de norma.")
    
if not lineas_saturadas.empty:
    print("Líneas saturadas:")
    print(lineas_saturadas)
else:
    print("No hay líneas saturadas.")

# Evaluación de las cargas de las líneas
print("\nCargas de las líneas en condiciones iniciales:")  
print(net.res_line)  

# Crear gráfico
plt.figure(figsize=(12, 6))  
plt.bar(net.line.index, net.res_line.loading_percent, color='b', alpha=0.7)  
plt.axhline(y=100, color='r', linestyle='--', label='Límite de carga (100%)')  
plt.xlabel('Índice de la Línea')  
plt.ylabel('Carga de la Línea (%)')  
plt.title('Estado de Carga de Cada Línea')  
plt.xticks(net.line.index, net.line.name, rotation=45)  
plt.legend()  
plt.tight_layout()  
plt.show()  

# Crear gráfico de tensiones para cada caso
resultados_tensiones = {}  

for caso, limites in casos.items():
    pp.runpp(net)
    resultados_tensiones[caso] = net.res_bus.vm_pu.copy()

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = range(len(net.bus))

for i, (caso, voltajes) in enumerate(resultados_tensiones.items()):
    ax.bar([p + i * bar_width for p in index], voltajes, bar_width, label=f'Tensiones ({caso})')

ax.set_xlabel('Buses')  
ax.set_ylabel('Voltaje (pu)')  
ax.set_title('Tensiones para cada caso')  
ax.set_xticks([p + bar_width for p in index])  
ax.set_xticklabels(net.bus.name)  
ax.axhline(y=casos['estado_normal']['v_limit_sup'], color='b', linestyle='--', label='Límite Sup. Estado Normal')  
ax.axhline(y=casos['estado_normal']['v_limit_inf'], color='b', linestyle='--', label='Límite Inf. Estado Normal')  
ax.axhline(y=casos['estado_alerta']['v_limit_sup'], color='r', linestyle=':', label='Límite Sup. Estado Alerta')  
ax.axhline(y=casos['estado_alerta']['v_limit_inf'], color='r', linestyle=':', label='Límite Inf. Estado Alerta')  
ax.legend()  

# Zoom
limite_inferior_zoom = min(casos['estado_alerta']['v_limit_inf'], casos['estado_normal']['v_limit_inf']) - 0.02  
limite_superior_zoom = max(casos['estado_alerta']['v_limit_sup'], casos['estado_normal']['v_limit_sup']) + 0.02  
ax.set_ylim([limite_inferior_zoom, limite_superior_zoom])  

plt.tight_layout()  
plt.show()  

# Verificar tensiones
for caso, limites in casos.items():
    pp.runpp(net)
    
    # Verificar tensiones
    barras_fuera_norma = net.res_bus[(net.res_bus.vm_pu > limites['v_limit_sup']) | (net.res_bus.vm_pu < limites['v_limit_inf'])]
    
    # Verificar líneas saturadas
    lineas_saturadas = net.res_line[net.res_line.loading_percent > 100]
    
    print(f"\nResultados para {caso}:")
    
    if not barras_fuera_norma.empty:
        print("Barras fuera de norma:")
        print(barras_fuera_norma)
    else:
        print("No hay buses fuera de norma.")
    
    if not lineas_saturadas.empty:
        print("Líneas saturadas:")
        print(lineas_saturadas)
    else:
        print("No hay líneas saturadas.")

# Estado normal
v_limit_sup = 1.05  
v_limit_inf = 0.95  

# Desconectar switches
net.switch.loc[sw1, 'closed'] = False  
net.switch.loc[sw2, 'closed'] = False  

# flujo de potencia
pp.runpp(net)  

# Verificar tensiones y líneas
barras_fuera_norma_post = net.res_bus[(net.res_bus.vm_pu > v_limit_sup) | (net.res_bus.vm_pu < v_limit_inf)]  
lineas_saturadas_post = net.res_line[net.res_line.loading_percent > 100]  

# Resultados después de desconectar switches
print("\nCondiciones después de desconectar switches:")  
if not barras_fuera_norma_post.empty:
    print("Barras fuera de norma:")
    print(barras_fuera_norma_post)
else:
    print("No hay barras fuera de norma.")
    
if not lineas_saturadas_post.empty:
    print("Líneas saturadas:")
    print(lineas_saturadas_post)
else:
    print("No hay líneas saturadas.")

# Evaluación de las cargas
print("\nCargas de las líneas después de desconectar switches:")  
print(net.res_line)  

# Crear gráfico de las cargas
plt.figure(figsize=(12, 6))  
plt.bar(net.line.index, net.res_line.loading_percent, color='b', alpha=0.7)  
plt.axhline(y=100, color='r', linestyle='--', label='Límite de carga (100%)')  
plt.xlabel('Índice de la Línea')  
plt.ylabel('Carga de la Línea (%)')  
plt.title('Estado de Carga de Cada Línea después de desconectar switches')  
plt.xticks(net.line.index, net.line.name, rotation=45)  
plt.legend()  
plt.tight_layout()  
plt.show()  

# Crear gráfico de tensiones
fig, ax = plt.subplots(figsize=(12, 6))  
bar_width = 0.35  
index = range(len(net.bus))  

ax.bar(index, net.res_bus.vm_pu, bar_width, label='Tensiones(después de desconexión)')  

ax.set_xlabel('Barras')  
ax.set_ylabel('Voltaje (pu)')  
ax.set_title('Tensiones para cada caso después de desconectar switches')  
ax.set_xticks(index)  
ax.set_xticklabels(net.bus.name)  
ax.axhline(y=v_limit_sup, color='b', linestyle='--', label='Límite Sup. Estado Normal')  
ax.axhline(y=v_limit_inf, color='b', linestyle='--', label='Límite Inf. Estado Normal')  
ax.legend()

# Zoom
limite_inferior_zoom = v_limit_inf - 0.02  
limite_superior_zoom = v_limit_sup + 0.02  
ax.set_ylim([limite_inferior_zoom, limite_superior_zoom])  

plt.tight_layout()  
plt.show()  

# Reconexión switches
net.switch.loc[sw1, 'closed'] = True  
net.switch.loc[sw2, 'closed'] = True  


#aumento de reactivos

# Calcular el 20% de los reactivos generados por las cargas
total_reactivos = net.load.q_mvar.sum()  
generador_q_mvar = 0.2 * total_reactivos  

# Crear un generador que supla los reactivos adicionales
pp.create_gen(net, bus=b1, p_mw=0, vm_pu=1.0, name="Gen Reactivos", max_q_mvar=generador_q_mvar, min_q_mvar=-generador_q_mvar)  

# Ejecutar flujo de potencia después del aumento de reactivos
pp.runpp(net)  

# Verificar tensiones y líneas después del aumento de reactivos
barras_fuera_norma_reactivos = net.res_bus[(net.res_bus.vm_pu > v_limit_sup) | (net.res_bus.vm_pu < v_limit_inf)]  
lineas_saturadas_reactivos = net.res_line[net.res_line.loading_percent > 100]  

# Resultados después del aumento de reactivos
print("\nCondiciones después del aumento de reactivos:")  
if not barras_fuera_norma_reactivos.empty:
    print("Barras fuera de norma:")
    print(barras_fuera_norma_reactivos)
else:
    print("No hay barras fuera de norma.")
    
if not lineas_saturadas_reactivos.empty:
    print("Líneas saturadas:")
    print(lineas_saturadas_reactivos)
else:
    print("No hay líneas saturadas.")

# Evaluación de las cargas
print("\nCargas de las líneas después del aumento de reactivos:")  
print(net.res_line)  

# Crear gráfico de las cargas
plt.figure(figsize=(12, 6))  
plt.bar(net.line.index, net.res_line.loading_percent, color='b', alpha=0.7)  
plt.axhline(y=100, color='r', linestyle='--', label='Límite de carga (100%)')  
plt.xlabel('Índice de la Línea')  
plt.ylabel('Carga de la Línea (%)')  
plt.title('Estado de Carga de Cada Línea después del aumento de reactivos')  
plt.xticks(net.line.index, net.line.name, rotation=45)  
plt.legend()  
plt.tight_layout()  
plt.show()  

# Crear gráfico de tensiones
fig, ax = plt.subplots(figsize=(12, 6))  
bar_width = 0.35  
index = range(len(net.bus))  

ax.bar(index, net.res_bus.vm_pu, bar_width, label='Tensiones (después de aumento de reactivos)')  

ax.set_xlabel('Barras')  
ax.set_ylabel('Voltaje (pu)')  
ax.set_title('Tensiones para cada caso después del aumento de reactivos')  
ax.set_xticks(index)  
ax.set_xticklabels(net.bus.name)  
ax.axhline(y=v_limit_sup, color='b', linestyle='--', label='Límite Sup. Estado Normal')  
ax.axhline(y=v_limit_inf, color='b', linestyle='--', label='Límite Inf. Estado Normal')  
ax.legend()  

# Zoom
limite_inferior_zoom = v_limit_inf - 0.02  
limite_superior_zoom = v_limit_sup + 0.02  
ax.set_ylim([limite_inferior_zoom, limite_superior_zoom])  

plt.tight_layout()  
plt.show()  

# Variación de cargas en ±15%
variaciones = [-0.15, 0, 0.15]  
resultados_tensiones = {}  
resultados_cargas = {}  

for idx, load in enumerate(net.load.index):
    for variacion in variaciones:
        net.load.at[load, 'p_mw'] *= (1 + variacion)
        pp.runpp(net)
        
        # Guardar resultados
        resultados_tensiones[(idx, variacion)] = net.res_bus.vm_pu.copy()
        resultados_cargas[(idx, variacion)] = net.res_line.loading_percent.copy()
        
        # Restaurar el valor original de la carga
        net.load.at[load, 'p_mw'] /= (1 + variacion)

# Graficar resultados de tensiones
fig, ax = plt.subplots(figsize=(12, 6))  

bar_width = 0.2
index = range(len(net.bus))

for (idx, variacion), voltajes in resultados_tensiones.items():
    offset = variaciones.index(variacion) * bar_width
    ax.bar([p + offset for p in index], voltajes, bar_width, label=f'Load {idx} ±{abs(variacion * 100)}%')

ax.set_xlabel('Barras')  
ax.set_ylabel('Voltaje (pu)')  
ax.set_title('Variación de Tensiones con ±15% de cambio en cargas')  
ax.set_xticks([p + bar_width for p in index])  
ax.set_xticklabels(net.bus.name)  
ax.axhline(y=v_limit_sup, color='b', linestyle='--', label='Límite Sup. Estado Normal')  
ax.axhline(y=v_limit_inf, color='b', linestyle='--', label='Límite Inf. Estado Normal')  
ax.legend()  

plt.tight_layout()  
plt.show()  

# Graficar resultados de cargas de líneas
fig, ax = plt.subplots(figsize=(12, 6))  

index = range(len(net.line))  

for (idx, variacion), cargas in resultados_cargas.items():
    offset = variaciones.index(variacion) * bar_width
    ax.bar([p + offset for p in index], cargas, bar_width, label=f'Load {idx} ±{abs(variacion * 100)}%')

ax.set_xlabel('Líneas')  
ax.set_ylabel('Carga de la Línea (%)')  
ax.set_title('Variación de Cargas de Líneas con ±15% de cambio en cargas')  
ax.set_xticks([p + bar_width for p in index])  
ax.set_xticklabels(net.line.name, rotation=45)  
ax.axhline(y=100, color='r', linestyle='--', label='Límite de carga (100%)')  
ax.legend()  

plt.tight_layout()  
plt.show()  



