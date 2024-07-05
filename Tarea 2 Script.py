import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import numpy as np

# Creación Red
net = pp.create_empty_network(name="empty")

# Barras
b1 = pp.create_bus(net, vn_kv=110.0, name="bus 1")
b2 = pp.create_bus(net, vn_kv=220.0, name="bus 2")
b3 = pp.create_bus(net, vn_kv=220.0, name="bus 1a")
b4 = pp.create_bus(net, vn_kv=220.0, name="bus 2a")
b5 = pp.create_bus(net, vn_kv=220.0, name="bus 3a")
b6 = pp.create_bus(net, vn_kv=220.0, name="bus 2b")
b7 = pp.create_bus(net, vn_kv=220.0, name="bus 1b")

# Generador
pp.create_ext_grid(net, b1, vm_pu=1.0, name="external grid", va_degree=0)

# Transformador
trafo1 = pp.create_transformer(net, b2, b1, name="110kV/220kV transformer", std_type="100 MVA 220/110 kV")

# Líneas
line1 = pp.create_line(net, b2, b3, length_km=10, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 1")
line2 = pp.create_line(net, b3, b4, length_km=15, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 2")
line3 = pp.create_line(net, b4, b5, length_km=20, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 3")
line4 = pp.create_line(net, b5, b6, length_km=15, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 4")
line5 = pp.create_line(net, b6, b7, length_km=30, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 5")
line6 = pp.create_line(net, b2, b7, length_km=10, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", name="Line 6")

# Cargas
load1 = pp.create_load(net, b3, p_mw=150, q_mvar=100, scaling=0.2, name="load 1a")
load2 = pp.create_load(net, b4, p_mw=150, q_mvar=100, scaling=0.35, name="load 2a")
load3 = pp.create_load(net, b5, p_mw=150, q_mvar=100, scaling=0.15, name="load 3a")
load4 = pp.create_load(net, b6, p_mw=150, q_mvar=100, scaling=0.6, name="load 2b")
load5 = pp.create_load(net, b7, p_mw=150, q_mvar=100, scaling=0.1, name="load 1b")

# barra PV (barra 2)
pp.create_gen(net, bus=b2, p_mw=0, vm_pu=1.0, name="PV generator") 

# Resolver flujo de carga
pp.runpp(net)

# Resultados del flujo de carga
Tensiones = net.res_bus.vm_pu
Angulos = net.res_bus.va_degree

print("Tensiones pu")
print(Tensiones)

print("Ángulos(grados):")
print(Angulos)

# 5 BARRAS PQ Y UNA BARRA PV, ENTONCES HAY 11 ECUACIONES Y 11 INCÓGNITAS.

# Matriz Ybus
Y_bus = net._ppc['internal']['Ybus'].todense()

# Se Definen los valores iniciales para las tensiones y ángulos.
V = np.ones(len(net.bus))  # Tensiones en 1.0 p.u.
theta = np.zeros(len(net.bus))  # Ángulos en 0 radianes(python lo calcula así).

# Se Definen P, Q en p.u.
S_base = 100  # Base de potencia en MVA
P_variable = np.zeros(len(net.bus))  # Se empieza con ceros
Q_variable = np.zeros(len(net.bus))  # Se empieza con ceros

# Configurar las potencias especificadas según las cargas definidas
for load in net.load.itertuples():
    P_variable[load.bus] -= load.p_mw / S_base  # Convertir MW a pu 
    Q_variable[load.bus] -= load.q_mvar / S_base  # Convertir MVar a pu 
# Se definen las barras PV y PQ (Menos la barra slack)
slack_bus = int(net.ext_grid.bus.values[0])
pv_buses = [bus for bus in net.gen.bus if bus != slack_bus]
pq_buses = [bus for bus in net.load.bus if bus != slack_bus]


# Función para calcular el desbalance de potencia
def calc_power_mismatch(Y_bus, V, theta, P_spec, Q_spec, pv_buses, pq_buses):
    # Número de barras
    n_buses = len(V)
    
    # Potencias P y Q
    P = np.zeros(n_buses)
    Q = np.zeros(n_buses)
    
    for i in range(n_buses):
        for j in range(n_buses):
            P[i] += V[i] * V[j] * (Y_bus[i,j].real * np.cos(theta[i] - theta[j]) + Y_bus[i,j].imag * np.sin(theta[i] - theta[j]))
            Q[i] += V[i] * V[j] * (Y_bus[i,j].real * np.sin(theta[i] - theta[j]) - Y_bus[i,j].imag * np.cos(theta[i] - theta[j]))
    
    # Delta P y delta Q
    dP = P_variable - P
    dQ = Q_variable - Q
    
    # Vector
    mismatch = []
    
    # Para barras PV, sólo  P
    for i in pv_buses:
        mismatch.append(dP[i])
    
    # Para barras PQ, sólo P y Q
    for i in pq_buses:
        mismatch.append(dP[i])
        mismatch.append(dQ[i])
    
    return np.array(mismatch)

# Función para la matriz Jacobiana
def calc_jacobian(Y_bus, V, theta, P, Q, pv_buses, pq_buses):
    n_buses = len(V)
    H = np.zeros((n_buses, n_buses))
    N = np.zeros((n_buses, n_buses))
    M = np.zeros((n_buses, n_buses))
    L = np.zeros((n_buses, n_buses))
    
    for i in range(n_buses):
        for j in range(n_buses):
            if i != j:
                H[i, j] = -V[i] * V[j] * (Y_bus[i, j].real * np.sin(theta[i] - theta[j]) - Y_bus[i, j].imag * np.cos(theta[i] - theta[j]))
                N[i, j] = -V[i] * (Y_bus[i, j].real * np.cos(theta[i] - theta[j]) + Y_bus[i, j].imag * np.sin(theta[i] - theta[j]))
                M[i, j] = V[i] * V[j] * (Y_bus[i, j].real * np.cos(theta[i] - theta[j]) + Y_bus[i, j].imag * np.sin(theta[i] - theta[j]))
                L[i, j] = -V[i] * (Y_bus[i, j].real * np.sin(theta[i] - theta[j]) - Y_bus[i, j].imag * np.cos(theta[i] - theta[j]))
            else:
                H[i, i] = Q[i] + V[i]**2 * Y_bus[i, i].imag
                N[i, i] = -P[i] / V[i] - V[i] * Y_bus[i, i].real
                M[i, i] = -P[i] + V[i]**2 * Y_bus[i, i].real
                L[i, i] = -Q[i] / V[i] + V[i] * Y_bus[i, i].imag
    
    # Excluir la barra slack
    pv_pq_buses = pv_buses + pq_buses
    H = H[np.ix_(pv_pq_buses, pv_pq_buses)]
    N = N[np.ix_(pv_pq_buses, pq_buses)]
    M = M[np.ix_(pq_buses, pv_pq_buses)]
    L = L[np.ix_(pq_buses, pq_buses)]
    
    return H, N, M, L

# Iteraciones de Newton-Raphson
max_iter = 3
tol = 1e-6

for k in range(max_iter):
    P = net.res_bus.p_mw.values / S_base
    Q = net.res_bus.q_mvar.values / S_base
    
    F_xk = calc_power_mismatch(Y_bus, V, theta, P_variable, Q_variable, pv_buses, pq_buses)
    
    if np.linalg.norm(F_xk) < tol:
        print(f'Converged in {k} iterations')
        break
    
    H, N, M, L = calc_jacobian(Y_bus, V, theta, P, Q, pv_buses, pq_buses)
    
    J_top = np.hstack((H, N))
    J_bottom = np.hstack((M, L))
    J = np.vstack((J_top, J_bottom))
    
    J_inv = np.linalg.inv(J)
    delta_x = -np.dot(J_inv, F_xk)
    
    # Actualizar tensiones y ángulos
    delta_theta = delta_x[:len(pv_buses) + len(pq_buses)]
    delta_V = delta_x[len(pv_buses) + len(pq_buses):]
    
    for i, bus in enumerate(pv_buses + pq_buses):
        theta[bus] += delta_theta[i]
    
    for i, bus in enumerate(pq_buses):
        V[bus] += delta_V[i]
    
    # Convertir los ángulos a grados
    theta_deg = np.degrees(theta)
    #Tensión p.u.
    V_pu = V
    print(f'\nIteration {k + 1}:')
    for i, bus in enumerate(net.bus.index):
        print(f'{net.bus.at[bus, "name"]}: Theta = {theta_deg[bus]:.4f} grados, V = {V[bus]:.4f} pu'),
    

reference_va_degree = net.res_bus.va_degree.values
reference_vm_pu = net.res_bus.vm_pu.values

print('\nResultados de referencia (pandapower):')
for bus in net.bus.index:
    print(f'{net.bus.at[bus, "name"]}: Theta = {reference_va_degree[bus]:.4f} grados, V = {(reference_vm_pu[bus]):.4f} pu')

print('\nResultados finales del método de Newton-Raphson:')
for bus in net.bus.index:
    print(f'{net.bus.at[bus, "name"]}: Theta = {theta_deg[bus]:.4f} grados, V = {V_pu[bus]:.4f} pu')

# Newton-Raphson con tolerancias variables
def newton_raphson(Y_bus, V, theta, P_variable, Q_variable, pv_buses, pq_buses, tol, max_iter=20):
    for k in range(max_iter):
        P = net.res_bus.p_mw.values / S_base
        Q = net.res_bus.q_mvar.values / S_base
        
        F_xk = calc_power_mismatch(Y_bus, V, theta, P_variable, Q_variable, pv_buses, pq_buses)
        
        if np.linalg.norm(F_xk) < tol:
            return V, theta, k + 1
        
        H, N, M, L = calc_jacobian(Y_bus, V, theta, P, Q, pv_buses, pq_buses)
        
        J_top = np.hstack((H, N))
        J_bottom = np.hstack((M, L))
        J = np.vstack((J_top, J_bottom))
        
        J_inv = np.linalg.inv(J)
        delta_x = -np.dot(J_inv, F_xk)
        
        # Actualizar tensiones y ángulos
        delta_theta = delta_x[:len(pv_buses) + len(pq_buses)]
        delta_V = delta_x[len(pv_buses) + len(pq_buses):]
        
        for i, bus in enumerate(pv_buses + pq_buses):
            theta[bus] += delta_theta[i]
        
        for i, bus in enumerate(pq_buses):
            V[bus] += delta_V[i]
    
    return V, theta, max_iter  # Si no converge en max_iter iteraciones, retorna valores


#distintas tolerancias
tolerancias = [1e-4, 1e-6, 1e-8]
resultados = {}

for tol in tolerancias:
    V, theta, iterations = newton_raphson(Y_bus, V.copy(), theta.copy(), P_variable, Q_variable, pv_buses, pq_buses, tol)
    theta_deg = np.degrees(theta)
    resultados[tol] = (V, theta_deg, iterations)

# Comparación de resultados
reference_va_degree = net.res_bus.va_degree.values
reference_vm_pu = net.res_bus.vm_pu.values

print('\nResultados de referencia (pandapower):')
for bus in net.bus.index:
    print(f'{net.bus.at[bus, "name"]}: Theta = {format(reference_va_degree[bus], ".4e")} grados, V = {format(reference_vm_pu[bus], ".4e")} pu')

for tol, (V_pu, theta_deg, iterations) in resultados.items():
    print(f'\nResultados con tolerancia {tol}:')
    print(f'Converged in {iterations} iterations')
    for bus in net.bus.index:
        print(f'{net.bus.at[bus, "name"]}: Theta = {format(theta_deg[bus], ".4e")} grados, V = {format(V_pu[bus], ".4e")} pu')