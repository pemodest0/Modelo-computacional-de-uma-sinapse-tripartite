import numpy as np
from numpy import exp, zeros
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import parameters


def neuron_model(t, dt, p, I_config={'Init': 50, 'End': 150, 'A': 150}):
    """Simulate the neuron model with multiple compartments."""

    # Constantes
    F = 96485.3329  # C/mol, constante de Faraday
    R = 8.314  # J/(mol·K), constante dos gases ideais
    T = 310.15  # K, temperatura (37 °C)

    # Gating variables
    # Modificação: vetor para múltiplos compartimentos
    m = np.ones(4) * p['m_inf_Na'](p['V_rest'])
    h = np.ones(4) * p['h_inf_Na'](p['V_rest'])
    n = np.ones(4) * p['n_inf_DR'](p['V_rest'])
    m_NaP = np.ones(4) * p['m_inf_NaP'](p['V_rest'])
    h_NaP = np.ones(4) * p['h_inf_NaP'](p['V_rest'])
    u = np.ones(4) * p['u_inf_HVA'](p['V_rest'])
    v = np.ones(4) * p['v_inf_HVA'](p['V_rest'])
    a_KS = np.ones(4) * p['a_inf_KS'](p['V_rest'])
    b_KS = np.ones(4) * p['b_inf_KS'](p['V_rest'])
    # A concentração de cálcio `Ca` está sendo convertida de mol/m³ para µM (1e6 * mol/m³)
    c = np.ones(4) * p['c_inf'](p['V_rest']*1e3 + 40*np.log10(p['Ca_irest']*1e6))

    # Storage for results
    T = int(t / dt)
    V = np.ones(4) * p['V_rest']  # Array para armazenar V de todos os compartimentos
    Ca = np.ones(4) * p['Ca_irest']  # Array para armazenar Ca de todos os compartimentos (mol/m³)
    K_o = np.ones(4) * p['K_o_rest']  # [K⁺]_o inicial em cada compartimento
    
    # Output
    V_out = np.zeros(shape=(T, 4))
    Ca_out = np.zeros(shape=(T, 4))
    K_o_out = np.zeros(shape=(T, 4))
    
    # Connection Matrix
    M = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]])

    for i in range(T):
        t = i * dt

        # Corrente externa aplicada (A)
        I_ext = I_config['A'] * (I_config['Init'] < t) * (t < I_config['End'])

        ### Atualizando variáveis de gating ###
        # I_Na
        m += (p['m_inf_Na'](V) - m) / p['tau_m_Na'](V) * dt
        h += (p['h_inf_Na'](V) - h) / p['tau_h_Na'](V) * dt

        # I_NaP
        m_NaP += (p['m_inf_NaP'](V) - m_NaP) / p['tau_m_NaP'](V) * dt
        h_NaP += (p['h_inf_NaP'](V) - h_NaP) / p['tau_h_NaP'](V) * dt

        # I_HVA
        u += (p['u_inf_HVA'](V) - u) / p['tau_u_HVA'](V) * dt
        v += (p['v_inf_HVA'](V) - v) / p['tau_v_HVA'](V) * dt

        # I_DR
        n += (p['n_inf_DR'](V) - n) / p['tau_n_DR'](V) * dt

        # I_KS
        a_KS += (p['a_inf_KS'](V) - a_KS) / p['tau_a_KS'](V) * dt
        b_KS += (p['b_inf_KS'](V) - b_KS) / p['tau_b_KS'](V) * dt

        # I_C (Corrente de cálcio)
        V_s = V * 1e3 + 40 * np.log10(Ca * 1e3)  # Usando log10 de concentração em µM
        c += (p['c_inf'](V_s) - c) / p['tau_c'](V_s) * dt

        ### Cálculo das correntes ###
        # I_Na
        I_Na = p['g_Na'] * m ** 3 * h * (V - p['E_Na'])
        # I_NaP
        I_NaP = p['g_NaP'] * m_NaP * h_NaP * (V - p['E_Na'])
        # I_HVA
        I_HVA = p['g_HVA'] * u ** 2 * v * (V - p['f_EC'](Ca))
        # I_DR
        I_DR = p['g_DR'] * n ** 4 * (V - p['f_EK'](K_o))
        # I_KS
        I_KS = p['g_KS'] * a_KS * b_KS * (V - p['f_EK'](K_o))
        # I_C (Corrente de cálcio)
        I_C = p['g_C'] * c ** 2 * (V - p['f_EK'](K_o))
        # Corrente de vazamento
        I_leak = p['g_leak'] * (V - p['E_leak'])

        # Corrente entre Compartimentos 
        I_R = -1/p['R_a']*(M.dot(V) - M.sum(axis = 1)*V)

        ### Atualização dos potenciais de membrana ###
        dV_soma = (-I_Na - I_DR - I_leak - I_HVA - I_NaP - I_KS - I_C + I_ext - I_R) / p['C_m']
        V += dV_soma * dt

        ### Atualização da concentração de cálcio (mol/m³) ###
        dCa_dt = -1e-3 * p['phi'] / (F * p['Vol']) * I_HVA + (p['Ca_irest'] - Ca) / p['tau_Ca']
        Ca += dCa_dt * dt
        
        # Atualiza [K⁺]_o com base nas correntes de potássio
        dK_o_dt = -1e-3 * p['phi_K'] / (F * p['Vol_K']) * (I_DR + I_KS + I_C) + (p['K_o_rest'] - K_o)/p['tau_K']
        K_o += dK_o_dt * dt

        
        ### Armazenamento dos resultados ###
        V_out[i] = V
        Ca_out[i] = Ca
        K_o_out[i] = K_o

    return V_out, Ca_out, K_o_out
