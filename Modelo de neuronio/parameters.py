#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy import exp, zeros
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import parameters


def define_neuron_parameters():
    """Create a Python dictionary and return all neuron's model parameters."""

    p = {}
    ### Sodium Channel (Na) ###
    p['g_Na'] = np.array([860, 280, 280, 280])  # S/m**2, Maximum conductance of sodium channels

    # Activation (m)
    p['alpha_m_Na'] = lambda V: -0.2816e3 * (V + 0.028) / (-1 + np.exp(-(V + 0.028) / 0.0093))
    p['beta_m_Na'] = lambda V: 0.2464e3 * (V + 0.001) / (-1 + np.exp((V + 0.001) / 0.006))
    p['m_inf_Na'] = lambda V: p['alpha_m_Na'](V)/(p['alpha_m_Na'](V) + p['beta_m_Na'](V))
    p['tau_m_Na'] = lambda V: 1e-3/(p['alpha_m_Na'](V) + p['beta_m_Na'](V))

    # Inactivation (h)
    p['alpha_h_Na'] = lambda V: 0.098 * np.exp(-(V + 0.0431) / 0.020)
    p['beta_h_Na'] = lambda V: 1.4 / (1 + np.exp(-(V + 0.0131) / 0.010))
    p['h_inf_Na'] = lambda V: p['alpha_h_Na'](V)/(p['alpha_h_Na'](V) + p['beta_h_Na'](V))
    p['tau_h_Na'] = lambda V: 1e-3/(p['alpha_h_Na'](V) + p['beta_h_Na'](V))

    ### Persistent Sodium Channel (NaP) ###
    p['g_NaP'] = np.array([2.2, 1.0, 1.0, 0.0])

    # Activation (m)
    p['alpha_m_NaP'] = lambda V: -0.2816e3 * (V + 0.012) / (-1 + np.exp(-(V + 0.012) / 0.0093))
    p['beta_m_NaP'] = lambda V: 0.2464e3 * (V - 0.015) / (-1 + np.exp((V - 0.015) / 0.006))
    p['m_inf_NaP'] = lambda V: p['alpha_m_NaP'](V)/(p['alpha_m_NaP'](V) + p['beta_m_NaP'](V))
    p['tau_m_NaP'] = lambda V: 1e-3/(p['alpha_m_NaP'](V) + p['beta_m_NaP'](V))

    # Inactivation (h)
    p['alpha_h_NaP'] = lambda V: 2.8e-5 * np.exp(-(V + 0.0428477) / 0.0040248)
    p['beta_h_NaP'] = lambda V: 0.02 / (1 + np.exp(-(V - 0.4139284) / 0.1482589))
    p['h_inf_NaP'] = lambda V: p['alpha_h_NaP'](V)/(p['alpha_h_NaP'](V) + p['beta_h_NaP'](V))
    p['tau_h_NaP'] = lambda V: 1e-3/(p['alpha_h_NaP'](V) + p['beta_h_NaP'](V))


    ### High-voltage-activated Ca2+ Channel (HVA) ###
    p['g_HVA'] = np.array([0.34, 0.7, 0.7, 0.34])
    # Activation (u)
    p['u_inf_HVA'] = lambda V: 1/(1 + np.exp(-(V + 0.0246)/0.0113))
    p['tau_u_HVA'] = lambda V: 1.25e-3/np.cosh(-0.031e3*(V + 0.0371))

    # Inactivation (v)
    p['v_inf_HVA'] = lambda V: 1/(1 + np.exp((V + 0.0126)/0.0189))
    p['tau_v_HVA'] = lambda V: 420e-3

    ### Delayed Rectifier Potassium Channel (DR) ###
    p['g_DR'] = np.array([338, 92, 92, 92])

    # Activation (n)
    p['alpha_n_DR'] = lambda V: -0.018e3 * (V - 0.013) / (-1 + np.exp(-(V - 0.013) / 0.025))
    p['beta_n_DR'] = lambda V: 0.0054e3 * (V - 0.023) / (-1 + np.exp((V - 0.023) / 0.012))
    p['n_inf_DR'] = lambda V: p['alpha_n_DR'](V)/(p['alpha_n_DR'](V) + p['beta_n_DR'](V))
    p['tau_n_DR'] = lambda V: 1e-3/(p['alpha_n_DR'](V) + p['beta_n_DR'](V))
    
    ### Parâmetros de Potássio (K⁺) ###
    p['E_K'] = -77e-3  # V, potencial de reversão para canais de potássio

    ### Parâmetros para Dinâmica de Potássio Extracelular ###
    p['K_i'] = 140e-3       # mol/m³ (140 mM), concentração intracelular de K⁺
    p['K_o_rest'] = 3.82e-3  # mol/m³ (4 mM), concentração extracelular de K⁺ em repouso
    p['Vol_K'] = 70e-9       # m³, volume extracelular
    p['tau_K'] = 7e-3      # ms, constante de tempo para troca de K⁺

    
    ### Slow Potassium Channel (KS) ###
    p['g_KS'] = np.array([0.14, 0.24, 0.24, 0.24])

    # Activation (m)
    p['a_inf_KS'] = lambda V: 1 / (1 + np.exp(-(V + 0.034) / 0.0065))
    p['tau_a_KS'] = lambda V: 6e-3

    # Inactivation (h)
    p['b_inf_KS'] = lambda V: 1 / (1 + np.exp((V + 0.065) / 0.0066))
    p['tau_b_KS'] = lambda V: 200e-3 + 3200e-3 / (1 + np.exp(-(V + 0.0636) / 0.004))


    ### Calcium Channel (Ca) ###
    p['g_C'] = np.array([2.2, 3.8, 3.8, 2.2])

    # Calcium-dependent potassium channel (Ca)
    p['alpha_c'] = lambda V: (-0.00642*V - 0.1152) / (-1 + np.exp(-((V + 0.018e3) / 0.012e3)))
    p['beta_c'] = lambda V: 1.7 * np.exp(-(V + 0.152e3) / 0.030e3)
    p['c_inf'] = lambda V: p['alpha_c'](V) / (p['alpha_c'](V) + p['beta_c'](V))
    p['tau_c'] = lambda V: (1.1e-3*(1 / (p['alpha_c'](V) + p['beta_c'](V)) < 1.1e-3) + 
                                (1e-3 / (p['alpha_c'](V) + p['beta_c'](V)))*(1 / (p['alpha_c'](V) + p['beta_c'](V)) > 1.1e-3))

    # Equilibrium function for calcium
    p['f_EC'] = lambda Ca: 12.5e-3 * np.log(2e-3 / Ca)
    p['f_EK'] = lambda K: 25.0e-3 * np.log(K/p['K_i'])

    ### Physical Constants ###
    p['T'] = 303.16              # K
    p['F'] = 96500               # C/mole
    p['R'] = 8.314               # J/(mole * K)

    ### Membrane Capacitance ###
    p['C_m'] = np.array([1.2e-2, 2.3e-2, 2.3e-2, 2.3e-2])            # F/m**2

    ### Resting State ###
    p['V_rest'] = -0.065         # Resting membrane potential (V)
    p['E_Na']   = 0.050          # Sodium reversal potential (V)
    p['E_K']    = -0.077         # Potassium reversal potential (V)
    p['E_leak'] = -0.070         # Leak reversal potential (V)
    p['Ca_irest'] = 50e-9
    p['K_rest'] =  3.82e-3       # mol/l

    
    ### Leak Channels ###
    p['g_leak'] = np.array([0.333, 0.639, 0.639, 0.639])  # S/m**2, Maximum conductance of leak channels
    
    ### Morphological Parameters ###    
    p['Vol'] = np.array([2e-10, 2e-10, 2e-10, 2e-10])
                                                                                                                    
    
    p['phi'] = np.array([386e-9, 965e-9, 965e-9, 965e-9])  # Ca²⁺ buffer concentration
    p['tau_Ca'] = np.array([250e-3, 120e-3, 120e-3, 80e-3])  # Time constants for calcium dynamics (s)
    
    p['phi_K'] = 2

    # Resistência axial, geralmente em ohm·cm (valor típico de 150 mΩ·cm)
    p['R_a'] = 150e-3  

    ### Synaptic Transmission Parameters ###
    p['g_AMPA'] = 0.1e-9  # S, Maximum conductance of AMPA receptors
    p['E_AMPA'] = 0  # V, Reversal potential for AMPA receptors
    p['g_NMDA'] = 0.01e-9  # S, Maximum conductance of NMDA receptors
    p['E_NMDA'] = 0  # V, Reversal potential for NMDA receptors
    p['Mg_conc'] = 1e-3  # mM, Extracellular magnesium concentration
    p['K_Mg'] = 0.1e-3  # mM, Magnesium block parameter

    return p
