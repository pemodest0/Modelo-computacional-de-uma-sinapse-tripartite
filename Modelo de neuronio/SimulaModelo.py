# %%
import modelo_neuronio
import parameters

import numpy as np
import matplotlib.pyplot as plt

# %%
plt.rcParams["font.family"] = 'Arial'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

# %%
# Parâmetros de simulação
dt = 0.01e-3  # Passo de tempo (ms)
tf = 100.0e-3  # Duração da simulação (ms)
I_config = {'Init': 0, 'End': 100e-3, 'A': 5e-2}

p = parameters.define_neuron_parameters()

# Rodando o modelo
V, Ca, K_o = modelo_neuronio.neuron_model(tf, dt, p, I_config)

# %%
# Plotando o resultado
t = np.linspace(0, tf, V.shape[0])

label = ['soma', 'proximal', 'distal', 'basal']

fig, ax = plt.subplots(1, 2, figsize=(6, 4))

for i in range(4):         
    ax[0].plot(t*1e3, V[:, i]*1e3, label=label[i])
    ax[1].plot(t*1e3, Ca[:, i]*1e3, label=label[i])
    
ax[0].set_xlabel('t (ms)')
ax[0].set_xlim(0, 100)
ax[0].set_ylabel(r'$V~\mathrm{(mV)}$')
ax[0].set_ylim(-80, 60)


ax[1].set_xlabel('t (ms)')
ax[1].set_xlim(0, 100)
ax[1].set_ylabel(r'$\mathrm{[Ca^{2+}]}~\mathrm{(mM)}$')
ax[1].set_ylim(0, 0.1)

for axis in ax.flatten():
    for spine in ('right', 'top'): axis.spines[spine].set_visible(False)

ax[1].legend()

# %%
5000/55

# %%
# Plotando o resultado
t = np.linspace(0, tf, V.shape[0])

plt.figure(figsize=(10, 6))
for i in range(4): plt.plot(t*1e3, 1e3*Ca[:, i], label='V_soma')
plt.xlabel('Tempo (ms)')
plt.ylabel('[Ca2+] (mM)')

plt.ylim(0, 0.1)

plt.legend()

plt.show()


