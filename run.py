from ACAModel0D import ACAModel0D
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load reaction rate equation parameters
# (using maximum a posteriori from Bayesian inference with molecular beam data)
params = np.loadtxt(f'{script_dir}/DenormedMAP_params.txt')
print('Parameter array size:',params.shape)

# Specify flow and material conditions
# (currently set for a molecular beam experiment with an oxygen beam)
T = np.linspace(800,2600,100) # Temperature [K]
P = 2.4e-2                    # Pressure [Pa]
B = 1e-5                      # Active surface reaction site density [mol / m^2]
Ofrac = 1.0                   # Oxygen fraction in flow
O2frac = 0.0                  # O2 fraction in flow
Nfrac = 0.0                   # Nitrogen fraction in flow
conditions = T,P,B,Ofrac,O2frac,Nfrac

# Run ACA model and pack output into a dictionary
outputs = ACAModel0D(*conditions, *params)
species_names = ['O', 'O2', 'CO', 'CO2', 'N', 'N2', 'CN', 'O2react']
prediction = dict(zip(species_names, outputs))

# Create colormap
colors = plt.cm.tab10.colors[:8]

# Plot results
for i, sp in enumerate(prediction):
    if any(prediction[sp] > 0.0):
        plt.plot(T,prediction[sp],label=sp,color=colors[i])
        mbdata = np.loadtxt(f'{script_dir}/MolecularBeamExperiments/p{sp}.csv',delimiter=',')
        plt.scatter(mbdata[:,0],mbdata[:,1],label=f'{sp} MB',marker='x',s=10,color=colors[i])

# Plot format
plt.ylim(0,1)
plt.xlim(np.min(T),np.max(T))
plt.xlabel('Temperature [K]')
plt.ylabel('Probability of Reaction Product')
plt.legend()
plt.show()