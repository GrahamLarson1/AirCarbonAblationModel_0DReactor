This repo contains a Zero-dimensional implementation of the Air Carbon Ablation model by Prata et. al. [1]

The 0D model takes the following as inputs: 
Pressure [Pa]
Temperature [K]
Active surface reaction site density [mol/m^2]
Atomic oxygen fraction
Molecular oxygen fraction
Atomic nitrogen fraction
Reaction rate parameters (32 total)

The 0D model outputs the probability of each reactant product from 0-1 of the following species:
O
O2
CO
CO2
N
N2
CN
O2 reaction

run.py loads the parameter set contained in DenormedMAP_params.txt (maximum a posteriori from Bayesian inference) 
and executes one run of the ACA 0D model. The results are plotted against molecular beam data from Murray et al. [2]

[1] Prata, K. S., Schwartzentruber, T. E., and Minton, T. K. (2022). 
"Air–Carbon Ablation Model for Hypersonic Flight from Molecular-Beam Data." AIAA Journal, 60(2), February 2022.

[2] Murray, V. J., Recio, P., Caracciolo, A., Miossec, C., Balucani, N., Casavecchia, P., and Minton, T. K. (2020). 
"Oxidation and Nitridation of Vitreous Carbon at High Temperatures." Carbon, 167, 388–402.
