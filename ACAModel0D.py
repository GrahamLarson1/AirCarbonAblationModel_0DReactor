import numpy as np
import math

def ACAModel0D(T, P, B,
                Ofrac, O2frac, Nfrac,
                y_kO3,y_kO4,y_kO7,y_kO8,y_kO9,
                y_kN3,y_kN4,y_kN5,y_kN6,
                y_kOx2,y_kOx3,y_kOx5,
                EO2,EO3,EO4,EO6,EO7,EO8,EO9,
                EN1,EN2,EN3,EN4,EN5,EN6,
                EOx1,EOx2,EOx3,EOx4,EOx5,
                SO1,SO5,
                validation_output=0,
                newton_output=0):
    
    '''
    Zero-dimensional reactor of ACA model

    --- Parameters ---
    T: Temperature [K] - 1xN, 1D array of arbitrary length N (user specified)
    P: Pressure [Pa] - Scalar
    B: Total active site density [mol / m^2] - Scalar
    Ofrac: mole fraction of atomic oxygen - Scalar
    O2frac: mole fraction of molecular oxygen - Scalar
    Nfrac: mole fraction of atomic nitrogen - Scalar

    --- Pre-Exponential Factors - Scalars (corresponding to the associated reaction rate in the ACA model paper [Prata et el]) ---
    y_k03
    y_kO7
    y_kO8
    y_kO9
    y_kN3
    y_kN4
    y_kN5
    y_kN6
    y_kOx2
    y_kOx3
    y_kOx5

    --- Activation Energies - Scalars (corresponding to the associated reaction rate in the ACA model paper [Prata et el]) ---
    EO2
    EO3
    EO4
    EO6
    EO7
    EO8
    EO9
    EN1
    EN2
    EN3
    EN4
    EN5
    EN6
    EOx1
    EOx2
    EOx3
    EOx4
    EOx5

    --- Selectivity Coefficients - Scalars ---
    SO1
    SO5

    --- Diagnostic Outputs ---
    validation_output: binary output condition, 1 = print, default = 0
    newton_output: binary output condition, 1 = print, default = 0

    --- returns ---
    pO2: reaction probability of O2 - 1xN, 1D array of length N
    pO: reaction probability of O - 1xN, 1D array of length N
    pCO: reaction probability of CO - 1xN, 1D array of length N
    pCO2: reaction probability of CO2 - 1xN, 1D array of length N
    pN: reaction probability of N - 1xN, 1D array of length N
    pN2: reaction probability of N2 - 1xN, 1D array of length N
    pCN: reaction probability of CN - 1xN, 1D array of length N
    pO2react: reaction probability of O2 oxidation - 1xN, 1D array of length N
    '''

    #Physical Constants
    k_b = 1.380649 * 10 ** -23  # Boltzmann Constant
    A_v = 6.02214076 * 10 ** 23  # Avogadro Number
    h = 6.6260715 * 10 ** -34  # Planck Constant
    R = 8.31446261815324  # Universal Gas Constant [J / kg * mol]
    m_O = 2.6566962 * 10 ** -26  # Mass of oxygen atom [kg]
    M_O = 15.9994  # Molar mass of atomic oxygen
    m_O2 = m_O * 2  # Mass of oxygen molecule [kg]
    M_O2 = M_O * 2  # Molar mass of molecular oxygen
    m_N = 2.3258671 * 10 ** -26  # Mass of nitrogen atom [kg]
    M_N = 14.0067  # Molar mass of atomic nitrogen

    # Experiment Constants
    P_O = P * Ofrac  # Partial pressure of atomic oxygen
    P_O2 = P * O2frac  # Partial pressure of molecular oxygen
    P_N = P * Nfrac  # Partial pressure of atomic nitrogen
    O = P_O / (k_b * A_v * T)  # [O] value [mol / m^3]
    O2 = P_O2 / (k_b * A_v * T)  # [O2] value [mol / m^3]
    N = P_N / (k_b * A_v * T)  # [N] value [mol / m^3]

    # Vector Update
    T = np.atleast_1d(T)

    ''' Gas Properties '''
    def QuarterMeanThermalSpeed(T,m):
        F = 1/4*np.sqrt((8*k_b*T)/(np.pi*m))
        return F

    def MobileAbsorbedMeanThermalSpeed(T,m):
        F2D = np.sqrt((np.pi*k_b*T)/(2*m))
        return F2D

    F_O = QuarterMeanThermalSpeed(T,m_O) # Quarter mean thermal speed of atomic oxygen
    F_O2D = MobileAbsorbedMeanThermalSpeed(T,m_O) # Mean thermal speed of mobile adsorbed atomic oxygen
    F_N = QuarterMeanThermalSpeed(T,m_N) # Quarter mean thermal speed of atomic nitrogen
    F_N2D = MobileAbsorbedMeanThermalSpeed(T,m_N) # Mean thermal speed of mobile adsorbed atomic nitrogen
    F_O2 = QuarterMeanThermalSpeed(T,m_O2) # Quarter mean thermal speed of molecular oxygen
    fOin = F_O * O
    fO2in = F_O2 * O2
    fNin = F_N * N

    ''' Reaction Coefficient Equations '''
    # Atomic Oxygen Reactions
    kO1 = F_O/B*SO1
    kO2 = ((2*np.pi*m_O*k_b**2*T**2)/(A_v*B*h**3))*math.e**(-EO2/T)
    kO3 = F_O/B*y_kO3*math.e**(-EO3/T)
    kO4 = F_O/B*y_kO4*math.e**(-EO4/T)
    kO5 = F_O/B*SO5
    kO6 = ((2*np.pi*m_O*k_b**2*T**2)/(A_v*B*h**3))*math.e**(-EO6/T)
    kO7 = F_O/B*y_kO7*math.e**(-EO7/T)
    kO8 = np.sqrt(A_v/B)*F_O2D*y_kO8*math.e**(-EO8/T)
    kO9 = np.sqrt(A_v/B)*F_O2D*y_kO9*math.e**(-EO9/T)

    # Atomic Nitrogen Reactions
    kN1 = F_N/B*math.e**(-EN1/T)
    kN2 = ((2*np.pi*m_N*k_b**2*T**2)/(A_v*B*h**3))*math.e**(-EN2/T)
    kN3 = F_N/B*y_kN3*math.e**(-EN3/T)
    kN4 = F_N/B*y_kN4*math.e**(-EN4/T)
    kN5 = np.sqrt(A_v/B)*F_N2D*y_kN5*math.e**(-EN5/T)
    kN6 = y_kN6*math.e**(-EN6/T)

    # Molecular Oxygen Reactions
    kOx1 = F_O2/B**2*math.e**(-EOx1/T)
    kOx2 = F_O2/B*y_kOx2*math.e**(-EOx2/T)
    kOx3 = F_O2/B*y_kOx3*math.e**(-EOx3/T)
    kOx4 = F_O2/B**2*math.e**(-EOx4/T)
    kOx5 = F_O2/B*y_kOx5*math.e**(-EOx5/T)

    ''' ACA Steady State Model '''
    A1 = 2*kOx1*O2
    B1 = kO1*O
    C1 = 2*kO9
    D1 = kO2+(kO3+kO4)*O+(kOx2+kOx3)*O2
    A2 = 2*kOx4*O2
    B2 = kO5*O
    C2 = 2*kO8
    D2 = kO6+kO7*O+kOx5*O2
    A3 = 0
    B3 = kN1*N
    C3 = 2*kN5
    D3 = kN2+kN6+(kN3+kN4)*N

    ''' Newton Method Root Solver '''
    # Surface density of empty sites (balanced RHS to zero)
    def s_function(s):
        func = (B -
                (2*(A1*s**2 + B1*s) / (D1 + np.sqrt(D1**2 + 4*C1*(A1*s**2 + B1*s)))) -
                (2*(A2*s**2 + B2*s) / (D2 + np.sqrt(D2**2 + 4*C2*(A2*s**2 + B2*s)))) -
                ((2*B3*s) / (D3 + np.sqrt(D3**2 + 4*C3*B3*s)))
                -s)
        return func

    # First derivative of surface density of empty sites
    def s_prime(s):
        func = (-(2*(2*A1*s+B1))/(D1 + np.sqrt(D1**2 + 4*C1*(A1*s**2 + B1*s))) + (4*C1*(2*A1*s + B1)*(A1*s**2 + B1*s)) / (np.sqrt(D1**2 + 4*C1*(A1*s**2 + B1*s)) * ((D1 + np.sqrt(D1**2 + 4*C1*(A1*s**2 + B1*s)))**2))
              -(2*(2*A2*s+B2))/(D2 + np.sqrt(D2**2 + 4*C2*(A2*s**2 + B2*s))) + (4*C2*(2*A2*s + B2)*(A2*s**2 + B2*s)) / (np.sqrt(D2**2 + 4*C2*(A2*s**2 + B2*s)) * ((D2 + np.sqrt(D2**2 + 4*C2*(A2*s**2 + B2*s)))**2))
              -(2*(2*A3*s+B3))/(D3 + np.sqrt(D3**2 + 4*C3*(A3*s**2 + B3*s))) + (4*C3*(2*A3*s + B3)*(A3*s**2 + B3*s)) / (np.sqrt(D3**2 + 4*C3*(A3*s**2 + B3*s)) * ((D3 + np.sqrt(D3**2 + 4*C3*(A3*s**2 + B3*s)))**2))
                -1)
        return func

    def NewtonMethod():
        s = 0
        for i in range(10):
            s = s - s_function(s) / s_prime(s)
        if newton_output == 1:
            print('s value: ', s)
            print('Root validation: ', s_function(s))
            print('')
        return s

    ''' [O(s)], [O*(s)], [N(s)] relations for steady state '''
    def O_s(s):
        O_s = (2*(A1*s**2+B1*s))/(D1+(D1**2+4*C1*(A1*s**2+B1*s))**(1/2))
        return O_s
    def Ostar_s(s):
        Ostar_s = (2*(A2*s**2+B2*s))/(D2+(D2**2+4*C2*(A2*s**2+B2*s))**(1/2))
        return Ostar_s
    def N_s(s):
        N_s = (2*B3*s)/(D3+np.sqrt(D3**2+4*C3*B3*s))
        return N_s

    ''' ODE System (used to calculate reaction probabilities) '''
    def dOs(s,O_s):
        dOs = kO1*O*s - kO2*O_s - kO3*O*O_s - kO4*O*O_s - 2*kO9*O_s**2 + 2*kOx1*O2*s**2 - kOx2*O2*O_s - kOx3*O2*O_s #dO_s/dt
        return dOs
    def dOstars(s,Ostar_s):
        dOstars = kO5*O*s - kO6*Ostar_s - kO7*O*Ostar_s - 2*kO8*Ostar_s**2 + 2*kOx4*O2*s**2 - kOx5*O2*Ostar_s #dOstar_s/dt
        return dOstars
    def dCO(O_s, Ostar_s):
        dCO = kO3*O*O_s + kO7*O*Ostar_s + kOx2*O2*O_s + kOx5*O2*Ostar_s #dCO/dt
        return dCO
    def dCO2(O_s):
        dCO2 = kO4*O*O_s + kOx3*O2*O_s #dCO2/dt
        return dCO2
    def dO(s,O_s, Ostar_s):
        dO = P_O/(A_v*(2*np.pi*m_O*k_b*T)**(1/2)) - kO1*O*s + kO2*O_s - kO4*O*O_s - kO5*O*s + kO6*Ostar_s + kOx3*O2*O_s #dO/dt
        return dO
    def dO2(s,O_s,Ostar_s):
        dO2 = P_O2/(A_v*(2*np.pi*m_O2*k_b*T)**(1/2)) + kO8*Ostar_s**2 + kO9*O_s**2 - kOx1*O2*s**2 - kOx3*O2*O_s - kOx4*O2*s**2 #dO2/dt
        return dO2
    def dNs(s,N_s):
        dNs = kN1*N*s - kN2*N_s - kN3*N*N_s - kN4*N*N_s - 2*kN5*N_s**2 - kN6*N_s #dN_s/dt
        return dNs
    def dCN(N_s):
        dCN = kN3*N*N_s + kN6*N_s #dCN/dt
        return dCN
    def dN(s, N_s):
        dN = P_N/(A_v*np.sqrt(2*np.pi*m_N*k_b*T)) - kN1*N*s + kN2*N_s - kN4*N*N_s #dN/dt
        return dN
    def dN2(N_s):
        dN2 = kN4*N*N_s + kN5*N_s**2 #dN2/dt
        return dN2

    ''' Reaction Probability Equations'''
    def probO(fO,fOin):
        if Ofrac == 0:
            pO = np.zeros(len(T))
        else:
            pO = fO / fOin
        return pO
    def probO2(fO2, fOin):
        if Ofrac == 0:
            pO2 = np.zeros(len(T))
        else:
            pO2 = 2*fO2 / fOin
        return pO2
    def probN(fN, fNin):
        if Nfrac == 0:
            pN = np.zeros(len(T))
        else:
            pN = fN/fNin
        return pN
    def probN2(fN2, fNin):
        if Nfrac == 0:
            pN2 = np.zeros(len(T))
        else:
            pN2 = 2*fN2 / fNin
        return pN2
    def probCN(fCN, fNin):
        if Nfrac == 0:
            pCN = np.zeros(len(T))
        else:
            pCN = fCN / fNin
        return pCN
    def probO2react(fCO2, fCO, fO2in):
        if O2frac == 0:
            pO2react = np.zeros(len(T))
        else:
            pO2react = (2 * fCO2 + fCO) / fO2in
        return pO2react
    def probCO(fCO, fOin, fO2in):
        if Ofrac == 0 and O2frac == 0:
            pCO = np.zeros(len(T))
        else:
            pCO = fCO / (fOin + 2*fO2in)
        return pCO
    def probCO2(fCO2, fOin, fO2in):
        if Ofrac == 0 and O2frac == 0:
            pCO2 = np.zeros(len(T))
        else:
            pCO2 = 2*fCO2 / (fOin + 2*fO2in)
        return pCO2

    s_test = NewtonMethod() # Determine s value

    ''' Probability Calculations '''
    fO2 = dO2(s_test, O_s(s_test), Ostar_s(s_test)) # Flux of O2
    p_O2 = probO2(fO2, fOin) # Probability of O2

    fO = dO(s_test, O_s(s_test), Ostar_s(s_test))
    p_O = probO(fO, fOin)  # Probability of O2

    fCO = dCO(O_s(s_test), Ostar_s(s_test))
    p_CO = probCO(fCO,fOin,fO2in)

    fCO2 = dCO2(O_s(s_test))
    p_CO2 = probCO2(fCO2, fOin, fO2in)

    fN = dN(s_test,N_s(s_test))
    p_N = probN(fN, fNin)

    fN2 = dN2(N_s(s_test))
    p_N2 = probN2(fN2, fNin)

    p_O2react = probO2react(fCO2, fCO, fO2in)

    fCN = dCN(N_s(s_test))
    p_CN = probCN(fCN, fNin)

    # Validation Step
    if validation_output == 1:
        validation = O_s(s_test) + Ostar_s(s_test) + N_s(s_test) + s_test - B
        print('Solver Validation: [O(s)] + [O*(s)] + [N(s)] + [(s)] - B = 0 :', validation)
        print('')
    return p_O, p_O2, p_CO, p_CO2, p_N, p_N2, p_CN, p_O2react