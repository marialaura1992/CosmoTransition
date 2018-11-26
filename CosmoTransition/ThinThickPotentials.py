from cosmoTransitions.tunneling1D import SingleFieldInstanton
import matplotlib.pyplot as plt
import numpy as np

# Thin-walled
def V1(phi): return 0.25*phi**4 - 0.49*phi**3 + 0.235 * phi**2
def dV1(phi): return phi*(phi-.47)*(phi-1)
phi = np.arange(0, 1.7, 0.1)
plt.plot(phi, V1(phi), label = 'thin')

# Thick-walled
def V2(phi): return 0.25*phi**4 - 0.4*phi**3 + 0.1 * phi**2
def dV2(phi): return phi*(phi-.2)*(phi-1)
plt.plot(phi, V2(phi), label = 'thick')
plt.legend()
plt.xlabel(r"Field \phi")
plt.ylabel(r"Potential $V(\phi)$")
plt.show()
