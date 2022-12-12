import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

def linfun(x,a,b):
    func = a*x + b
    return func

conductivity = np.array([401,398,392,388,383,377,371, 364, 357, 350, 342, 334])
temp = np.array([273, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]) - 273

popt_lin, pconv_lin = curve_fit(linfun, temp, conductivity)

plt.figure(1)
plt.plot(temp, conductivity)
plt.plot(temp, linfun(temp, popt_lin[0], popt_lin[1]))
plt.show()