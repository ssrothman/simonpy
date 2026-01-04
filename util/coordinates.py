import numpy as np

def xyz_to_eta_phi(x, y, z):
    """Convert Cartesian coordinates to (eta, phi)"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(y, x)
    return eta, phi

def eta_to_theta(eta):
    """Convert pseudorapidity to polar angle theta"""
    return 2 * np.arctan(np.exp(-eta))