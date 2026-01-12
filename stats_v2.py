from typing import Literal, overload
import fasteigenpy as eigen
import numpy as np

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[True]) -> tuple[np.ndarray, np.ndarray]:
    ...

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[False]) -> np.ndarray:
    ...

def smart_inverse(matrix : np.ndarray, return_eigenspectrum : bool):
    err = np.sqrt(np.diag(matrix))
    err[err==0] = 1.0 # prevent division by zero
    inverr = 1.0 / err
    corr = np.diag(inverr) @ matrix @ np.diag(inverr)

    solver = eigen.SelfAdjointEigenSolver(corr)

    if solver.info() != eigen.ComputationInfo.Success:
        print("Eigen decomposition failed!")
        print(solver.info())
        raise RuntimeError("Eigen decomposition failed!")
    
    eigvals = np.asarray(solver.eigenvalues()).copy()
    eigvecs = np.asarray(solver.eigenvectors())
    
    eigvals[eigvals < 0] = 0.0

    denom = np.where(eigvals == 0, 1, eigvals) # prevent division by zero
    inveigvals = 1.0 / denom
    inveigvals[eigvals == 0] = 0.0 # set back to zero
    
    inverse = eigvecs @ np.diag(inveigvals) @ eigvecs.T

    if return_eigenspectrum:
        return inverse, eigvals
    else:
        return inverse