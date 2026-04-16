from typing import List, Literal, assert_never, overload
import numpy as np

from simonpy.AbitraryBinning import ArbitraryBinning

@overload
def apply_jacobian(val : np.ndarray, cov : None, binning : ArbitraryBinning, jac_details : dict) -> np.ndarray:
    ...

@overload
def apply_jacobian(val : None, cov : np.ndarray, binning : ArbitraryBinning, jac_details : dict) -> np.ndarray:
    ...

@overload
def apply_jacobian(val : np.ndarray, cov : np.ndarray, binning : ArbitraryBinning, jac_details : dict) -> tuple[np.ndarray, np.ndarray]:
    ...

def apply_jacobian(val : np.ndarray | None, cov : np.ndarray | None, binning : ArbitraryBinning, jac_details : dict):
    wrt = jac_details['wrt']
    radial_coords = jac_details['radial_coords']
    clip_negativeinf = jac_details['clip_negativeinf']
    clip_positiveinf = jac_details['clip_positiveinf']

    if val is None and cov is None:
        raise ValueError("At least one of val or cov must be provided!")
    elif val is not None:
        thelen = val.shape[0]
        thedtype = val.dtype
    elif cov is not None:
        thelen = cov.shape[0]
        thedtype = cov.dtype
    else:
        raise ValueError("This shouldn't be possible. Somehow both val and cov are None, without triggering the earlier check?")

    # get bin edges
    lower_edges = binning.lower_edges()
    upper_edges = binning.upper_edges()

    # clip non-finite edges appropriately
    for key in clip_negativeinf:
        if key in lower_edges:
            edges = lower_edges[key]
            lower_edges[key] = np.where(edges == -np.inf, clip_negativeinf[key], edges)
            
    for key in clip_positiveinf:
        if key in upper_edges:
            edges = upper_edges[key]
            upper_edges[key] = np.where(edges == np.inf, clip_positiveinf[key], edges)

    # error if there are any remaining non-finite edges
    for key in wrt:
        if np.any(~np.isfinite(lower_edges[key])):
            raise ValueError(f"Lower edges for axis {key} contain non-finite values even after clipping!")
    for key in wrt:
        if np.any(~np.isfinite(upper_edges[key])):
            raise ValueError(f"Upper edges for axis {key} contain non-finite values even after clipping!")

    # compute bin area
    # for radial coordinates the jacobian is 
    # proportional to r^2
    # whereas for other coordinates 
    # it's proportional to the bin width
    widths = {}
    for key in wrt:
        if key in radial_coords:
            widths[key] = np.square(upper_edges[key]) - np.square(lower_edges[key])
        else:
            widths[key] = upper_edges[key] - lower_edges[key]

    jacobian = np.ones(shape = (thelen,), dtype= thedtype)
    for key in wrt:
        jacobian *= widths[key].ravel()

    jacobian[jacobian == 0] = 1.0 #avoid division by zero

    if val is not None:
        density_val = val / jacobian

    if cov is not None:
        cov_jacobian = np.outer(jacobian, jacobian)
        density_cov = cov / cov_jacobian

    if val is None and cov is None:
        raise ValueError("This shouldn't be possible. Somehow both val and cov are None?")
    elif val is None:
        return density_cov # pyright: ignore[reportPossiblyUnboundVariable]
    elif cov is None:
        return density_val # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        return density_val, density_cov # pyright: ignore[reportPossiblyUnboundVariable]


@overload
def normalize_per_block(val : np.ndarray, cov : np.ndarray, binning : ArbitraryBinning, axes : List[str]) -> tuple[np.ndarray, np.ndarray]:
    ...

@overload
def normalize_per_block(val : np.ndarray, cov : None, binning : ArbitraryBinning, axes : List[str]) -> np.ndarray:
    ...

def normalize_per_block(val : np.ndarray, cov : np.ndarray | None, binning : ArbitraryBinning, axes : List[str]):
    fluxes, shapes, _ = binning.get_fluxes_shapes(val, axes)

    if cov is not None:
        _, covshapes, _ = binning.get_fluxes_shapes_cov2d(fluxes, shapes, cov,  axes)
        return shapes, covshapes
    else:
        return shapes

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[True]) -> tuple[np.ndarray, np.ndarray]:
    ...

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[False]) -> np.ndarray:
    ...

def smart_inverse(matrix : np.ndarray, return_eigenspectrum : bool):
    import fasteigenpy as eigen

    print("Calling smart_inverse")
    
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
    
    eigvals[eigvals < 0] = 0.0 # clip negative eigenvalues

    denom = np.where(eigvals == 0, 1, eigvals) # prevent division by zero
    inveigvals = 1.0 / denom
    inveigvals[eigvals == 0] = 0.0 # set back to zero
    
    inverse = eigvecs @ np.diag(inveigvals) @ eigvecs.T

    # inverse is currently the inverse of the correlation matrix. We need to convert it back to the inverse of the covariance matrix
    inverse = np.diag(inverr) @ inverse @ np.diag(inverr)

    if return_eigenspectrum:
        return inverse, eigvals
    else:
        return inverse