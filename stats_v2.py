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
def divide_out_profile(val : np.ndarray, cov : np.ndarray, binning : ArbitraryBinning, axes : List[str]) -> tuple[np.ndarray, np.ndarray]:
    ...

@overload
def divide_out_profile(val : np.ndarray, cov : None, binning : ArbitraryBinning, axes : List[str]) -> np.ndarray:
    ...

def divide_out_profile(val : np.ndarray, cov : np.ndarray | None, binning : ArbitraryBinning, axes : List[str]):
    fluxes, shapes, _ = binning.get_fluxes_shapes(val, axes)
    blocks = binning.get_blocks(axes)
    lenfactor = np.empty_like(shapes)
    for block in blocks:
        shapeblock = shapes[block['slice']]
        lenfactor[block['slice']] = len(shapeblock)
    
    shapes *= lenfactor
    
    if cov is not None:
        _, covshapes, _ = binning.get_fluxes_shapes_cov2d(fluxes, shapes, cov, axes)
        covshapes *= np.outer(lenfactor, lenfactor)
        return shapes, covshapes
    else:
        return shapes

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

def _transform_and_decompose(matrix : np.ndarray):
    import fasteigenpy as eigen

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

    return corr, err, inverr, eigvals, eigvecs

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[True]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

@overload
def smart_inverse(matrix : np.ndarray, return_eigenspectrum : Literal[False]) -> np.ndarray:
    ...

def smart_inverse(matrix : np.ndarray, return_eigenspectrum : bool):
    corr, err, inverr, eigvals, eigvecs = _transform_and_decompose(matrix)
    
    eigvals[eigvals < 0] = 0.0 # clip negative eigenvalues

    denom = np.where(eigvals == 0, 1, eigvals) # prevent division by zero
    inveigvals = 1.0 / denom
    inveigvals[eigvals == 0] = 0.0 # set back to zero
    
    inverse = eigvecs @ np.diag(inveigvals) @ eigvecs.T

    # inverse is currently the inverse of the correlation matrix. 
    # We need to convert it back to the inverse of the covariance matrix
    inverse = np.diag(inverr) @ inverse @ np.diag(inverr)

    if return_eigenspectrum:
        return inverse, eigvals, eigvecs
    else:
        return inverse
    
def smart_sqrt(matrix : np.ndarray):
    corr, err, inverr, eigvals, eigvecs = _transform_and_decompose(matrix)

    eigvals[eigvals < 0] = 0.0 # clip negative eigenvalues
    
    sqrt_eigvals = np.sqrt(eigvals)

    denom = np.where(sqrt_eigvals == 0, 1, sqrt_eigvals) # prevent division by zero
    sqrt_inv_eigvals = 1/denom
    sqrt_inv_eigvals[sqrt_eigvals == 0] = 0.0 # set back to zero
    print("min(sqrt_eigvals):", np.min(sqrt_eigvals))
    print("max(sqrt_eigvals):", np.max(sqrt_eigvals))
    print("min(sqrt_inv_eigvals):", np.min(sqrt_inv_eigvals))
    print("max(sqrt_inv_eigvals):", np.max(sqrt_inv_eigvals))

    L = eigvecs @ np.diag(sqrt_eigvals)
    Linv = eigvecs @ np.diag(sqrt_inv_eigvals)

    # L is currently the sqrt of the correlation matrix. 
    # We need to convert it back to the sqrt of the covariance matrix
    L = np.diag(err) @ L
    Linv = np.diag(inverr) @ Linv

    return L, Linv

def multivariate_gaussian_rvs(mu, L, Nsamples):
    standard_normal = np.random.normal(size=(mu.shape[0], Nsamples))

    result = (mu[:, None] + L @ standard_normal).T

    # test
    result_diff = result - mu[None, :]
    cov_estimate = result_diff.T @ result_diff / (Nsamples - 1)
    print()
    print("Estimated covariance from samples:", cov_estimate.sum())
    print("Original covariance from L:", (L @ L.T).sum())
    print("L.sum()", L.sum())
    print()

    return result
