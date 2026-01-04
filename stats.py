import fasteigenpy as eigen
import numpy as np
import scipy

def marginalize(x, invhess, slice_start, slice_end):
    '''
    It's trivial to marginalize a multivariate Gaussian
    Just slice out the dimensions you want to marginalize over

    NB "marginalize" = "profile" = "let float"
    '''
    before_dim0 = invhess[:slice_start, :]
    after_dim0 = invhess[slice_end:, :]
    tmp0 = np.concatenate((before_dim0, after_dim0), axis=0)

    before_dim1 = tmp0[:, :slice_start]
    after_dim1 = tmp0[:, slice_end:]

    newhess = np.concatenate((before_dim1, after_dim1), axis=1)

    newx = np.concatenate((x[:slice_start], x[slice_end:]), axis=0)

    return newx, newhess


def condition(x, invhess, slice_start, slice_end, values):
    '''
    This requires a bit more math. 
    I'm copying the math from https://www.wikiwand.com/en/articles/Multivariate_normal_distribution#Marginal_distributions

    NB "condition" means "set to a given value".
    If we conditionally set a nuisance to zero that's 
    equivalent to if we never had it, up to the gaussian assumption
    '''
    print("Conditioning...")
    xkeep = np.concatenate((x[:slice_start], x[slice_end:]), axis=0)
    xkill = x[slice_start:slice_end]

    Hbefore_dim0 = invhess[:slice_start, :]
    Hafter_dim0 = invhess[slice_end:, :]
    Htmp0 = np.concatenate((Hbefore_dim0, Hafter_dim0), axis=0)

    Hbefore_dim1 = Htmp0[:, :slice_start]
    Hafter_dim1 = Htmp0[:, slice_end:]

    H11 = np.concatenate((Hbefore_dim1, Hafter_dim1), axis=1)
    H12 = Htmp0[:, slice_start:slice_end]
    H22 = invhess[slice_start:slice_end, slice_start:slice_end]

    print("H11 shape:", H11.shape)
    print("H12 shape:", H12.shape)
    print("H22 shape:", H22.shape)

    codH22 = eigen.CompleteOrthogonalDecomposition(H22)

    solved = codH22.solve(values - xkill)
    print("solved shape:", solved.shape)
    if len(xkill) == 1:
        newx = xkeep + H12.squeeze() * codH22.solve(values - xkill).squeeze()
    else:
        newx = xkeep + H12 @ codH22.solve(values - xkill).squeeze()

    solved = codH22.solve(H12.T)
    if len(solved.shape) == 1:
        solved = solved[None, :]
    newhess = H11 - H12 @ solved
    return newx, newhess

def nuisance_impact(x, invhess, whichnuisance):
    xmask = np.ones(invhess.shape[0], dtype=bool)
    xmask[whichnuisance] = False
    x1 = x[xmask]
    x2 = x[whichnuisance]

    H11mask = np.ones(invhess.shape[0], dtype=bool)
    H11mask[whichnuisance] = False
    H11 = invhess[H11mask, :][:, H11mask]

    H22 = invhess[whichnuisance, whichnuisance]

    H12 = invhess[whichnuisance, :]
    H12mask = np.ones(invhess.shape[0], dtype=bool)
    H12mask[whichnuisance] = False
    H12 = H12[H12mask]

    H21 = invhess[:, whichnuisance]
    H21mask = np.ones(invhess.shape[0], dtype=bool)
    H21mask[whichnuisance] = False
    H21 = H21[H21mask]

    H12 = H12[:, None]
    H21 = H21[None, :]

    xshift = np.squeeze((1/H22) * H12 * x2)
    
    H11shift = - (1/H22) * H12 @ H21

    return xshift, H11shift, x1 + xshift, H11 + H11shift

def multivariate_gaussian_rvs(mu, L, Nsamples):
    standard_normal = np.random.normal(size=(mu.shape[0], Nsamples))

    return (mu[:, None] + L @ standard_normal).T

def regularized_inverse(matrix, l, force_positive=True, wrt_corr=True):
    if wrt_corr:
        err = np.sqrt(np.diag(matrix))
        err[err == 0] = 1
        inverr = 1 / err
        C = np.diag(inverr) @ matrix @ np.diag(inverr)
    else:
        C = matrix

    regmat = np.where(C < l, 0, C)

    _, inverse, _ = inverse_and_eigenspectrum(
        regmat, 
        clip_lowest_N=0,
        force_positive=False,
        return_sqrt=False,
        wrt_corr=wrt_corr
    )

    if wrt_corr:
        inverse = np.diag(inverr) @ inverse @ np.diag(inverr)

    return inverse

def inverse_and_eigenspectrum(matrix, 
                              clip_lowest_N=0,
                              force_positive=True,
                              return_sqrt=False,
                              wrt_corr=True):
    print("Computing Eigendecomposition...")
    if wrt_corr:
        err = np.sqrt(np.diag(matrix))
        err[err == 0] = 1
        inverr = 1 / err
        corr = np.diag(inverr) @ matrix @ np.diag(inverr)

        solver = eigen.SelfAdjointEigenSolver(corr)
    else:
        solver = eigen.SelfAdjointEigenSolver(matrix)

    if solver.info() != eigen.ComputationInfo.Success:
        print("Eigen decomposition failed")
        print(solver.info())
        raise RuntimeError("Eigen decomposition failed")

    print("Inverting...")
    if return_sqrt:
        inverse, reconstructed, sqrt_inverse, sqrt_reconstructed = inverse_from_eigenspectrum(
            solver, 
            clip_lowest_N=clip_lowest_N,
            force_positive=force_positive,
            return_sqrt=True
        )
    else:
        inverse, reconstructed = inverse_from_eigenspectrum(
            solver, 
            clip_lowest_N=clip_lowest_N,
            force_positive=force_positive,
            return_sqrt=False
        )

    if wrt_corr:
        inverse = np.diag(inverr) @ inverse @ np.diag(inverr)
        reconstructed = np.diag(err) @ reconstructed @ np.diag(err)
        if return_sqrt:
            sqrt_inverse = np.diag(inverr) @ sqrt_inverse 
            sqrt_reconstructed = np.diag(err) @ sqrt_reconstructed 

    if return_sqrt:
        return solver, inverse, reconstructed, sqrt_inverse, sqrt_reconstructed
    else:
        return solver, inverse, reconstructed

def inverse_from_eigenspectrum(solver, 
                               clip_lowest_N=0,
                               force_positive=True,
                               return_sqrt=False):
    eigenvals = solver.eigenvalues().copy()
    eigenvecs = solver.eigenvectors()

    if force_positive:
        eigenvals = np.where(eigenvals < 0, 0, eigenvals)

    if clip_lowest_N > 0:
        eigenvals[:clip_lowest_N] = 0

    denom = np.where(eigenvals == 0, 1, eigenvals)
    inv_eigenvals = 1 / denom

    inv_eigenvals = np.where(eigenvals == 0, 0, inv_eigenvals)

    reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    inverse = eigenvecs @ np.diag(inv_eigenvals) @ eigenvecs.T
        
    if return_sqrt:
        sqrt_inv_eigenvals = np.sqrt(inv_eigenvals)
        sqrt_eigenvals = np.sqrt(eigenvals)

        sqrt_reconstructed = eigenvecs @ np.diag(sqrt_eigenvals)
        sqrt_inverse = eigenvecs @ np.diag(sqrt_inv_eigenvals)

        return inverse, reconstructed, sqrt_inverse, sqrt_reconstructed
    else:
        return inverse, reconstructed

def get_chi2(vals1, vals2, cov1, cov2, binning=None, cut=None, normalize=False):
    if normalize:
        vals1, cov1 = normalize_distribution(vals1, cov1)
        vals2, cov2 = normalize_distribution(vals2, cov2)

    covdiff = cov1 + cov2
    diff = vals1 - vals2

    if cut is not None:
        diff = binning.get_slice(diff.T, **cut).T
        covdiff = binning.get_slice(covdiff.T, **cut)
        covdiff = binning.get_slice(covdiff.T, **cut)

    #_, invcov, _ = inverse_and_eigenspectrum(
    #        covdiff,
    #        clip_lowest_N=0,
    #        force_positive=True,
    #        return_sqrt=False,
    #        wrt_corr=True
    #)
    err = np.sqrt(np.diag(covdiff))
    err[err == 0] = 1
    cdiff = np.diag(1/err) @ covdiff @ np.diag(1/err)
    codcdiff = eigen.CompleteOrthogonalDecomposition(cdiff) 
    invc = codcdiff.pseudoInverse()
    invcov = np.diag(1/err) @ invc @ np.diag(1/err)

    chi2 = diff @ invcov @ diff
    print("CHI2",chi2)

    return chi2, len(diff)

def normalize_distribution(vals, cov, return_N = False):
    '''
    Covariance of normalized distribution is worked out in 
    the statistical reference section of the AN. 
    The correlation between each bin and the normalization factor
    results in a small decrease in uncertainty
    '''
    N = vals.sum()

    result_vals = vals/N

    sumcov_dim0 = np.sum(cov, axis=0)
    sumcov_dim1 = np.sum(cov, axis=1)
    sumcov_total = np.sum(cov)

    result_cov_term1 = cov 
    result_cov_term2 = -np.outer(result_vals, sumcov_dim0) 
    result_cov_term3 = -np.outer(sumcov_dim1, result_vals) 
    result_cov_term4 = np.outer(result_vals, result_vals) * sumcov_total 

    result_cov = (result_cov_term1 +
                  result_cov_term2 +
                  result_cov_term3 +
                  result_cov_term4) / (N * N)

    if return_N:
        return result_vals, result_cov, N
    else:
        return result_vals, result_cov

def conormalize_distributions(vals1, cov1, vals2, cov2, cov12):
    result_vals1, result_cov1, N1 = normalize_distribution(vals1, cov1, return_N=True)
    result_vals2, result_cov2, N2 = normalize_distribution(vals2, cov2, return_N=True)

    if cov12 is not None:
        sumcov_dim0 = np.sum(cov12, axis=0)
        sumcov_dim1 = np.sum(cov12, axis=1)
        sumcov_total = np.sum(cov12)

        result_cov12_term1 = cov12 
        result_cov12_term2 = -np.outer(result_vals1, sumcov_dim0)
        result_cov12_term3 = -np.outer(sumcov_dim1, result_vals2)
        result_cov12_term4 = np.outer(result_vals1, result_vals2) * sumcov_total

        result_cov12 = (result_cov12_term1 +
                        result_cov12_term2 +
                        result_cov12_term3 +
                        result_cov12_term4) / (N1 * N2)
    else:
        result_cov12 = None

    return result_vals1, result_cov1, result_vals2, result_cov2, result_cov12

def sum_distribution(vals1, cov1, vals2, cov2, cov12):
    result_vals = vals1 + vals2

    result_cov = cov1 + cov2
    if cov12 is not None:
        result_cov += cov12 + cov12.T

    return result_vals, result_cov

def difference_distribution(vals1, cov1, vals2, cov2, cov12):
    result_vals = vals1 - vals2

    result_cov = cov1 + cov2
    if cov12 is not None:
        result_cov -= cov12 + cov12.T

    return result_vals, result_cov

def product_distribution(vals1, cov1, vals2, cov2, cov12):
    result_vals = vals1 * vals2

    result_cov = np.outer(vals1, vals1) * cov2 + np.outer(vals2, vals2) * cov1 
    if cov12 is not None:
        term = np.outer(vall2, vals1) * cov12
        result_cov += term + term.T

    return result_vals, result_cov

def quotient_distribution(vals1, cov1, vals2, cov2, cov12):
    result_vals = vals1 / vals2

    vals2denom = np.outer(1/vals2, 1/vals2)
    term1 = vals2denom * cov1
    term2 = vals2denom * vals2denom * np.outer(vals1, vals2) * cov2
    result_cov = term1 + term2

    if cov12 is not None:
        term3 = vals2denom * result_vals[None, :] * cov12
        result_cov -= term3 + term3.T

    return result_vals, result_cov

def flux_and_shape_covariance(data, covyy, covty, binning, axes):
    blocks = binning.get_blocks(axes)
    fluxes, shapes, fluxbinning = binning.get_fluxes_shapes(data, axes)

    Nflux = len(blocks)

    covshapes = np.zeros(covyy.shape, dtype=covyy.dtype)
    covflux = np.zeros((Nflux, Nflux), dtype=covyy.dtype)
    covfluxshape = np.zeros((Nflux, covyy.shape[0]), dtype=covyy.dtype)
    
    if covty is not None:
        covtshapes = np.zeros((covty.shape[0], covyy.shape[0]), dtype=covyy.dtype)
        covtflux = np.zeros((covty.shape[0], Nflux), dtype=covyy.dtype)

    #mapping from shape to flux
    fluxindex = np.zeros(covyy.shape[0], dtype=np.int32)
    for i, block in enumerate(blocks):
        fluxindex[block['slice']] = i

    #covflux
    for a, blockA in enumerate(blocks):
        sliceA = blockA['slice']
        for b, blockB in enumerate(blocks):
            sliceB = blockB['slice']

            covflux[a, b] = np.sum(covyy[sliceA, :][:, sliceB])

    #covfluxshape
    for a, blockA in enumerate(blocks):
        sliceA = blockA['slice']

        for b, blockB in enumerate(blocks):
            sliceB = blockB['slice']
            
            covfluxshape[a, sliceB] += np.sum(covyy[sliceA, sliceB], axis=0) / fluxes[b]

            covfluxshape[a, sliceB] -= (shapes[sliceB]/fluxes[b])  * np.sum(covyy[sliceA, :][:, sliceB], axis=None)

    #covshape
    for a, blockA in enumerate(blocks):
        sliceA = blockA['slice']

        for b, blockB in enumerate(blocks):
            sliceB = blockB['slice']

            covshapes[sliceA, :][:, sliceB] += covyy[sliceA, :][:, sliceB] / (fluxes[a] * fluxes[b])

            covshapes[sliceA, :][:, sliceB] += (np.outer(shapes[sliceA], shapes[sliceB]) / (fluxes[a] * fluxes[b])) * np.sum(covyy[sliceA, :][:, sliceB], axis=None)

            covshapes[sliceA, :][:, sliceB] -= np.outer(shapes[sliceA]/(fluxes[a] * fluxes[b]), np.sum(covyy[sliceA, :][:, sliceB], axis=0))
            covshapes[sliceA, :][:, sliceB] -= np.outer(np.sum(covyy[sliceA, :][:, sliceB], axis=1), shapes[sliceB]/(fluxes[a] * fluxes[b]))

    if covty is not None:
        #covty
        for a, blockA in enumerate(blocks):
            sliceA = blockA['slice']

            covtflux[:, a] = np.sum(covty[:, sliceA], axis=1)

        #covtshapes
        for a, blockA in enumerate(blocks):
            sliceA = blockA['slice']

            covtshapes[:, sliceA] += covty[:, sliceA] / fluxes[a]
            covtshapes[:, sliceA] -= np.outer(np.sum(covty[:, sliceA], axis=1),
                                              shapes[sliceA] / fluxes[a])
    else:
        covtflux = None
        covtshapes = None

    return fluxes, shapes, covflux, covshapes, covfluxshape, covtflux, covtshapes, fluxbinning

def compute_flux(vals, cov, binning, ptslice, Rslice, 
                 normalize=True, jacobian=True):

    flux, block = binning.get_continuous_slice(vals, pt=ptslice, R=Rslice)
    covflux = binning.get_slice_cov2d(cov, pt=ptslice, R=Rslice)

    r_edges = np.asarray(block.ax_details['r']['edges'])
    c_edges = np.asarray(block.ax_details['c']['edges'])

    if normalize:
        flux, covflux = normalize_distribution(flux, covflux)

    if jacobian:
        jac = 0.5 * (r_edges[1:]**2 - r_edges[:-1]**2)[:, None] \
                * (c_edges[1:] - c_edges[:-1])[None, :]
        jac = jac.ravel()

        flux = flux/jac
        covflux = covflux/np.outer(jac, jac)

    return flux, covflux, r_edges, c_edges

def angular_averaged_flux(vals, cov, binning, ptslice, Rslice,
                           normalize=True, jacobian=True):

    flux, covflux, r_edges, c_edges = compute_flux(
        vals, cov, binning, ptslice, Rslice,
        normalize=normalize, jacobian=jacobian
    )

    flux = flux.reshape((len(r_edges)-1, len(c_edges)-1))
    covflux = covflux.reshape((*flux.shape, *flux.shape))

    fluxsum = flux.sum(axis=1)
    covsum = covflux.sum(axis=(1,3))

    N = flux.shape[1]
    fluxavg = fluxsum / N
    covavg = covsum / (N * N)

    return fluxavg, covavg, r_edges, c_edges

def radial_summed_flux(vals, cov, binning, ptslice, Rslice, 
                       normalize=True, jacobian=True):

    flux, covflux, r_edges, c_edges = compute_flux(
        vals, cov, binning, ptslice, Rslice,
        normalize=normalize, jacobian=jacobian
    )

    flux = flux.reshape((len(r_edges)-1, len(c_edges)-1))
    covflux = covflux.reshape((*flux.shape, *flux.shape))

    fluxsum = flux.sum(axis=0)
    covsum = covflux.sum(axis=(0,2))

    return fluxsum, covsum, r_edges, c_edges

def radial_slice_flux(vals, cov, binning, ptslice, Rslice, rbin,
                      normalize=True, jacobian=True):
    flux, covflux, r_edges, c_edges = compute_flux(
        vals, cov, binning, ptslice, Rslice,
        normalize=normalize, jacobian=jacobian
    )

    flux = flux.reshape((len(r_edges)-1, len(c_edges)-1))
    covflux = covflux.reshape((*flux.shape, *flux.shape))

    fluxsum = flux[rbin,:]
    covsum = covflux[rbin, :, rbin, :]

    print(fluxsum.shape)
    print(covsum.shape)

    fluxsum, covsum = normalize_distribution(fluxsum, covsum)

    return fluxsum, covsum, r_edges, c_edges
