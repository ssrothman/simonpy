# simonpy

A Python utility library for HEP analysis, providing common tools for binning, statistics, coordinates, and data manipulation.

## Overview

`simonpy` is a collection of frequently-used utility functions and classes for High Energy Physics analysis, with a focus on binning schemes, statistical computations, and array operations.

## Modules

### AbitraryBinning.py

Implements flexible, non-rectangular binning schemes for histograms, useful for unfolding and complex analysis geometries.

**Key Classes:**
- `_BinningBlock`: Internal representation of contiguous rectangular bin blocks
- `ArbitraryBinning`: Wraps a single `hist.Hist` into an arbitrary binning scheme
- `ArbitraryGenRecoBinning`: Handles generator-level and reconstructed-level binning for unfolding studies

**Features:**
- Create binning from `hist.Hist` histograms
- Serialize/deserialize to/from JSON or dict
- Support for non-rectangular bin geometries
- Equality comparison between binning schemes

### akutil.py

Utilities for working with Awkward arrays.

**Functions:**
- `unflatMatrix(arr, nrows, ncols)`: Unflatten 1D array into ragged 2D matrices
- `unflatVector(arr, ncols)`: Unflatten 1D array into ragged vectors

### coordinates.py

Coordinate transformation utilities for HEP data.

**Functions:**
- `xyz_to_eta_phi(x, y, z)`: Convert Cartesian coordinates to (η, φ)
- `eta_to_theta(eta)`: Convert pseudorapidity to polar angle θ

### dictmerge.py

Dictionary utilities for configuration management.

**Functions:**
- `merge_dict(original, update, allow_new_keys, replace_dict)`: Recursively merge two dictionaries with type checking
- `accumulate_dict(original, update)`: Accumulate numerical values in nested dictionaries

### sanitization.py

Data validation and transformation utilities.

**Functions:**
- `maybe_valcov_to_definitely_valcov(evaluated)`: Normalize histogram/covariance inputs to consistent format
- `ensure_same_length(*args)`: Broadcast list arguments to common length
- `all_same_key(things, skip)`: Check if objects share a common key attribute

### stats.py

Statistical utilities for fitting and covariance manipulation.

**Functions:**
- `marginalize(x, invhess, slice_start, slice_end)`: Marginalize over Gaussian nuisance parameters
- `condition(x, invhess, slice_start, slice_end, values)`: Condition Gaussian on fixed parameter values
- `nuisance_impact(x, invhess, whichnuisance)`: Compute impact of single nuisance parameter
- `multivariate_gaussian_rvs(mu, L, Nsamples)`: Sample from multivariate Gaussian distribution

### stats_v2.py

Extended statistical functionality (see file for details).

### text.py

String processing utilities for physics notation.

**Functions:**
- `clean_string(s)`: Clean LaTeX and unit notation from strings
- `strip_units(s)`: Remove unit brackets from strings
- `strip_dollar_signs(s)`: Remove LaTeX delimiters
- `attempt_regex_match(pattern, axiskey)`: Match axis keys with wildcard patterns
- `find_match(keys, patterns, ignore_case)`: Find matching pattern for a key

## Dependencies

- `numpy`
- `hist` (for binning support)
- `awkward` (for array utilities)
- `scipy` (for statistics)
- `fasteigenpy` (for eigen decomposition in statistics)

## Notes

- `AbitraryBinning` is the primary class for complex binning geometries
- Statistics functions assume Gaussian uncertainties (Hessian-based approach)
- Text utilities support LaTeX and physics notation cleaning
- Dictionary merging enforces strict type matching to prevent configuration errors
