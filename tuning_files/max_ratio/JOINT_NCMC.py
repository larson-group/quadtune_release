#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint Noise Covariance Matrix Construction
==========================================
Extends NCMC_IND.py to compute cross-variable covariance blocks
and assemble a joint covariance matrix across all variables.

Workflow:
  1. For each member: load monthly regional data for all variables
  2. Compute per-variable covariance blocks (Sigma_SS, Sigma_LL, ...)
  3. Compute cross-variable covariance blocks (Sigma_SL, Sigma_SP, ...)
  4. Assemble joint covariance matrix Sigma_joint (n_vars*N x n_vars*N)
  5. Add measurement error + joint SVD truncation
  6. Save joint Gamma, Gamma_reg, Gamma_inv per member

Output:
  noise_covariance_joint/
    ├── noise_covariance_joint_member1.nc
    ├── noise_covariance_joint_member2.nc
    ├── ...
"""

import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')


# ====================================================================
# Configuration
# ====================================================================

class Config:
    BASE_DIR = "./"
    MONTHLY_DIR = os.path.join(BASE_DIR, "regional_output_monthly")
    OUTPUT_DIR = os.path.join(BASE_DIR, "noise_covariance_joint")

    # Grid parameters
    INTLL = 15
    EDGELAT = 12
    EDGELON = 24
    N_REGIONS = EDGELAT * EDGELON  # 288

    # Ensemble members
    MEMBERS = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33]

    # Variables (order matters — defines block layout)
    VARS = ['SWCF', 'LWCF', 'PRECT','TMQ']
    N_VARS = len(VARS)

    GLOBAL_MEANS_OBS = {
        'SWCF': -45.30901732406915,
        'LWCF': 25.81230998182771,
        'PRECT': 2.6904643040278655,
        'TMQ':24.39980138451228
    }
    N_TOTAL = N_REGIONS * N_VARS  # 1152

    # Experiment prefix
    EXP_PREFIX = "20260310"

    # Measurement error fraction
    MEASUREMENT_ERROR_FRACTION = 0.1

    # Howland et al. (2007) Eq. 11 coefficient
    HOWLAND_C = 0.2 

    # SVD truncation threshold
    VARIANCE_THRESHOLD = 0.99

    # Numerical floor for error std
    MIN_ERROR_STD = 1e-6


# ====================================================================
# Load monthly data for a member
# ====================================================================

def load_member_monthly_data(member, var):
    """Load monthly data for a single member and variable."""
    file_path = os.path.join(
        Config.MONTHLY_DIR,
        f'{Config.EXP_PREFIX}_{member}_monthly_regional.nc'
    )
    if not os.path.exists(file_path):
        return None

    ds = xr.open_dataset(file_path)
    data = ds[var].values if var in ds.variables else None
    ds.close()
    return data


def load_all_variables(member):
    """
    Load and normalize all variables for a member using hardcoded global means.
    """
    
    data_norm = {}
    global_means = {}

    for var in Config.VARS:
        data = load_member_monthly_data(member, var)
        if data is None:
            print(f"      ERROR: No data for {var}")
            return None, None, False

        gm = np.abs(Config.GLOBAL_MEANS_OBS[var])
        
        data_norm[var] = data / gm
        global_means[var] = gm

    return data_norm, global_means, True


# ====================================================================
# Covariance blocks
# ====================================================================

def compute_auto_covariance(data_norm):
    """
    Compute auto-covariance block for one variable using deseasonalized anomalies.
    """
    M, N = data_norm.shape

    # Long-term mean (MUST be preserved for Delta calculation later!)
    data_mean = np.nanmean(data_norm, axis=0)

    # Deseasonalization: Remove the deterministic seasonal cycle
    if M % 12 == 0:
        n_years = M // 12
        # Reshape array into (n_years, 12 months, N regions)
        data_reshaped = data_norm.reshape(n_years, 12, N)
        # Compute monthly climatology (mean over all years for each month)
        monthly_clim = np.nanmean(data_reshaped, axis=0) # Shape: (12, N)
        # Subtract monthly climatology from raw data and flatten back
        data_centered = (data_reshaped - monthly_clim).reshape(M, N)
    else:
        # Fallback only if data is not in full years
        print("      WARNING: Time dimension is not a multiple of 12. Falling back to global mean subtraction.")
        data_centered = data_norm - data_mean
    

    

        
    data_centered = np.nan_to_num(data_centered, nan=0.0)
    
    # Compute covariance of the pure noise
    Sigma = np.dot(data_centered.T, data_centered) / (M - 1)
    
    # Return the uncentered data_mean so build_joint_measurement_error() functions correctly
    return Sigma, data_mean, data_centered


def compute_cross_covariance(data_S_centered, data_L_centered):
    """
    Compute cross-covariance block using deseasonalized data.
    """
    M = data_S_centered.shape[0]
    # Simple multiplication of the deseasonalized anomalies
    Sigma_SL = np.dot(data_S_centered.T, data_L_centered) / (M - 1)
    return Sigma_SL


# ====================================================================
# Assemble joint covariance matrix
# ====================================================================

def assemble_joint_covariance(data_norm):
    """
    Assemble joint covariance matrix from all variables.

    Parameters:
        data_norm: dict {var: (M, N) normalized array}

    Returns:
        Sigma_joint: (N_TOTAL, N_TOTAL)
        data_means: dict {var: (N,)}
        diagnostics: dict with block info
    """
    n_vars = Config.N_VARS
    N = Config.N_REGIONS
    N_total = Config.N_TOTAL

    # Storage for blocks
    auto_blocks = {}   # {var: (N, N)}
    data_means = {}    # {var: (N,)}
    centered = {}      # {var: (M, N)}

    # Step 1: auto-covariance blocks
    for var in Config.VARS:
        Sigma, mean, centered_data = compute_auto_covariance(data_norm[var])
        auto_blocks[var] = Sigma
        data_means[var] = mean
        centered[var] = centered_data

    # Step 2: assemble joint matrix
    Sigma_joint = np.zeros((N_total, N_total), dtype=np.float64)

    for i, var_i in enumerate(Config.VARS):
        for j, var_j in enumerate(Config.VARS):
            row_slice = slice(i * N, (i + 1) * N)
            col_slice = slice(j * N, (j + 1) * N)

            if i == j:
                # Auto-covariance block
                Sigma_joint[row_slice, col_slice] = auto_blocks[var_i]
            elif i < j:
                # Cross-covariance block (compute once)
                Sigma_cross = compute_cross_covariance(
                    centered[var_i], centered[var_j]
                )
                Sigma_joint[row_slice, col_slice] = Sigma_cross
                Sigma_joint[col_slice, row_slice] = Sigma_cross.T
            # i > j handled by symmetry above

    # Symmetrize to remove numerical asymmetry
    Sigma_joint = 0.5 * (Sigma_joint + Sigma_joint.T)

    diagnostics = {
        'auto_blocks': auto_blocks,
        'data_means': data_means,
    }
    return Sigma_joint, diagnostics


def build_joint_measurement_error(data_means, auto_blocks):
    """
    Build joint measurement error matrix Delta_joint using Howland et al. Eq. 11.
    """
    N = Config.N_REGIONS
    N_total = Config.N_TOTAL
    Delta_joint = np.zeros((N_total, N_total), dtype=np.float64)
    delta_per_var = {}

    C = Config.HOWLAND_C
    Cm = Config.MEASUREMENT_ERROR_FRACTION

    # Define physical boundaries for each variable (lower, upper). None means unbonded.
    bounds = {
        'SWCF': (None, None),
        'LWCF': (None, None),
        'PRECT': (0.0, None),
        'TMQ': (0.0, None)
    }

    for i, var in enumerate(Config.VARS):
        mean = data_means[var]
        sigma_block = auto_blocks[var]
        sigma_kk = np.diag(sigma_block) 
        

        ci_half_width = 2.0 * np.sqrt(np.maximum(sigma_kk, 0.0))
        
        upper_bound_val = mean + ci_half_width
        lower_bound_val = mean - ci_half_width
        
        lower_lim, upper_lim = bounds.get(var, (None, None))
        
        dist_lower = np.full_like(mean, np.inf)
        dist_upper = np.full_like(mean, np.inf)
        
        if lower_lim is not None:
            dist_lower = np.abs(lower_bound_val - lower_lim)
            dist_upper = np.abs(upper_bound_val - lower_lim)
            dist_lower[lower_bound_val < lower_lim] = 0.0
            dist_upper[upper_bound_val < lower_lim] = 0.0
            
        if upper_lim is not None:
            d_l = np.abs(lower_bound_val - upper_lim)
            d_u = np.abs(upper_bound_val - upper_lim)
            d_l[lower_bound_val > upper_lim] = 0.0
            d_u[upper_bound_val > upper_lim] = 0.0
            dist_lower = np.minimum(dist_lower, d_l)
            dist_upper = np.minimum(dist_upper, d_u)
            
        min_dist = np.minimum(dist_lower, dist_upper)
        
        # Term 1: C * min distance to boundaries
        term1 = C * min_dist
        
        # Term 2: Cm * data value (percentage cap)
        term2 = Cm * np.abs(mean)
        
        # Handle unbounded variables where min_dist is inf
        term1 = np.where(np.isinf(term1), term2, term1)
        err_std = np.minimum(term1, term2)
        err_std = np.maximum(err_std, Config.MIN_ERROR_STD)


        var_diag = err_std ** 2
        delta_per_var[var] = var_diag
        
        row_slice = slice(i * N, (i + 1) * N)
        Delta_joint[row_slice, row_slice] = np.diag(var_diag)

    return Delta_joint, delta_per_var


# ====================================================================
# Joint SVD regularization
# ====================================================================

def regularize_joint_svd(Gamma_joint, threshold=0.99):
    """
    Apply Tikhonov regularization to the joint covariance matrix and 
    normalize the trace of the inverse to maintain global scaling.
    """
    U, s, Vt = np.linalg.svd(Gamma_joint)
    total_var = np.sum(s)
    cum_var = np.cumsum(s) / total_var
    n_components = int(np.searchsorted(cum_var, threshold) + 1)
    n_components = min(n_components, len(s))


    raw_floor = s[n_components - 1] if n_components <= len(s) else s[-1]
    min_safe_floor = 1e-2 * (total_var / len(s))
    lambda_sq = max(raw_floor, min_safe_floor)
    
    s_reg = s + lambda_sq

    print(f"    Joint SVD: {len(s)} singular values, "
          f"Tikhonov floor (lambda^2) = {lambda_sq:.2e} based on {cum_var[n_components-1]*100:.2f}% variance")


    Gamma_reg = (U * s_reg) @ Vt
    Gamma_inv = (U / s_reg) @ Vt
    

    Gamma_inv = Gamma_inv * (len(s) / np.trace(Gamma_inv))

    cond = np.linalg.cond(Gamma_reg)
    print(f"    Joint condition number after Tikhonov: {cond:.2e}")

    return {
        'Gamma_reg': Gamma_reg,
        'Gamma_inv': Gamma_inv,
        'singular_values': s,
        'n_components': n_components,
        'condition_number': cond,
        'cumulative_variance': cum_var,
    }


# ====================================================================
# Save
# ====================================================================

def save_joint_covariance(member, Sigma_joint, Delta_joint, reg_results,
                          diagnostics, global_means):
    """Save joint covariance matrices to NetCDF and npz."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    N_total = Config.N_TOTAL
    n_sv = len(reg_results['singular_values'])

    output_nc = os.path.join(
        Config.OUTPUT_DIR,
        f'noise_covariance_joint_member{member}.nc'
    )

    ds = xr.Dataset(
        {
            'Sigma_joint': (['i', 'j'], Sigma_joint),
            'Delta_joint': (['i', 'j'], Delta_joint),
            'Gamma_joint': (['i', 'j'], Sigma_joint + Delta_joint),
            'Gamma_reg': (['i', 'j'], reg_results['Gamma_reg']),
            'Gamma_inv': (['i', 'j'], reg_results['Gamma_inv']),
            'singular_values': (['sv'], reg_results['singular_values']),
            'cumulative_variance': (['sv'], reg_results['cumulative_variance']),
        },
        coords={
            'i': np.arange(N_total),
            'j': np.arange(N_total),
            'sv': np.arange(n_sv),
        },
        attrs={
            'member': member,
            'variables': ','.join(Config.VARS),
            'n_vars': Config.N_VARS,
            'n_regions': Config.N_REGIONS,
            'n_total': N_total,
            'n_components': reg_results['n_components'],
            'condition_number': reg_results['condition_number'],
            'variance_threshold': Config.VARIANCE_THRESHOLD,
            'measurement_error_fraction': Config.MEASUREMENT_ERROR_FRACTION,
        }
    )
    for var in Config.VARS:
        ds.attrs[f'global_mean_{var}'] = float(global_means[var])
    ds.to_netcdf(output_nc)

    # npz (useful for direct loading into optimizer)
    np.savez(
        os.path.join(Config.OUTPUT_DIR,
                     f'noise_covariance_joint_member{member}.npz'),
        Sigma_joint=Sigma_joint,
        Delta_joint=Delta_joint,
        Gamma_joint=Sigma_joint + Delta_joint,
        Gamma_reg=reg_results['Gamma_reg'],
        Gamma_inv=reg_results['Gamma_inv'],
        singular_values=reg_results['singular_values'],
        n_components=reg_results['n_components'],
        variables=np.array(Config.VARS),
        global_means=np.array([global_means[v] for v in Config.VARS]),
    )

    print(f"    Saved: {output_nc}")


def save_block_diagnostics(member, diagnostics):
    """Optionally save the individual auto-covariance blocks for inspection."""
    block_dir = os.path.join(Config.OUTPUT_DIR, 'blocks')
    os.makedirs(block_dir, exist_ok=True)

    for var, block in diagnostics['auto_blocks'].items():
        np.savez(
            os.path.join(block_dir, f'auto_block_{var}_member{member}.npz'),
            Sigma=block,
            mean=diagnostics['data_means'][var],
        )


# ====================================================================
# Per-member workflow
# ====================================================================

def process_member(member):
    """Process one ensemble member: build joint covariance matrix."""
    print(f"\n  {'='*60}")
    print(f"  Member {member}")
    print(f"  {'='*60}")

    # 1. Load and normalize all variables
    data_norm, global_means, valid = load_all_variables(member)
    if not valid:
        print(f"    Skipping member {member}")
        return None

    M = data_norm[Config.VARS[0]].shape[0]
    print(f"    Loaded {M} time samples, {Config.N_REGIONS} regions, "
          f"{Config.N_VARS} variables")

    # 2. Assemble joint covariance Sigma_joint
    print("    Assembling joint covariance matrix...")
    Sigma_joint, diagnostics = assemble_joint_covariance(data_norm)

    # 3. Build joint measurement error Delta_joint
    Delta_joint, delta_per_var = build_joint_measurement_error(
        diagnostics['data_means'], diagnostics['auto_blocks']
    )

    # 4. Total covariance
    Gamma_joint = Sigma_joint + Delta_joint

    # 5. Joint SVD regularization
    reg_results = regularize_joint_svd(
        Gamma_joint, Config.VARIANCE_THRESHOLD
    )

    # 6. Save
    save_joint_covariance(
        member, Sigma_joint, Delta_joint, reg_results,
        diagnostics, global_means
    )
    save_block_diagnostics(member, diagnostics)

    return {
        'member': member,
        'n_samples': M,
        'n_components': reg_results['n_components'],
        'condition_number': reg_results['condition_number'],
        'global_means': global_means,
    }


# ====================================================================
# Summary
# ====================================================================

def print_summary(all_results):
    print("\n" + "="*70)
    print("JOINT COVARIANCE SUMMARY")
    print("="*70)
    print(f"Variables: {Config.VARS}")
    print(f"Joint matrix size: {Config.N_TOTAL} x {Config.N_TOTAL}")
    print(f"{'Member':<10} {'Samples':<10} {'Components':<12} "
          f"{'Cond. Number':<15}")
    print("-"*60)
    for r in all_results:
        if r is None:
            continue
        print(f"{r['member']:<10} {r['n_samples']:<10} "
              f"{r['n_components']:<12} {r['condition_number']:<15.2e}")
    print("="*70)
    print(f"Output: {Config.OUTPUT_DIR}")
    print("="*70)


# ====================================================================
# Main
# ====================================================================

def main():
    print("="*70)
    print("JOINT NOISE COVARIANCE MATRICES (ALL VARIABLES)")
    print("="*70)
    print(f"Variables:  {Config.VARS}")
    print(f"Regions:    {Config.N_REGIONS} per variable")
    print(f"Joint size: {Config.N_TOTAL} x {Config.N_TOTAL}")
    print(f"Members:    {len(Config.MEMBERS)}")
    print(f"Note: Each member has 60 monthly samples -> rank <= 59")
    print("="*70)

    if not os.path.exists(Config.MONTHLY_DIR):
        print(f"ERROR: {Config.MONTHLY_DIR} not found.")
        print("Run NCMC_IND.py step1 first to generate monthly regional data.")
        return

    all_results = []
    for member in Config.MEMBERS:
        result = process_member(member)
        all_results.append(result)

    if any(r is not None for r in all_results):
        print_summary(all_results)

    print("\nDONE!")


if __name__ == "__main__":
    main()
