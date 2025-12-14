# -*- coding: utf-8 -*-

# Run this app with `python3 quadtune_driver.py` and
# view the plots at http://127.0.0.1:8050/ in your web browser.
# (To open a web browser on a larson-group computer,
# login to malan with `ssh -X` and then type `firefox &`.)

"""
In this file, users may specify input data to quadtune_driver.
This includes assigning filenames for input netcdf files,
regional metric weights, and observed values of parameters.
"""

from typing import Callable
import numpy as np
import pandas as pd
import sys


def config_core():
    from set_up_inputs import (
        setUp_x_MetricsList,
        setUpDefaultMetricValsCol, setUp_x_ObsMetricValsDict,
        setUpObsCol,
        calcObsGlobalAvgCol
    )

    # Flag for using bootstrap sampling
    doBootstrapSampling = False

    # doPiecewise = True if using a piecewise linear emulator
    doPiecewise = False

    # Flag for enabling additional output of multiple functions
    beVerbose = False

    # Flag for whether to bound quadtune parameter values to remain within
    # the range spanned by the default and sensitivity runs.
    doSensParamBounds = False

    # L1 regularization coefficient, i.e., penalty on param perturbations in objFnc
    # Increase this value to 0.1 or 0.5 or so if you want to eliminate
    # unimportant parameters.
    reglrCoef = 0.0

    # Non-dimensional pre-factor of penalty term in loss function that penalizes when
    #   the tuner leaves a global-mean bias, i.e., when the residuals don't sum to zero.
    #   Set to 1.0 for a "medium" penalty, and set to 0.0 for no penalty.
    penaltyCoef = np.array([0.0])

    # Use these flags to determine whether or not to create specific plots
    #    in create_nonbootstrap_figs.py
    doCreatePlots = True

    # Flag to enable reading SST4K regional files
    doMaximizeRatio = True

    

    # Set debug level
    debug_level = 1
    # Set perturbation for the recovery test
    recovery_test_dparam = 0.5
    
    #varPrefixes = ['SWCF', 'TMQ', 'LWCF', 'PRECT']
    #varPrefixes = ['SWCF', 'LWCF', 'FSNTC', 'FLNTC']
    varPrefixes = ['SWCF']
    # mapVarIdx is the field is plotted in the 20x20 maps created by PcSensMap.

    doObsOffset = False
    obsOffset = np.array([0])
    if ( len(obsOffset) != len(varPrefixes) ):
        sys.exit("Error: obsOffset must be the same size as the number of variables to tune.")

    # Number of metrics to tune.
    # If there are more metrics than this, then
    #   the metrics in the list beyond this number
    #   will appear in plots but not be counted in the tuning.
    boxSize = 20
    numBoxesInMap = np.rint( (360/boxSize) * (180/boxSize) )
    # numMetricsToTune includes all (e.g., 20x20 regions) and as many
    #   variables as we want to tune, up to all varPrefixes.
    numMetricsToTune = numBoxesInMap * len(varPrefixes)
    #numMetricsToTune = numBoxesInMap * (len(varPrefixes)-1)  # Omit a variable from tuning.
    #numMetricsToTune = numBoxesInMap  # Only tune for first variable in varPrefixes
    numMetricsToTune = numMetricsToTune.astype(int)

    obsOffsetCol = (obsOffset[:, np.newaxis] * np.ones((1, numBoxesInMap.astype(int)))).reshape(-1, 1)


    # Directory where the regional files are stored (plus possibly a filename prefix)
    #folder_name = 'Regional_files/20250725_2yr_20x20_ANN_BCASE/20.0beta06_'
    #folder_name = 'Regional_files/20241022_1yr_20x20regs/30.0sens1022_'
    #folder_name = 'Regional_files/20241022_1yr_sst4k_30x30/30p4k1022_'
    #folder_name = 'Regional_files/20250429_1yr_20x20_ANN_CAM/20.0cam078_'
    folder_name = 'Regional_files/20241022_1yr_20x20regs/20.0sens1022_'
    # folder_name = "data/eam/20.0sens1022_"
    #folder_name = 'Regional_files/20241022_2yr_20x20regs_take3/20.0sens1022_'
    #folder_name = 'Regional_files/20241022_2yr_20x20regs_msq/20.0sens1022_'
    #folder_name = 'Regional_files/20231211_20x20regs/sens0707_'
    ###folder_name = 'Regional_files/20231204_30x30regs/sens0707_'
    #folder_name = 'Regional_files/20degree_CAM_TAUS_202404_DJF/20.0thresp26_'
    #folder_name = 'Regional_files/30degree_CAM_TAUS_202404/30.0thresp26_'
    #folder_name = 'Regional_files/RG_20240402_sens/thresp26_'
    #folder_name = 'Regional_files/20240614_e3sm_20x20regs/thresp26_'
    #folder_name = 'Regional_files/20240409updated/thresp26_'
    #folder_name = 'Regional_files/stephens_20240131/btune_regional_files/btune_'





    # Netcdf file containing metric and parameter values from the default simulation
    #defaultNcFilename = \
    #    folder_name + 'Regional.nc'
    #    'Regional_files/stephens_20240131/btune_regional_files/b1850.076base.n2th1b_Regional.nc'
    #    'Regional_files/20240409updated/thresp26_Regional.nc'
    #    'Regional_files/stephens_20230920/117.f2c.taus_new_base_latest_mods6e_Regional.nc'
    defaultNcFilename = \
        (
            ###folder_name + 'dflt_Regional.nc'
            folder_name + '1_Regional.nc'
        )



    # Metrics from the global simulation that uses the tuner-recommended parameter values
    globTunedNcFilename = \
        (
            #'Regional_files/20250502_1yr_20x20_ANN_CAM/20.0cam078_' + 'qt1_Regional.nc'
            #'Regional_files/20250514_1yr_20x20_ANN_CAM/20.0cam078_' + 'qt2_Regional.nc'
            #'Regional_files/20250530_1yr_20x20_ANN_CAM/20.0cam078_' + 'qt4_Regional.nc'
            #defaultNcFilename
            folder_name + '84_Regional.nc'
            # folder_name + '69_Regional.nc'
    #    'Regional_files/20231211_20x20regs/20sens0707_61_Regional.nc'
    #    'Regional_files/20degree_CAM_TAUS_202404_DJF/20.0Tuner_20240702_20d_DJF_Regional.nc'
    #    'Regional_files/stephens_20240131/btune_regional_files/b1850.076base.n2th1b_Regional.nc'
    #    'Regional_files/20240409updated/thresp26_Regional.nc'
    # 'Regional_files/stephens_20230920/117.f2c.taus_new_base_latest_mods6e_Regional.nc'
    #globTunedNcFilename = \
    #       folder_name + 'sens0707_25_Regional.nc'
           #folder_name + 'sens0707_29_Regional.nc'
           # folder_name + 'chrysalis.bmg20220630.sens1107_30.ne30pg2_r05_oECv3_Regional.nc'
    #        folder_name + 'chrysalis.bmg20220630.sens1107_23.ne30pg2_r05_oECv3_Regional.nc'
        )

    # Parameters are tunable model parameters, e.g. clubb_C8.
    # The float listed below after the parameter name is a factor that is used below for scaling plots.
    #   It is not a weight and doesn't affect optimized values; it just makes the plots more readable.
    # Each parameter is associated with two sensitivity simulations; in one, the parameter is perturbed
    #    up and in the other, it is perturbed down.
    #    The output from each sensitivity simulation is expected to be stored in its own netcdf file.
    #    Each netcdf file contains metric values and parameter values for a single simulation.
    paramsNamesScalesAndSuffixes = \
        [
        ###['clubb_c8', 1.0e-1,
        ### 'clubb_c8m_Regional.nc',
        ### 'clubb_c8p_Regional.nc'],
        ['clubb_c8', 1.0e0,
         '14_Regional.nc',
         '15_Regional.nc'],
        #['clubb_up2_sfc_coef', 1.0,
        # 'clubb_up2_sfc_coefm_Regional.nc',
        # 'clubb_up2_sfc_coefp_Regional.nc'],
        ['clubb_c_invrs_tau_n2', 1.0,
         '10_Regional.nc',
         '11_Regional.nc'],
        #['clubb_altitude_threshold', 0.001, \
        # 'clubb_altitude_thresholdm_Regional.nc',
        # 'clubb_altitude_thresholdp_Regional.nc'], \
        ['clubb_c_invrs_tau_sfc', 1.0,
         '6_Regional.nc',
         '7_Regional.nc'],
        ['clubb_c_invrs_tau_wpxp_n2_thresh', 1.e3,
         '8_Regional.nc',
         '9_Regional.nc'],
        ['clubb_c_invrs_tau_n2_wp2', 1.0,
         '4_Regional.nc',
         '5_Regional.nc'],
        #['clubb_c_invrs_tau_shear', 1.0, \
        # '2_Regional.nc', \
        # '3_Regional.nc'], \
        #['clubb_c_invrs_tau_bkgnd', 1.0, \
        # '16_Regional.nc',
        # '17_Regional.nc'], \
        #['clubb_c11', 1.0, \
        #  'clubb_c11m_Regional.nc',  \
        #  'clubb_c11p_Regional.nc'], \
        ###['clubb_c1', 1.0,
        ###  'clubb_c1m_Regional.nc',
        ###  'clubb_c1p_Regional.nc'],
        ###['clubb_gamma_coef', 1.0,
        ### 'clubb_gamma_coefm_Regional.nc',
        ### 'clubb_gamma_coefp_Regional.nc'],
        #['clubb_c4', 1.0, \
        # 'clubb_c4m_Regional.nc',  \
        # 'clubb_c4p_Regional.nc'], \
        ###['clubb_c14', 1.0, \
        ### 'clubb_c14m_Regional.nc', \
        ### 'clubb_c14p_Regional.nc'], \
        ###['clubb_wpxp_l_thresh', 1.0e-2,
        ### 'clubb_wpxp_l_threshm_Regional.nc',
        ### 'clubb_wpxp_l_threshp_Regional.nc'],
        #['clubb_c6rt_lscale0', 0.1,
        # 'clubb_c6rt_lscale0m_Regional.nc',
        # 'clubb_c6rt_lscale0p_Regional.nc'],
        ###['clubb_c2rt', 0.1,
        ### 'clubb_c2rtm_Regional.nc',
        ### 'clubb_c2rtp_Regional.nc'],
        #['clubb_c6rt', 0.1,
        # 'clubb_c6rtm_Regional.nc',
        # 'clubb_c6rtp_Regional.nc'],
        #['clubb_c6rtb', 0.1,
        # 'clubb_c6rtbm_Regional.nc',
        # 'clubb_c6rtbp_Regional.nc'],
        #['clubb_c_invrs_tau_n2', 1.0, \
        # 'n2p55_Regional.nc', \
        # 'n2p75_Regional.nc'], \
        #['clubb_c_invrs_tau_n2_xp2', 1.0, \
        # 'clubb_c_invrs_tau_n2_xp2m_Regional.nc', \
        # 'clubb_c_invrs_tau_n2_xp2p_Regional.nc'], \
        #['clubb_c_invrs_tau_n2_wp2', 1.0, \
        # 'wp20_Regional.nc', \
        # 'wp24_Regional.nc'], \
        #['clubb_c_invrs_tau_wpxp_ri', 1.0, \
        # 'clubb_c_invrs_tau_wpxp_rim_Regional.nc', \
        # 'clubb_c_invrs_tau_wpxp_rip_Regional.nc'], \
        # ['clubb_wpxp_ri_exp', 1.0, \
        # 'clubb_wpxp_ri_expm_Regional.nc', \
        # 'clubb_wpxp_ri_expp_Regional.nc'], \
        #['clubb_c_invrs_tau_n2_clear_wp3', 1.0, \
        # 'clubb_c_invrs_tau_n2_clear_wp3m_Regional.nc', \
        # 'clubb_c_invrs_tau_n2_clear_wp3p_Regional.nc'], \
        #['clubb_c_k10', 1.0, \
        # '12_Regional.nc', \
        # '13_Regional.nc'], \
        #['clubb_bv_efold', 1.0, \
        # 'clubb_bv_efoldm_Regional.nc', \
        # 'clubb_bv_efoldp_Regional.nc'], \
        #['clubb_c_uu_shr', 1.0,
        # 'clubb_c_uu_shrm_Regional.nc',
        # 'clubb_c_uu_shrp_Regional.nc'],
        #['clubb_c_invrs_tau_bkgnd', 1.0, \
        # 'bkg1_Regional.nc',
        # 'bkg2_Regional.nc'], \
        #['clubb_c_invrs_tau_sfc', 1.0, \
        # 'sfc0_Regional.nc',
        # 'sfcp3_Regional.nc'], \
        #['clubb_c_invrs_tau_shear', 1.0, \
        #  'shr0_Regional.nc', \
        #  'shrp3_Regional.nc'], \
        #['clubb_z_displace', 0.01, \
        #  'zd10_Regional.nc', \
        #  'zd100_Regional.nc'], \
        ###['relvar', 1.0e-2,
        ### 'relvarm_Regional.nc',
        ### 'relvarp_Regional.nc'],
        #['clubb_detliq_rad', 1.0,
        # 'clubb_detliq_radm_Regional.nc',
        # 'clubb_detliq_radp_Regional.nc'],
        ###['clubb_detice_rad', 1.0e4,
        ### 'clubb_detice_radm_Regional.nc',
        ### 'clubb_detice_radp_Regional.nc'],
        #['cldfrc2m_rhmini', 1.0,
        # 'cldfrc2m_rhminim_Regional.nc',
        # 'cldfrc2m_rhminip_Regional.nc'],
        ###['cldfrc_dp1', 1.0e1, \
        ### 'cldfrc_dp1m_Regional.nc', \
        ### 'cldfrc_dp1p_Regional.nc'], \
        #['cldfrc_dp2', 1e-3, \
        # 'cldfrc_dp2m_Regional.nc', \
        # 'cldfrc_dp2p_Regional.nc'], \
        ###['micro_mg_autocon_lwp_exp', 1.,
        ### 'micro_mg_autocon_lwp_expm_Regional.nc',
        ### 'micro_mg_autocon_lwp_expp_Regional.nc'],
        #['micro_mg_accre_enhan_fact', 1.,
        # 'micro_mg_accre_enhan_factm_Regional.nc',
        # 'micro_mg_accre_enhan_factp_Regional.nc'],
        #['micro_mg_dcs', 1000.,
        # 'micro_mg_dcsm_Regional.nc',
        # 'micro_mg_dcsp_Regional.nc'],
        #['micro_mg_berg_eff_factor', 1.,
        # 'micro_mg_berg_eff_factorm_Regional.nc',
        # 'micro_mg_berg_eff_factorp_Regional.nc'],
        #['micro_mg_vtrmi_factor', 1.0,
        #  'micro_mg_vtrmi_factorm_Regional.nc',
        #  'micro_mg_vtrmi_factorp_Regional.nc'],
        #['micro_mg_vtrms_factor', 1.0, \
        #     'micro_mg_vtrms_factorm_Regional.nc',
        #     'micro_mg_vtrms_factorp_Regional.nc'], \
        #['micro_mg_evap_scl_ifs', 1.0, \
        # 'micro_mg_evap_scl_ifsm_Regional.nc',
        # 'micro_mg_evap_scl_ifsp_Regional.nc'],
        #['microp_aero_wsub_scale', 1.0, \
        # 'microp_aero_wsub_scalem_Regional.nc',
        # 'microp_aero_wsub_scalep_Regional.nc'], \
        #['microp_aero_wsubi_scale', 1.0, \
        # 'microp_aero_wsubi_scalem_Regional.nc',
        # 'microp_aero_wsubi_scalep_Regional.nc'], \
        #['zmconv_c0_lnd', 100.0, \
        # 'zmconv_c0_lndm_Regional.nc',
        # 'zmconv_c0_lndp_Regional.nc'], \
        ###['zmconv_dmpdz', 1000.,
        ### 'zmconv_dmpdzm_Regional.nc',
        ### 'zmconv_dmpdzp_Regional.nc'],
        #['zmconv_tau', 1.0e-4,
        # 'zmconv_taum_Regional.nc',
        # 'zmconv_taup_Regional.nc'],
        ###['zmconv_num_cin', 1.0,
        ### 'zmconv_num_cinm_Regional.nc',
        ### 'zmconv_num_cinp_Regional.nc'],
        ###['zmconv_capelmt', 1.0e-2,
        ### 'zmconv_capelmtm_Regional.nc',
        ### 'zmconv_capelmtp_Regional.nc'],
        ###['zmconv_parcel_hscale', 1.0,
        ### 'zmconv_parcel_hscalem_Regional.nc',
        ### 'zmconv_parcel_hscalep_Regional.nc'],
        ###['zmconv_tiedke_add', 1.0,
        ### 'zmconv_tiedke_addm_Regional.nc',
        ### 'zmconv_tiedke_addp_Regional.nc'],
        ###['zmconv_c0_ocn', 1.0e2,
        ### 'zmconv_c0_ocnm_Regional.nc',
        ### 'zmconv_c0_ocnp_Regional.nc'],
        ###['zmconv_c0_lnd', 1.0e2,
        ### 'zmconv_c0_lndm_Regional.nc',
        ### 'zmconv_c0_lndp_Regional.nc'],
        #['zmconv_ke', 1e5,
        # 'zmconv_kem_Regional.nc',
        # 'zmconv_kep_Regional.nc'],
        #['zmconv_ke_lnd', 1e5, \
        # 'zmconv_ke_lndm_Regional.nc',
        # 'zmconv_ke_lndp_Regional.nc'], \
        ###['dust_emis_fact', 1.0,
        ### 'dust_emis_factm_Regional.nc',
        ### 'dust_emis_factp_Regional.nc'],
        #['hetfrz_dust_scalfac', 1.0,
        # 'hetfrz_dust_scalfacm_Regional.nc',
        # 'hetfrz_dust_scalfacp_Regional.nc'],
        ]

    interactParamsNamesAndFilenames = \
    [
    #    ('clubb_c_invrs_tau_wpxp_n2_thresh', 'clubb_c_invrs_tau_n2',
    #     'Regional_files/20241022_2yr_20x20regs/20sens1022_74_Regional.nc'),
        ('clubb_c_invrs_tau_wpxp_n2_thresh', 'clubb_c8',
         'Regional_files/20241022_2yr_20x20regs/20sens1022_75_Regional.nc'),
    #    ('clubb_c8', 'clubb_c_invrs_tau_n2',
    #     'Regional_files/20241022_2yr_20x20regs/20sens1022_76_Regional.nc')
    ]
    interactParamsNamesAndFilenames = []
    interactParamsNamesAndFilenamesType = np.dtype([('jParamName', object),
                                                    ('kParamName', object),
                                                    ('filename',   object)])
    interactParamsNamesAndFilenames = np.array(interactParamsNamesAndFilenames,
                                               dtype=interactParamsNamesAndFilenamesType)

    # SST4K: Output just the filename suffixes here.  Then prepend the normal-SST and SST4K folder names separately.
    #        Create sensNcFilenamesSST4K, etc.

    # Below we designate the subset of paramsNames that vary from [0,1] (e.g., C5)
    #    and hence will be transformed to [0,infinity] in order to make
    #    the relationship between parameters and metrics more linear:
    #transformedParamsNames = np.array(['clubb_c8','clubb_c_invrs_tau_n2', 'clubb_c_invrs_tau_n2_clear_wp3'])
    transformedParamsNames = np.array([''])

    prescribedParamsNamesScalesAndValues = \
        [
            # ['clubb_c11b', 1.0, 0.5,
            #  'clubb_c11bm_Regional.nc',
            #  'clubb_c11bp_Regional.nc'],
                    #['clubb_c8', 1.0, 0.4, \
                    # 'clubb_c8m_Regional.nc',  \
                    # 'clubb_c8p_Regional.nc'], \
#                   ['clubb_wpxp_ri_exp', 1.0, 0.5, \
#                     'clubb_wpxp_ri_expm_Regional.nc', \
#                     'clubb_wpxp_ri_expp_Regional.nc'], \
#                    ['clubb_c8', 1.0, 0.4, \
#                     'clubb_c8m_Regional.nc',  \
#                     'clubb_c8p_Regional.nc'], \
#                    ['clubb_c_invrs_tau_n2_xp2', 1.0, 0.15, \
#                     'clubb_c_invrs_tau_n2_xp2m_Regional.nc', \
#                     'clubb_c_invrs_tau_n2_xp2p_Regional.nc'], \
#                    ['clubb_c8', 1.0, 0.7, \
#                     'sens0707_14_Regional.nc',  \
#                     'sens0707_15_Regional.nc'], \
#                    ['clubb_c_k10', 1.0, 0.3, \
#                     'sens0707_12_Regional.nc', \
#                     'sens0707_13_Regional.nc'], \
#                    ['clubb_c_invrs_tau_n2', 1.0, 0.4, \
#                     'sens0707_10_Regional.nc',
#                     'sens0707_11_Regional.nc'], \
                    #['clubb_c_invrs_tau_sfc', 1.0, 0.05, \
                    # 'sens0707_6_Regional.nc',
                    # 'sens0707_7_Regional.nc'], \
#                    ['clubb_c_invrs_tau_wpxp_n2_thresh', 1.e3, 0.00045, \
#                     'sens0707_8_Regional.nc', \
#                     'sens0707_9_Regional.nc'], \
#                    ['clubb_c_invrs_tau_shear', 1.0, 0.22, \
#                     'sens0707_2_Regional.nc', \
#                     'sens0707_3_Regional.nc'], \
#                    ['clubb_c_invrs_tau_bkgnd', 1.0, 1.1, \
#                     'sens0707_16_Regional.nc',
#                     'sens0707_17_Regional.nc'], \
                    #['clubb_c_invrs_tau_n2_wp2', 1.0, 0.1, \
                    # 'sens0707_4_Regional.nc',
                    # 'sens0707_5_Regional.nc'], \
        ]



    # Read observed values of regional metrics on regular tiled grid into a Python dictionary
    (obsMetricValsDict, obsWeightsDict) = \
        (
            setUp_x_ObsMetricValsDict(varPrefixes, suffix='_[0-9]+_',
                                      obsPathAndFilename='Regional_files/20250711_2yr_20x20_ANN_BCASE/'
                                                         + '20.0_OBS.nc')
            #setUp_x_ObsMetricValsDict(varPrefixes, suffix='_[0-9]+_',
            #                          obsPathAndFilename='Regional_files/20250429_1yr_20x20_ANN_CAM/'
            #                                             + '20.0sens1022_20241011_20.0_OBS.nc')
            #setUp_x_ObsMetricValsDict(varPrefixes, suffix='_[0-9]+_',
            #                          obsPathAndFilename='Regional_files/20231204_30x30regs/'
            #                                             + 'sens0707_20241011_30.0_OBS.nc')
        #setUp_x_ObsMetricValsDict(varPrefixes, folder_name + "20241011_20.0_OBS.nc")
        # setUp_x_ObsMetricValsDict(varPrefixes, suffix='_[0-9]+_',obsPathAndFilename=folder_name + "20.0_OBS.nc")
        #setUp_x_ObsMetricValsDict(varPrefixes, "Regional_files/stephens_20240131/btune_regional_files/b1850.075plus_Regional.nc")
        )

#    # Add on RESTOM separately, since we typically want to prescribe its "observed" value
#    if 'RESTOM' in varPrefixes:

#        obsRESTOMValsDict = {}
#        obsRESTOMWeightsDict = {}

#        # Calculate number of regions in the east-west (X) and north-south (Y) directions
#        numXBoxes = np.rint(360 / boxSize).astype(int)  # 18
#        numYBoxes = np.rint(180 / boxSize).astype(int)  # 9

#        for xBox in range(1, numXBoxes + 1):
#            for yBox in range(1, numYBoxes + 1):

#                varName = f"RESTOM_{yBox}_{xBox}"

#                obsRESTOMValsDict[varName] = 0.0
#                obsRESTOMWeightsDict[varName] = 1.0

        # Append RESTOM values to existing dictionaries
#        obsMetricValsDict.update(obsRESTOMValsDict)
#        obsWeightsDict.update(obsRESTOMWeightsDict)

    # Set metricsNorms to be a global average

    obsGlobalAvgCol, obsGlobalStdCol, obsWeightsCol = \
    calcObsGlobalAvgCol(varPrefixes,
                        obsMetricValsDict, obsWeightsDict)

    # Warning: Using a global average as the constant weight produces little normalized
    #     sensitivity for PSL
    metricsNorms = np.copy(obsGlobalAvgCol)
    # metricsNorms = np.copy(obsGlobalStdCol)

    # obsMetricValsReshaped = obsMetricValsCol.reshape((9,18))
    # biasMat = defaultMetricValsReshaped - obsMetricValsReshaped
    # print("biasMat =")
    # print(np.around(biasMat,2))

    # mse = np.sum(metricsWeights*(defaultMetricValsCol - obsMetricValsCol)**2) \
    #   / np.sum(metricsWeights)
    # rmse = np.sqrt(mse)
    # print("rmse between default and obs =", rmse)



    # These are metrics from customized regions that differ from the standard 20x20 degree tiles.
    # Metrics are observed quantities that we want a tuned simulation to match.
    #    The first column is the metric name.
    #    The order of metricNames determines the order of rows in sensMatrix.
    # The second column is a vector of (positive) weights.  A small value de-emphasizes
    #   the corresponding metric in the fitting process.
    #   Use a large weight for global (GLB) metrics.
    # The third column is a vector of normalization values for metrics.
    #   If a value in the 3rd column is set to -999, then the metric is simply normalized by the observed value.
    #   Otherwise, the value in the 3rd column is itself the normalization value for the metric.
    metricsNamesWeightsAndNormsCustom = \
        [
            # #                        ['TMQ_RMSE', 1.00, 15.], \
            # #                        ['PSL_RMSE', 1.00, 1000.], \
            # #                        ['TS_RMSE', 1.00, 15.], \
            # #                        ['LHFLX_RMSE', 1.00, 15.], \
            # #                        ['SHFLX_RMSE', 1.00, 15.], \
            # #                        ['CLDLOW_RMSE', 1.00, 15.], \
            #                         #['SWCF_RACC', 0.01, 0.2], \
            #                         #['SWCF_RMSEP', 8.01, 15.], \
            #                         #['SWCF_RMSE', 0.01, 15.], \
            #                         ['RESTOM_GLB', 4.0, 10.], \
            #                         #['RESTOM_GLB', 4.0e-3, -999], \
            #                         ['SWCF_GLB', 16.0e-6, -999], \
            #                         ['SWCF_DYCOMS', 4.0e-6, -999], \
            #                         ['SWCF_HAWAII', 4.00e-6, -999], \
            #                         ['SWCF_VOCAL', 4.00e-6, -999], \
            #                         ['SWCF_VOCAL_near', 1.00e-6, -999], \
            #                         ['SWCF_LBA', 1.00e-6, -999], \
            #                         ['SWCF_WP', 1.00e-6, -999], \
            #                         ['SWCF_EP', 1.00e-6, -999], \
            #                         ['SWCF_NP', 1.00e-6, -999], \
            #                         ['SWCF_SP', 1.00e-6, -999],  \
            # ##                        ['SWCF_PA', 1.01, -999], \
            # #                        ['SWCF_CAF', 1.00, -999], \
            #                         ['SWCF_Namibia', 4.00e-6, -999], \
            #                         ['SWCF_Namibia_near', 1.00e-6, -999], \
            #                         #['LWCF_GLB',1.00e-6, -999], \
            # ###                        ['LWCF_DYCOMS', 1.01, -999], \
            # ###                        ['LWCF_HAWAII', 1.01, -999], \
            # ###                        ['LWCF_VOCAL', 1.01, -999], \
            # ##                        ['LWCF_LBA', 1.00, -999], \
            # ##                       ['LWCF_WP', 1.00, -999], \
            # ###                        ['LWCF_EP', 1.01, -999], \
            # ##                        ['LWCF_NP', 1.01, -999], \
            # ##                        ['LWCF_SP', 1.01, -999], \
            # ####                        ['LWCF_PA',  1.01, -999], \
            # ###                        ['LWCF_CAF', 1.01, -999], \
            #                         #['PRECT_GLB', 1.00, -999], \
            #                         #['PRECT_RACC', 0.01, 1.0], \
            #                         #['PRECT_RMSEP', 0.01, 1.0], \
            #                         #['PRECT_RMSE', 0.01, 1.0], \
            # ##                        ['PRECT_LBA', 1.00, -999], \
            # ##                        ['PRECT_WP', 1.00, -999], \
            # ###                        ['PRECT_EP', 1.01, -999], \
            # ###                        ['PRECT_NP', 1.01, -999], \
            # ###                        ['PRECT_SP', 1.01, -999], \
            # ####                        ['PRECT_PA', 1.01, -999], \
            # ##                        ['PRECT_CAF', 1.00, -999], \
            # #                        ['PSL_DYCOMS', 1.e0, 1e3], \
            # #                        ['PSL_HAWAII', 1.e0, 1e3], \
            # #                        ['PSL_VOCAL', 1.e0, 1e3], \
            # #                        ['PSL_VOCAL_near', 1.00, 1e3], \
            # #                        ['PSL_LBA', 1.e0, 1e3], \
            # #                        ['PSL_WP', 1.e0, 1e3], \
            # #                        ['PSL_EP', 1.e0, 1e3], \
            # #                        ['PSL_NP', 1.e0, 1e3], \
            # #                        ['PSL_SP', 1.e0, 1e3],  \
            # #                        ['PSL_PA', 1.00, 1e3], \
            # #                        ['PSL_CAF', 1.e0, 1e3], \
            # ##                        ['PSL_Namibia', 1.00, 1e3], \
            # ##                        ['PSL_Namibia_near', 1.00, 1e3], \
        ]

    #                        ['PRECT_DYCOMS', 0.01, -999], \
    #                        ['PRECT_HAWAII', 0.01, -999], \
    #                        ['PRECT_VOCAL', 0.01, -999], \



    # Observed values of our metrics, from, e.g., CERES-EBAF.
    # These observed metrics will be matched as closely as possible by analyzeSensMatrix.
    # NOTE: PRECT is in the unit of m/s
    #(obsMetricValsDictCustom, obsWeightsDictCustom) = \
    #    (
    #        setUp_x_ObsMetricValsDict(metricsNamesCustom, suffix="", obsPathAndFilename="Regional_files/20250429_1yr_20x20_ANN_CAM/" + "20.0sens1022_20241011_20.0_OBS.nc")
    #    )
    if False:
        obsMetricValsDictCustom = {
        'RESTOM_GLB': 1.5,
        'SWCF_RACC': 0,
        'SWCF_RMSEP': 0,
        'SWCF_RMSE': 0, 'TMQ_RMSE': 0, 'PSL_RMSE': 0, 'TS_RMSE': 0, 'LHFLX_RMSE': 0, 'SHFLX_RMSE': 0, 'CLDLOW_RMSE': 0,
        'LWCF_GLB': 28.008, 'PRECT_GLB': 0.000000031134259, 'SWCF_GLB': -45.81, 'TMQ_GLB': 24.423,
        'LWCF_DYCOMS': 19.36681938, 'PRECT_DYCOMS':0.000000007141516, 'SWCF_DYCOMS': -63.49394226, 'TMQ_DYCOMS':20.33586884,
        'LWCF_LBA': 43.83245087, 'PRECT_LBA':0.000000063727875, 'SWCF_LBA': -55.10041809, 'TMQ_LBA': 44.27890396,
        'LWCF_HAWAII': 23.6855, 'PRECT_HAWAII':0.00000002087774, 'SWCF_HAWAII': -33.1536, 'TMQ_HAWAII': 32.4904,
        'LWCF_WP': 54.5056, 'PRECT_WP':0.000000077433568, 'SWCF_WP': -62.3644, 'TMQ_WP':50.5412,
        'LWCF_EP': 33.42149734, 'PRECT_EP': 0.000000055586694, 'SWCF_EP': -51.79394531, 'TMQ_EP':44.34251404,
        'LWCF_NP': 26.23941231, 'PRECT_NP':0.000000028597503, 'SWCF_NP': -50.92364502, 'TMQ_NP':12.72111988,
        'LWCF_SP': 31.96141052, 'PRECT_SP':0.000000034625369, 'SWCF_SP': -70.26461792, 'TMQ_SP':10.95032024,
        'LWCF_PA': 47.32126999, 'PRECT_PA':0.000000075492694, 'SWCF_PA': -78.27433014, 'TMQ_PA':47.25967789,
        'LWCF_CAF': 43.99757003784179687500, 'PRECT_CAF':0.000000042313699, 'SWCF_CAF': -52.50243378, 'TMQ_CAF':36.79592514,
        'LWCF_VOCAL': 43.99757004, 'PRECT_VOCAL':0.000000001785546, 'SWCF_VOCAL': -77.26232147, 'TMQ_VOCAL':17.59922791,
        'LWCF_VOCAL_near': 15.4783, 'PRECT_VOCAL_near':0.0000000037719, 'SWCF_VOCAL_near': -58.4732, 'TMQ_VOCAL_near': 14.9315,
        'LWCF_Namibia': 12.3294, 'PRECT_Namibia':0.00000000177636 , 'SWCF_Namibia': -66.9495, 'TMQ_Namibia': 24.4823,
        'LWCF_Namibia_near': 10.904, 'PRECT_Namibia_near':0.00000000238369 , 'SWCF_Namibia_near': -36.1216, 'TMQ_Namibia_near': 17.5188,
        'PRECT_RACC': 0,
        'PRECT_RMSEP': 0,
        'PRECT_RMSE': 0,
        'PSL_DYCOMS': 101868.515625,
        'PSL_HAWAII': 101656.578125,
        'PSL_VOCAL': 101668.703125,
        'PSL_VOCAL_near': 101766.8203125,
        'PSL_Namibia_near': 101741.7265625,
        'PSL_Namibia far': 101550.6640625,
        'PSL_LBA': 101052.40625,
        'PSL_WP':  100909.4140625,
        'PSL_EP':  101116.875,
        'PSL_SP':  100021.4921875,
        'PSL_NP':  101314.546875,
        'PSL_PA':  100990.25,
        'PSL_CAF': 100941.7890625
        }


    return (numMetricsToTune,
     varPrefixes, boxSize,
     doCreatePlots, metricsNorms,
     obsMetricValsDict,
     obsOffsetCol, obsGlobalAvgCol, doObsOffset,
     obsWeightsCol,
     transformedParamsNames,
     defaultNcFilename, globTunedNcFilename,
     interactParamsNamesAndFilenames,
     doMaximizeRatio,
     doPiecewise,
     reglrCoef, penaltyCoef, doBootstrapSampling,
     paramsNamesScalesAndSuffixes, folder_name,
     prescribedParamsNamesScalesAndValues,
     metricsNamesWeightsAndNormsCustom,
     debug_level, recovery_test_dparam, sensParamBounds, beVerbose)

    # SST4K: Output defaultNcFilenameSST4K, etc.

def config_plots(beVerbose: bool, varPrefixes:list[str], paramsNames:list[str]) -> tuple[dict[str, bool], np.ndarray, int, Callable]:
    """
    Configure settings for creating plots.
    For example, specify which plots to create.
    
    :param beVerbose: Boolean flag to make output more verbose.
    """

    def abbreviateParamsNames(paramsNames):
        """
        Abbreviate parameter names so that they fit on plots.
        This is handled manually with the lines of code below.
        """

        paramsAbbrv = np.char.replace(paramsNames, 'clubb_', '')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'c_invrs_tau_', '')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'wpxp_n2', 'n2')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'altitude', 'alt')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'threshold', 'thres')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'thresh', 'thres')

        return paramsAbbrv

    # Use these flags to determine whether or not to create specific plots
    # in create_nonbootstrap_figs.py
    createPlotType = {
        'paramsErrorBarsFig': True,               # Parameter values with error bars
        'biasesOrderedArrowFig': False,            # Predicted vs. actual global-model bias removal
        'threeDotFig': True,                       # Quadratic fnc for each metric and parameter
        'metricsBarChart': True,                   # Visualization of tuning matrix eqn
        'paramsIncrsBarChart': True,               # Mean parameter contributions to removal of biases
        'paramsAbsIncrsBarChart': True,            # Squared parameter contributions to bias removal
        'paramsTotContrbBarChart': False,          # Linear + nonlinear contributions to bias removal
        'biasesVsDiagnosticScatterplot': False,    # Scatterplot of biases vs. other fields
        'dpMin2PtFig': False,                      # Min param perturbation needed to simultaneously remove 2 biases
        'dpMinMatrixScatterFig': False,            # Scatterplot of min param perturbation for 2-bias removal
        'projectionMatrixFigs': False,             # Color-coded projection matrix
        'biasesVsSensMagScatterplot': True,        # Biases vs. parameter sensitivities
        'biasesVsSvdScatterplot': False,           # Left SV1*bias vs. left SV2*bias
        'paramsCorrArrayFig': True,                # Color-coded matrix showing correlations among parameters
        'sensMatrixAndBiasVecFig': False,          # Color-coded matrix equation
        'PcaBiplot': False,                        # Principal components biplot
        'PcSensMap': True,                         # Maps showing sensitivities to parameters and left singular vectors
        'vhMatrixFig': True,                       # Color-coded matrix of right singular vectors
        'lossFncVsParamFig': True,                 # 2D loss function plots
        'SST4KPanelGallery': True                  # Maps showing metrics perturbation for parameters from Generalized Eigenvalue problem
    }

    if beVerbose:
        print(f"Creating {sum(createPlotType.values())} types of plots.")



    # mapVarIdx is the field is plotted in the 20x20 maps created by PcSensMap.
    mapVar = 'SWCF'
    mapVarIdx = varPrefixes.index(mapVar)

    # These are a selected subset of the tunable metrics that we want to include
    # in the metrics bar-chart, 3-dot plot, etc.
    # They must be a subset of metricsNames
    highlightedRegionsToPlot = np.array(['1_6', '1_14', '3_6', '3_14',
                                         '6_14', '6_18', '8_13'])
    mapVarPlusUnderscore = mapVar + '_'
    highlightedMetricsToPlot = np.char.add(mapVarPlusUnderscore, highlightedRegionsToPlot)      


    return createPlotType, highlightedMetricsToPlot, mapVarIdx, abbreviateParamsNames


def config_bootstrap(beVerbose: bool) -> int:
    """
    Configure settings for bootstrap sampling.
    For example specify how many bootstrap samples to create.
    
    :param beVerbose: Boolean flag to make output more verbose.
    """
    numBootstrapSamples: int = 100

    
    return numBootstrapSamples

def config_additional(beVerbose:bool) -> tuple[str, str]:
    """
    Configure additional settings.
    For example, specify SST4K filenames.
    """

    # Directory where the SST4K regional files are stored (plus possibly a filename prefix)
    folder_name_SST4K = 'Regional_files/20241022_1yr_sst4k_20x20/20p4k1022_'

    defaultSST4KNcFilename = \
    (
        folder_name_SST4K + '1_Regional.nc'
    )

    return folder_name_SST4K, defaultSST4KNcFilename




