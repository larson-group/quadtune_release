import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.colors as pc
import dash
from dash import dcc, Dash
from dash import html

import xarray as xr



def createMapPanel(fieldToPlotCol,
                   plotWidth,
                   plotTitle,
                   boxSize,
                   colorScale='RdBu_r',
                   minField=None, maxField=None,
                   panelLabel='', RsqdString=''):
    """Create a single global map that displays a field at the resolution of the tiles."""

    regionalMapPanel = go.Figure(go.Scattergeo())

    # Set boundaries for colored regional boxes
    regionalMapPanel.update_layout(xaxis=dict(range=[0, 360]),
                                   yaxis=dict(range=[-90, 90]))
    regionalMapPanel.update_xaxes(showgrid=False,
                                  tickmode="linear",
                                  dtick=60, tick0=0)
    regionalMapPanel.update_yaxes(showgrid=False,
                                  zeroline=False,
                                  tickmode="linear",
                                  dtick=30, tick0=-90)

    # Calculate number of regions in the east-west (X) and north-south (Y) directions
    numXBoxes = np.rint(360 / boxSize).astype(int)  # 18
    numYBoxes = np.rint(180 / boxSize).astype(int)  # 9

    # Draw longitudinal regional box number along top of plot
    xBoxNums = np.linspace(start=1, stop=numXBoxes, num=numXBoxes).astype(int)
    for xIdx, xBoxNum in np.ndenumerate(xBoxNums):
        xPos = int(0.5 * boxSize + xIdx[0] * boxSize)
        regionalMapPanel.add_annotation(x=xPos, yref="paper", y=1.1,
                                        text=xBoxNum.astype(str), showarrow=False)
    # Draw latitudinal box number along right-hand side of plot
    yBoxNums = np.linspace(start=1, stop=numYBoxes, num=numYBoxes).astype(int)
    for yIdx, yBoxNum in np.ndenumerate(yBoxNums):
        yPos = int(90 - 0.5 * boxSize - yIdx[0] * boxSize)
        regionalMapPanel.add_annotation(xref="paper", x=1.04,
                                        y=yPos,
                                        text=yBoxNum.astype(str), showarrow=False)

    #plotHeight = np.rint(plotWidth * (490 / 700))
    plotHeight = np.rint(plotWidth * (370 / 700))
    regionalMapPanel.update_layout(width=plotWidth, height=plotHeight)
    #regionalMapPanel.update_layout(width=700, height=450)
    #regionalMapPanel.update_layout(title=plotTitle, title_y=0.9, title_x=0.5)
    regionalMapPanel.update_layout(title=plotTitle + RsqdString,
                                   title_y=0.9, title_yanchor='bottom',
                                   title_x=0.45, title_xanchor='center')
    regionalMapPanel.add_annotation(
        xref="paper", yref="paper",
        x=-0.01, y=1.02, xanchor="right", yanchor="bottom",
        text=panelLabel,
        font=dict(size=17),
        showarrow=False
    )

    # Shift coastal boundaries to correct location on map
    regionalMapPanel.update_geos(lataxis_range=[-90, 90],
                                 lonaxis_range=[0, 360])

    fieldToPlotMatrix = fieldToPlotCol.reshape(numYBoxes, numXBoxes)
    # Shift from 0,360 to -180,180 degrees longitude
    #fieldToPlotMatrix = np.roll(fieldToPlotMatrix, -9, axis=1)

    # Set color scaling for colors of regional boxes
    # latRange = range(90, -90, -boxSize)
    latRange = np.arange(90, -90, -boxSize)
    # lonRange = range(0, 360, boxSize)
    lonRange = np.arange(0, 360, boxSize)
    normlzdColorMatrix = np.zeros_like(fieldToPlotMatrix)
    if minField == None or maxField == None:
        minField = np.min(fieldToPlotMatrix)
        maxField = np.max(fieldToPlotMatrix)
    rangeField = np.maximum( np.abs(maxField), np.abs(minField) )

    # If the data include 0, use a diverging colorscale.
    # Remap 0 to 0.5 and normalize the data so that
    #    the data lie within the range [0,1].
    if ( True ):
    #if ( maxField > 0) & (minField < 0):
        #colorScale = 'RdBu_r'
        for latIdx, lat in enumerate(latRange):
            for lonIdx, lon in enumerate(lonRange):
                normlzdColorMatrix[latIdx, lonIdx] = \
                    0.5 * fieldToPlotMatrix[latIdx, lonIdx] / rangeField + 0.5
                #if fieldToPlotMatrix[latIdx][lonIdx] < 0:
                #    normlzdColorMatrix[latIdx, lonIdx] = \
                #        0.5 * (fieldToPlotMatrix[latIdx, lonIdx]-minField)/np.abs(minField)
                #else:
                #    normlzdColorMatrix[latIdx, lonIdx] = \
                #        0.5 * fieldToPlotMatrix[latIdx, lonIdx] / maxField + 0.5
    #else:  # Don't use diverging colorscale
    #    #colorScale = 'Bluered'
    #    colorScale = 'Aggrnyl'
    #    normlzdColorMatrix = (fieldToPlotMatrix - minField) / \
    #                  (maxField - minField)

    # Draw a colored rectangle in each region in layer underneath

    # This new version is faster, since it uses fewer index lookups and generates all shapes first and then updates the layout only once.
    # The old version called update_layout repeatedly within the loop.
    normlzdColorMatrixflat = normlzdColorMatrix.ravel()
    sampled_colors = pc.sample_colorscale(colorscale=colorScale, samplepoints=normlzdColorMatrixflat)

    # Define positions of north,east,south,west edges of each box
    # Each array has numYBoxes*numXBoxes elements
    lonsWestEdges = np.tile(lonRange,numYBoxes)
    lonsEastEdges = lonsWestEdges + boxSize
    latsNorthEdges = np.repeat(latRange, numXBoxes)
    latsSouthEdges = latsNorthEdges - boxSize
    shapes = [dict(
        type="rect",
                xref="x",
                yref="y",
                x0=lonsWestEdge,
                y0=latsNorthEdge,
                x1=lonsEastEdge,
                y1=latsSouthEdge,
                line=dict(color="black", width=1),
                fillcolor=color,
                opacity=1.0,
                layer="below"
    ) for lonsWestEdge, latsNorthEdge, lonsEastEdge, latsSouthEdge, color in zip(lonsWestEdges, latsNorthEdges, lonsEastEdges, latsSouthEdges, sampled_colors)]

    regionalMapPanel.update_layout(shapes=shapes)

    # Draw map of land boundaries in layer on top
    regionalMapPanel.update_geos(showcoastlines=True,
                                 coastlinecolor='black',
                                 coastlinewidth=1,
                                 showlakes=False,
                                 showland=False,
                                 showocean=False,
                                 bgcolor='rgba(0,0,0,0)',
                                 lonaxis_showgrid=False,
                                 lataxis_showgrid=False,
                                 domain_x=[0.0, 1.0]
                                 )
    # Add colorbar by creating an invisible, fake scatterplot
    #if (colorScale == 'RdBu_r'):
    if (True):
        tickVals = [np.min(normlzdColorMatrix),
                    0.5*np.min(normlzdColorMatrix)+0.5*np.max(normlzdColorMatrix),
                    np.max(normlzdColorMatrix)]
        
        if rangeField > 0.009:
            tickText = [f"{-rangeField:.2f}",
                        '0.0',
                        f"{rangeField:.2f}"]
        else:
            tickText = [f"{-rangeField:.2e}",
                        '0.0',
                        f"{rangeField:.2e}"]
            

        #if (maxField > np.abs(minField)):
        #    #tickVals = [0.5 * minField / rangeField + 0.5,
        #    #            0.25 * minField / rangeField + 0.75,
        #    #            1.0]
        #    tickVals = [0.5 * np.min(fieldToPlotMatrix) / rangeField + 0.5,
        #                0.25 * np.min(fieldToPlotMatrix) / rangeField + 0.75,
        #                1.0]
        #    tickText = [f"{-maxField:.2f}",
        #                '0.0',
        #                f"{maxField:.2f}"]
        #else:
        #    #tickVals = [0.0,
        #    #            0.25 * maxField / rangeField + 0.25,
        #    #            0.5 * maxField / rangeField + 0.5]
        #    tickVals = [0.0,
        #                0.25 * np.max(fieldToPlotMatrix) / rangeField + 0.25,
        #                0.5 * np.max(fieldToPlotMatrix) / rangeField + 0.5]
        #    tickText = [f"{minField:.2f}",
        #                '0.0',
        #                f"{-minField:.2f}"]
    else:
        tickVals = [0.0, 1.0]
        tickText = ['0', f"{rangeField:.2f}"]
    colorbar_trace = go.Scatter(x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(
                                    #color=fieldToPlotCol,
                                    color=normlzdColorMatrix.reshape((numYBoxes*numXBoxes,)),
                                    colorscale=colorScale,
                                    #colorscale=colorList,
                                    showscale=True,
                                    colorbar=dict(thickness=15,
                                                  tickvals = tickVals,
                                                  ticktext = tickText
                                                  )
                                                  #outlinewidth=0)
                                )
                                )

    regionalMapPanel.add_trace(colorbar_trace)

    #regionalMapPanel.update_geos(fitbounds="locations")

    #regionalMapPanel.update_layout(margin=dict(l=10, r=10, t=40, b=20))
    regionalMapPanel.update_layout(margin=dict(l=5, r=5, t=50, b=20))

    return regionalMapPanel


from tuning_files.read_PPE_files import get_weights_names





def restrict_by_common_members(PPE_params, PPE_metrics, PPE_params_sst4k):
    """
    Restrict the PPE datasets to the common members between the two datasets,
    ensuring a 1-to-1 alignment.
    """
    default_params = PPE_params["params"].values
    sst4k_params = PPE_params_sst4k["params"].values

    matching_indices_default = []
    matching_indices_sst4k = []


    used_sst4k_indices = set()

    for idx, default_param_set in enumerate(default_params):
        matches = np.where(np.all(np.isclose(sst4k_params, default_param_set, atol=1e-6), axis=1))[0]

        for match_idx in matches:
            if match_idx not in used_sst4k_indices:
                matching_indices_default.append(idx)
                matching_indices_sst4k.append(match_idx)
                used_sst4k_indices.add(match_idx)
                break 

    print(f"Number of common, unique member pairs: {len(matching_indices_default)}")

    PPE_params = PPE_params.isel(ens_idx=matching_indices_default)
    PPE_metrics = PPE_metrics.isel(ens_idx=matching_indices_default)
    


    return PPE_params, PPE_metrics





if __name__ == '__main__':



    params_names =['clubb_c1',
        'clubb_gamma_coef',
        'zmconv_tau',
        'zmconv_dmpdz',
        'zmconv_micro_dcs',
        'nucleate_ice_subgrid',
        'p3_nc_autocon_expon',
        'p3_qc_accret_expon',
        'zmconv_auto_fac',
        'zmconv_accr_fac',
        'zmconv_ke',
        'cldfrc_dp1',
        'p3_embryonic_rain_size',
        'p3_mincdnc'
        ]


    basic_dataset = xr.open_dataset("./tuning_files/PPE_Data/PPE_7_5deg_metrics.nc")
    sst4k_dataset =xr.open_dataset("./tuning_files/PPE_Data/sst4k_PPE_7_5deg_metrics.nc")

    basic_dataset, _ = restrict_by_common_members(basic_dataset, basic_dataset, sst4k_dataset)

    

    boxSize = 7.5

    varPrefixes = ["SWCF"]

    metric_names = []
    for varPrefix in varPrefixes:
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metric_names.append(f"{varPrefix}_{i}_{j}")

    weights_names = get_weights_names(boxSize)


    weights = basic_dataset.isel(time=0,product=0)[weights_names].to_array(dim="metricsNames").to_numpy()
    metricsWeights = np.tile(weights,len(varPrefixes)).reshape(-1,1)



    basic_default = basic_dataset.sel(ens_idx="ctrl").isel(time=0,product=0)[metric_names]
    sst4k_default = sst4k_dataset.sel(ens_idx='ctrl').isel(time=0,product=0)[metric_names]

    vals_basic = basic_dataset.isel(product=0, time=0)[metric_names]
    vals_basic = vals_basic.drop_sel(ens_idx='ctrl')
    vals_sst4k = sst4k_dataset.isel(product=0, time=0)[metric_names]
    vals_sst4k = vals_sst4k.drop_sel(ens_idx='ctrl')
    
    vals_basic = (basic_default-vals_basic).to_array().values
    # vals_basic = vals_basic * metricsWeights
    vals_sst4k = (sst4k_default-vals_sst4k).to_array().values
    # vals_sst4k = vals_sst4k * metricsWeights


    denominator = np.sum(vals_basic**2, axis=0)
    numerator = np.sum(vals_sst4k**2, axis=0)

    ratios_array = numerator / denominator
    ratios = ratios_array.tolist()

    print("I am running")

    plotWidth = 800
    plotHeight = np.rint(plotWidth * (300 / 700))


    minFieldBias = np.minimum.reduce([np.min(vals_basic),np.min(vals_sst4k)])
    maxFieldBias = np.maximum.reduce([np.max(vals_basic),np.max(vals_sst4k)])
    panels=[]

    panelssst4k = []



    for idx, data in enumerate(vals_basic.T):
        panels.append(createMapPanel(data,plotWidth,f"SST Data for PPE Member {idx}, Ratio: {ratios[idx]}",boxSize,colorScale='RDBu_r',minField=minFieldBias,maxField=maxFieldBias))

    print("1/2 done")
    for idx, data in enumerate(vals_sst4k.T):
        panelssst4k.append(createMapPanel(data,plotWidth,f"SST4k Data for PPE Member {idx}",boxSize,colorScale='RDBu_r',minField=minFieldBias,maxField=maxFieldBias))
    print('done, starting server')
    app = Dash(__name__)

    app.layout= html.Div([
        html.H1("PPE SST4k plots"),
        html.Div([
            html.Div(
                style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'marginBottom': '50px'},
            children = [
                dcc.Graph(figure=panel,id=f"graph-{idx}",style={'width': '50%'})
            ]
            
            ) for idx, panel in enumerate(panels)
        ])
            
    ])


    app.layout= html.Div([
        html.H1("PPE Bias plots"),
        html.Div([
            html.Div(
                style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'marginBottom': '50px'},
            children = [
                dcc.Graph(figure=panel,id=f"graph-{idx}",style={'width': '50%'}),
                dcc.Graph(figure=sst4kpanel,id=f"sst4kpanel-{idx}",style={'width': '50%'})
            ]
            
            ) for idx, (panel, sst4kpanel) in enumerate(zip(panels, panelssst4k))
        ])
            
    ])

    
    app.run(debug=True,port=8080, use_reloader=False)