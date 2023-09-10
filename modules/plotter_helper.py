import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from scipy.ndimage import gaussian_filter

def plot_maxmin_points(ax, lon, lat, data, extent, extrema, nsize, symbol, color='k',
                       plotValue=True, fontsize=4, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)

    margin_x = (extent[1] - extent[0]) * 0.05
    margin_y = (extent[3] - extent[2]) * 0.05

    for i in range(len(mxy)):
        if (lon[mxx[i]] > (extent[0] + margin_x) and lon[mxx[i]] < (extent[1] - margin_x) and
            lat[mxy[i]] > (extent[2] + margin_y) and lat[mxy[i]] < (extent[3] - margin_y)):
            ax.text(lon[mxx[i]], lat[mxy[i]], symbol, color=color, size=fontsize,
                    clip_on=True, horizontalalignment='center', verticalalignment='center',
                    transform=transform)
            ax.text(lon[mxx[i]], lat[mxy[i]],
                    '\n' + str(int(data[mxy[i], mxx[i]])),
                    color=color, size=fontsize, clip_on=True, fontweight='bold',
                    horizontalalignment='center', verticalalignment='top', transform=transform)
        
