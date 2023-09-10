"""
Module docstring
"""

import glob
import argparse
import os
import json
import logging
import logging.config
import metpy
import re
import pdbufr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
import numpy as np
from logging.handlers import RotatingFileHandler
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from PIL import Image
from modules.plotter_helper import plot_maxmin_points
from datetime import datetime, timedelta
from pathlib import Path
from scipy.ndimage import gaussian_filter
from metpy.calc import reduce_point_density
from metpy.units import units
from metpy.plots import add_metpy_logo, current_weather, sky_cover, StationPlot

# Keep log file to a max size of 1 MB and a max of 1 backup file
LOG_MAX_BYTES = 1024 * 1024
LOG_MAX_BACKUP = 1
# Import logging configuration file
logging.config.fileConfig(fname=os.path.join(
    'config', 'log.config'), disable_existing_loggers=False)
# Get the logger specified in the file
f_handler = RotatingFileHandler(os.path.join('logs', 'obs_plotter.log'),
                                maxBytes=LOG_MAX_BYTES,
                                backupCount=LOG_MAX_BACKUP)
f_handler.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
f_format = logging.Formatter('%(asctime)s - \
%(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
log.addHandler(f_handler)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--product', default='UPPERAIR_ASEAN_WINDS_00', dest='product')
    parser.add_argument('--input-folder', dest='input_file_path',
                        default="./input/sample2/", help='Specify input file')
    parser.add_argument('--prefix-filter', dest='prefix_filter', type=str,
                        default='IU', help='`IU` for upper air data, `IS` for surface data')
    parser.add_argument('--output-folder', dest='out_file_path',
                        default=Path('./output/'), help='Specify output directory')
    parser.add_argument('--date', dest='date',
                        default=datetime.utcnow().strftime('%Y%m%d %H%M%S'),
                        help='Specify date')

    # Parse the arguments
    args = parser.parse_args()

    # Log the arguments
    log.debug(args)

    # Open the config file
    with Path('./config/config.json').open(mode='r') as read_file:
        # Load the JSON data into a variable
        json_config = json.load(read_file)
        log.info('Loaded config file')

    # Log the JSON data
    log.debug(json_config)

    # Create a date object based on the date argument
    run_time = datetime.strptime(args.date, '%Y%m%d %H%M%S')

    # List of files to load
    input_list = [os.path.join(args.input_file_path, f) for f in os.listdir(args.input_file_path)
                  if re.search(r'^' + 
                               args.prefix_filter + 
                               '.*' + 
                               run_time.strftime('%d%H') + '\d{2}$', f)]
    
    log.debug(f'List of files to load: {input_list}')

    if json_config[args.product]['data_type'] == 'TEMP_BUFR':
        ds = pd.DataFrame(columns=("WMO_station_id",
                                    "typicalDate",
                                    "typicalTime",
                                    "latitude",
                                    "longitude",
                                    "pressure",
                                    "airTemperature",
                                    "dewpointTemperature",
                                    "windDirection",
                                    "windSpeed"))
        ds = ds.astype({"WMO_station_id": np.int64,
                        "typicalDate": str,
                        "typicalTime": str,
                        "latitude": np.float64,
                        "longitude": np.float64,
                        "pressure": str,
                        "airTemperature": np.float64,
                        "dewpointTemperature": np.float64,
                        "windDirection": np.int64,
                        "windSpeed": np.float64})
        for file in input_list:
            try:
                temp_df = pdbufr.read_bufr(file,
                                            columns=("WMO_station_id",
                                                    "typicalDate",
                                                    "typicalTime",
                                                    "latitude",
                                                    "longitude",
                                                    "pressure",
                                                    "airTemperature",
                                                    "dewpointTemperature",
                                                    "windDirection",
                                                    "windSpeed"),
                                            filters={"typicalDate": run_time.strftime('%Y%m%d'),
                                                    "typicalTime": run_time.strftime('%H%M%S')})
                if len(temp_df) > 0:
                    ds = pd.concat([ds, temp_df], ignore_index=True)
                    log.info(f'Loaded {file}')
                else:
                    log.warning(f'No station data in {file}')
            except Exception as e:
                log.warning(f'Unable to decode {file}: {e}')

    # Preprocessing data
    
    if json_config[args.product]['data_type'] == 'SYNOP_SHIP':

        ds = pd.DataFrame()
        
        for file in input_list:
            try:
                temp_df = pdbufr.read_bufr(file, columns='all',
                                           filters={"typicalDate": run_time.strftime('%Y%m%d'),
                                                    "typicalTime": run_time.strftime('%H%M00')},
                                           flat=True)
                if len(temp_df) > 0:
                    ds = pd.concat([ds, temp_df], ignore_index=True)
                    log.info(f'Loaded {file}')
                else:
                    log.warning(f'No station data in {file}')
            except Exception as e:
                log.warning(f'Unable to decode {file}: {e}')

        # Keep only required cols and rename
        variables = ['typicalDate', 'typicalTime', '#1#shipOrMobileLandStationIdentifier', 
                     '#1#stationNumber', '#1#stationType',
                    '#1#stationOrSiteName', '#1#characteristicOfPressureTendency',
                    '#1#latitude', '#1#longitude', '#1#pressureReducedToMeanSeaLevel', 
                    '#1#airTemperature', '#1#dewpointTemperature', '#1#cloudCoverTotal', 
                    '#1#presentWeather', '#1#windSpeed', '#1#windDirection']
        ds = ds[variables]
        ds.columns = [c.split("#")[-1] for c in ds.columns]
        ds = ds.fillna(value=np.nan)

        # Convert Pa to hPA
        ds['pressureReducedToMeanSeaLevel'] = (ds['pressureReducedToMeanSeaLevel'] / 100).round(0)
        # Convert T to degC
        ds['airTemperature'] = (ds['airTemperature'] - 273.15).round(1)
        ds['dewpointTemperature'] = (ds['dewpointTemperature'] - 273.15).round(1)
        # Convert sky cover to oktas
        ds['cloudCoverTotal'] = ds['cloudCoverTotal'] * 8//100
        # Set missing values to 10
        ds['cloudCoverTotal'].replace(np.nan, 10, inplace=True)
        ds['cloudCoverTotal'] = ds['cloudCoverTotal'].astype(int)
        # Set missing present weather to code 0 - nil symbol
        ds['presentWeather'].replace(np.nan, 0, inplace=True)
        # Remove present weather with code >= 150 
        ds = ds[ds['presentWeather'] < 150]
        ds['presentWeather'] = ds['presentWeather'].astype(int)

    ds['u'] = -ds['windSpeed'] * np.sin(np.radians(ds['windDirection']))
    ds['v'] = -ds['windSpeed'] * np.cos(np.radians(ds['windDirection']))

    # Convert wind speed m/s to knots
    ds['var_x'] = ds['u'] * json_config[args.product]['scale_factor']
    ds['var_y'] = ds['v'] * json_config[args.product]['scale_factor']

    # Plotting parameter
    width = json_config[args.product]['width_px'] / \
        json_config[args.product]['dpi']
    height = json_config[args.product]['height_px'] / \
        json_config[args.product]['dpi']

    # Output params
    img_format = json_config[args.product]['img_format']
    output_fname = run_time.strftime(json_config[args.product]['output_fname'])
    output_fpath = f"{args.out_file_path}/{output_fname}"

    fig = plt.figure(figsize=(width, height),
                     dpi=json_config[args.product]['dpi'])

    # Create a figure with a grid of subplots
    log.info(
        f'Creating figure with {json_config[args.product]["layout"][0]}x{json_config[args.product]["layout"][1]} subplots')
    for n_row in range(json_config[args.product]['layout'][0]):
        for n_col in range(json_config[args.product]['layout'][1]):
            product_name = json_config[args.product]['product_name']
            proj = getattr(
                ccrs, json_config[args.product]['projection'][n_row][n_col])()
            plot_style = json_config[args.product]['plot_style'][n_row][n_col]
            mpl_style = json_config[args.product]['mpl_style']
            coastline_style = json_config[args.product]['coastline_style']
            contour_style = json_config[args.product]['contour_style']
            border_style = json_config[args.product]['border_style']

            variable = json_config[args.product]['variable'][n_row][n_col]

            if variable == 'winds':
                # Select data based on level and time
                data = ds[(ds['typicalTime'] == json_config[args.product]['typicalTime'][n_row][n_col]) &
                          (ds['pressure'] == json_config[args.product]['lvl_val'][n_row][n_col])].copy()
            elif variable == 'synop':
                data = ds[(ds['typicalTime'] == json_config[args.product]['typicalTime'][n_row][n_col])]
                point_locs = proj.transform_points(proj, data['longitude'].values,
                                   data['latitude'].values)
                data = data[reduce_point_density(point_locs, 2.5)]
                print (data.shape)

            extent = json_config[args.product]['extent'][n_row][n_col]

            ax = plt.subplot(json_config[args.product]['layout'][0],
                             json_config[args.product]['layout'][1],
                             n_row *
                             json_config[args.product]['layout'][1] +
                             n_col + 1,
                             projection=proj)

            log.debug(f'Plotting coastline for subplot {n_row}x{n_col}')
            ax.coastlines(**coastline_style)
            ax.add_feature(cf.BORDERS, **border_style)

            if json_config[args.product]['gaussian_smooth']:
                data = gaussian_filter(
                    data, sigma=json_config[args.product]['gaussian_sigma'])
                log.debug(
                    f'Gaussian smoothing data for subplot {n_row}x{n_col}')

            if plot_style == 'contour':
                log.debug(f'Plotting contour line for subplot {n_row}x{n_col}')
                cs2 = ax.contour(lon, lat, data, transform=proj, **mpl_style)
                plt.clabel(cs2, **contour_style)
                high_text_style = json_config[args.product]['high_text_style']
                low_text_style = json_config[args.product]['low_text_style']
                log.debug(
                    f'Plotting max/min points for subplot {n_row}x{n_col}')
                plot_maxmin_points(ax, lon, lat, data, extent,
                                   **high_text_style, transform=proj)
                plot_maxmin_points(ax, lon, lat, data, extent,
                                   **low_text_style, transform=proj)
            elif plot_style == 'contourf':
                log.debug(
                    f'Plotting filled contour for subplot {n_row}x{n_col}')
                cc = ax.contourf(lon, lat, data, transform=proj, **mpl_style)
                cbar = plt.colorbar(
                    cc, **json_config[args.product]['colorbar']['cbar_style'])
                cbar.ax.tick_params(
                    **json_config[args.product]['colorbar']['tick_style'])
                cbar.set_label(
                    **json_config[args.product]['colorbar']['label_style'])
            elif plot_style == 'barb':
                log.debug(
                    f'Plotting filled contour for subplot {n_row}x{n_col}')
                ax.barbs(data['longitude'], data['latitude'], data['var_x'], data['var_y'],
                         transform=proj, **json_config[args.product]['barb_style'])
            elif plot_style == 'station':
                log.debug(
                    f'Plotting synop symbols for subplot {n_row}x{n_col}')
                stationplot = StationPlot(ax, data['longitude'].values, data['latitude'].values,
                          clip_on=True, transform=proj, fontsize=6)

                # Plot the temperature and dew point to the upper and lower left, respectively, of
                # the center point. Each one uses a different color.
                stationplot.plot_parameter('NW', data['airTemperature'].values, color='red')
                stationplot.plot_parameter('SW', data['dewpointTemperature'].values,
                                        color='darkgreen')
                stationplot.plot_parameter('NE', data['pressureReducedToMeanSeaLevel'].values,
                                        formatter=lambda v: format(v, '.0f')[1:])
                stationplot.plot_symbol('C', data['cloudCoverTotal'], sky_cover)
                stationplot.plot_symbol('W', ds['presentWeather'].values, current_weather)
                stationplot.plot_barb(data['var_x'].values, data['var_y'].values)

            log.debug(f'Adding gridlines for subplot {n_row}x{n_col}')
            ax.set_title(f"{json_config[args.product]['label_text'][n_row][n_col]}",
                         **json_config[args.product]['label_style'])
            log.debug(f'Setting extent for subplot {n_row}x{n_col}: {extent}')
            ax.set_extent(extent, crs=proj)

            if json_config[args.product]['plot_ticks']:
                # Customize the appearance of the tick marks and labels for
                # each subplot
                x_ticks = np.arange(
                    extent[0], extent[1] + 1, json_config[args.product]['lon_spacing'])
                y_ticks = np.arange(
                    extent[2], extent[3] + 1, json_config[args.product]['lat_spacing'])
                ax.set_xticks(x_ticks, crs=proj)
                ax.set_yticks(y_ticks, crs=proj)
                ax.tick_params(
                    axis='x', **json_config[args.product]['ticks_style'])
                ax.tick_params(
                    axis='y', **json_config[args.product]['ticks_style'])
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)
                ax.yaxis.set_major_formatter(lat_formatter)

    plt.annotate(**json_config[args.product]['annotation'])
    plt.suptitle(f"{product_name} \nValid {run_time.strftime('%d %b %y %H:%M')} UTC",
                 **json_config[args.product]['title_style'])
    fig.subplots_adjust(**json_config[args.product]['subplots_adjust'])

    if img_format == 'gif':
        png_fpath = f"{output_fpath}.png"
        gif_fpath = f"{output_fpath}.gif"
        try:
            log.debug(f'Saving figure to {png_fpath}')
            fig.savefig(png_fpath, bbox_inches='tight')
            log.debug(f'Converting {png_fpath} to {gif_fpath}')
            img = Image.open(png_fpath)
            log.info(f'Saving figure to {gif_fpath}')
            img.save(gif_fpath, save_all=True,
                     append_images=[img], duration=0, loop=0)
            log.debug(f'Removing {png_fpath}')
            os.remove(png_fpath)
        except Exception as e:
            log.error(f'Error saving figure: {e}')
    else:
        try:
            log.info(
                f'Saving figure to {output_fpath}.{img_format}')
            fig.savefig(f'{output_fpath}.{img_format}', bbox_inches='tight')
        except Exception as e:
            log.error(f'Error saving figure: {e}')


if __name__ == "__main__":
    main()
