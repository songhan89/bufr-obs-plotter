import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pytest
from matplotlib import pyplot as plt
from modules.loader import plot_scatter
from pathlib import Path
import pandas as pd
import os


def test_plot_scatter():
    # Define input and output paths
    input_path = Path('./data/soi_test.csv')
    output_path = Path('./output/')

    # Read .csv and plot scatter
    plot_scatter(x='yyyymm', y='soi', color='blue', title='SOI',
                 x_label='Date', y_label='SOI', input_path=input_path,
                 save_path=output_path)

    # Check if the output file exists
    assert os.path.exists(output_path / 'scatter.png')

    # Check if the output file is not empty
    assert os.path.getsize(output_path / 'scatter.png') > 0

    # Check if the output file is the same as the reference file
    assert plt.imread(output_path / 'scatter.png').mean() == \
        plt.imread(Path('./tests/reference/scatter_test.png')).mean()

    os.remove(output_path / 'scatter.png')
