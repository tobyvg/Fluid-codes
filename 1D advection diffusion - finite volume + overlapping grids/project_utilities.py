import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import sys
from os import listdir







def init_mpl(global_dpi,labelsize = 15.5,legendsize = 11.40, fontsize = 13,mat_settings = False): # 13
    if mat_settings:
        fontsize = 10
        labelsize = 13
    mpl.rcParams['figure.dpi']= global_dpi
    mpl.rc('axes', labelsize=labelsize)
    font = {'size'   : fontsize}#'family' : 'normal', #'weight' : 'bold'
    mpl.rc('font', **font)
    mpl.rc('legend',fontsize = legendsize)





def visualize_matrix(matrix,color = mpl.cm.nipy_spectral, plot = True, vmin = None, vmax = None, ranges = None, rounding = 3, colorbar = True,fig_ax = None,x_points = 9, y_points =9,x_rotation = -45,labels = True,return_ax = False):
    if fig_ax == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    if ranges == None:
        img = ax.imshow(matrix, cmap = color, extent=[0,matrix.shape[1],matrix.shape[0],0], vmin = vmin, vmax = vmax)
    else:
        x_max = max(np.abs(ranges[0]),np.abs(ranges[1]))
        y_max = max(np.abs(ranges[2]),np.abs(ranges[3]))
        x_round = rounding - int(np.log10(x_max)//1)
        y_round = rounding - int(np.log10(y_max)//1)
        img = ax.imshow(matrix,cmap = color, vmin = vmin, vmax = vmax,extent=[-1,1,-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])])
        if len(matrix[0]) < x_points:
            ax.set_xticks(np.linspace(-1,1,x_points))
        else:
            ax.set_xticks(np.linspace(-1,1-2/len(matrix[0]),x_points))
        if len(matrix) < y_points:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0]),y_points))
        else:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])-
                                      len(matrix)/len(matrix[0])* 2/len(matrix),y_points))
        if x_round > 0 and x_round < 2*rounding:
            x = np.round(np.linspace(ranges[0],ranges[1],x_points),x_round)
        else:
            x = np.linspace(ranges[0],ranges[1],x_points)
            x = [np.format_float_scientific(i,precision = rounding-1) for i in x]
        if y_round > 0 and y_round < 2*rounding:
            y = np.round(np.linspace(ranges[3],ranges[2],y_points),y_round)
        else:
            y = np.linspace(ranges[3],ranges[2],y_points)
            y = [np.format_float_scientific(i,precision = rounding-1) for i in y]
        y = np.flip(y)
        ax.set_xticklabels(x,rotation = x_rotation)
        ax.set_yticklabels(y)
    if colorbar:
        fig.colorbar(img,ax = ax)
    if not labels:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    if plot:
        plt.show()
    if not plot and return_ax:
        return ax
