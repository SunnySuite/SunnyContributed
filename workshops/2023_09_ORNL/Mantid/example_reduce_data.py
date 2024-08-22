# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np


file = 'LaSrCrO4_60meV_5K.nxs' # Change this to refer to your data file
LoadMD(Filename=file,LoadHistory=False,OutputWorkspace=file)

MDNorm(InputWorkspace=file,
                   SolidAngleWorkspace=None,    # important! for normalization    
                   QDimension0='1,0,0', # Set up histogram axes
                   QDimension1='0,1,0', #
                   QDimension2='0,0,1', #
                   Dimension0Name='QDimension0', # Set up binning along each axis
                   Dimension0Binning='0,0.05,2', #
                   Dimension1Name='QDimension1', #
                   Dimension1Binning='0,0.05,2', #
                   Dimension2Name='QDimension2', #
                   Dimension2Binning='-5,5',     #
                   Dimension3Name='DeltaE',      #
                   Dimension3Binning='-2,2,80',  #    
                   SymmetryOperations='4/mmm', # Set symmetry to propagate data
                   OutputWorkspace='normData_'+file,
                   OutputDataWorkspace='dataMD_'+file,
                   OutputNormalizationWorkspace='normMD_'+file)
            
SaveMD(InputWorkspace='normData_'+file,Filename='experiment_data_normalized.nxs',SaveLogs=False,SaveInstrument=False,SaveSample=True)
SaveMD(InputWorkspace='normMD_'+file,Filename='experiment_normalization.nxs',SaveLogs=False,SaveInstrument=False,SaveSample=True)