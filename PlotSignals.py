import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Graficos():
    
    def __init__(self,Base):
        self.Base = Base
    
    def PlotTimeSeries(self,colors,*args):
        plt.plot(self.Base[list(args)])
        
    def TwinTimeSeries(self,colors,Marker,Alpha,Numerogrupo1,*args):
        fig,ax = plt.subplots()
        for i in np.arange(Numerogrupo1):
            if Marker[i] == '':
                ax.plot(self.Base[args[i]],color=colors[i],alpha=Alpha[i])
            else:   
                ax.plot(self.Base[args[i]],color=colors[i],marker=Marker[i],alpha=Alpha[i])
        ax2=ax.twinx()
        cont = Numerogrupo1
        for i in args[Numerogrupo1:]:
            ax2.plot(self.Base[i],'.',color=colors[cont],alpha=Alpha[cont])
            cont += 1
    