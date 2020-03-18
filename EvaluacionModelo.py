import numpy as np
import pandas as pd
class CalcularRetornos():
    def CalcularGananciasTrain(self,Base,EstrategiaNom):
        Ganancias = []
        Accion = []
        Base['Accion'+EstrategiaNom] = 0
        if Base[EstrategiaNom].iloc[0]==1:
            Precio = Base['Adj Close'].iloc[0]
            Base['Accion'+EstrategiaNom].iloc[0] = 1
            #Accion.append(1)
        else:
            Precio = 0
            Base['Accion'+EstrategiaNom].iloc[0] = -1

        for i in np.arange(1,Base.shape[0]):
            if (Base[EstrategiaNom].iloc[i]==1) & (Precio != 0):
                Precio = Precio
            elif (Base[EstrategiaNom].iloc[i]==0) & (Precio != 0):
                Ganancias.append(Base['Adj Close'].iloc[i]-Precio)
                Precio=0
                Base['Accion'+EstrategiaNom].iloc[i] = -1
            elif (Base[EstrategiaNom].iloc[i]==1) & (Precio == 0):
                Precio = Base['Adj Close'].iloc[i]
                Base['Accion'+EstrategiaNom][i] = 1
            elif (Base[EstrategiaNom].iloc[i]==0) & (Precio == 0):
                Precio=0
        return Ganancias,Base
    
    
    def CalcularRetornos(self,Base,EstrategiaNom):
        Ganancias = []
        if Base[EstrategiaNom].iloc[0]==1:
            Precio = Base['Adj Close'].iloc[0]
        else:
            Precio = 0

        for i in np.arange(1,Base.shape[0]):#(DatosTest.Base.shape[0])):
            if (Base[EstrategiaNom].iloc[i]==1) & (Precio != 0):
                Precio = Precio
            elif (Base[EstrategiaNom].iloc[i]==0) & (Precio != 0):
                Ganancias.append((Base['Adj Close'].iloc[i]-Precio)/Precio)
                Precio=0
            elif (Base[EstrategiaNom].iloc[i]==1) & (Precio == 0):
                Precio = Base['Adj Close'].iloc[i]
            elif (Base[EstrategiaNom].iloc[i]==0) & (Precio == 0):
                Precio=0
        return Ganancias
