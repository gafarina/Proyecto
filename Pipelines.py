import talib as ta
import numpy as np
import PreparacionModelo
from PreparacionModelo import PreParacionDatosModelo

class Pipelines():
    def __init__(self,Base):
        self.Base = Base
        
    def CalcularPrecision(self,NombreEstrategia):
        e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
        n = self.Base.shape[0]
        precision = e/n
        print('La Precision de la estrategia {} con respecto al target es de {}'.format(NombreEstrategia,precision))
        return precision
    
    
    def PipelineEstrategiaSMA(self,Lag,shiftParaComparar,NombreEstrategia):
        Est = PreParacionDatosModelo(self.Base)
        self.Base['SMA'+str(Lag)]  = ta.SMA(self.Base['Adj Close'],Lag)
        self.Base = Est.GenerarestrategiaSMA(shiftParaComparar,'EstrategiaSMA'+str(Lag),Lag)
        precision = self.CalcularPrecision(NombreEstrategia+str(Lag))
        return self.Base
    
    def PipelineEstrategiaOBV(self,shiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice):
        Est = PreParacionDatosModelo(self.Base)
        self.Base['OBV'] = ta.OBV(np.array(self.Base['Adj Close']),
                             np.array(self.Base['Volume']).astype(float))
        self.Base = Est.GenerarEstrategiaOBV(shiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice)
        precision = self.CalcularPrecision(NombreEstrategia)
        return self.Base
    
    def PipelineEstrategiaWPR(self,timeperiodEstrategy,
                              VectorLimits,
                              shiftParaComparar,
                              NombreEstrategia):
        Est = PreParacionDatosModelo(self.Base)
        self.Base['WPR'] = ta.WILLR(np.array(self.Base['High']),
                               np.array(self.Base['Low']),
                               np.array(self.Base['Close']),
                               timeperiod=timeperiodEstrategy)
        self.Base = Est.GenerarEstrategiaWPR(VectorLimits,shiftParaComparar,NombreEstrategia)
        precision = self.CalcularPrecision(NombreEstrategia)
        return self.Base
    
    def PipelineEstrategiaROC(self,timeperiod,shiftParaComparar, ParametroReturn,NombreEstrategia):
        Est = PreParacionDatosModelo(self.Base)
        self.Base['ROC'] = ta.ROC(np.array(self.Base['Adj Close']),timeperiod=timeperiod)
        self.Base = Est.GenerarEstrategiaROC(shiftParaComparar,ParametroReturn,NombreEstrategia)
        precision = self.CalcularPrecision(NombreEstrategia)
        return self.Base
    
    def PipelineEstrategiaEMA(self,Lags,shiftParaComparar,NombreEstrategia):
        Est = PreParacionDatosModelo(self.Base)
        for i in Lags:
            self.Base['EMA'+str(i)] = ta.EMA(self.Base['Adj Close'],i)
        self.Base = Est.GenerarestrategiaEMA(Lags,shiftParaComparar,NombreEstrategia)
        precision = self.CalcularPrecision(NombreEstrategia)
        return self.Base
    
#     def PipelineEstrategiaWPRTest(self,timeperiodEstrategy,
#                               VectorLimits,
#                               shiftParaComparar,
#                               NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['WPR'] = ta.WILLR(np.array(self.Base['High']),
#                                np.array(self.Base['Low']),
#                                np.array(self.Base['Close']),
#                                timeperiod=timeperiodEstrategy)
#         self.Base = Est.GenerarEstrategiaWPR(VectorLimits,shiftParaComparar,'EstrategiaWPR')
#         return self.Base
    
#     def PipelineEstrategiaSMA(self,Lag,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['SMA'+str(Lag)]  = ta.SMA(self.Base['Adj Close'],Lag)
#         self.Base = Est.GenerarestrategiaSMA(shiftParaComparar,'EstrategiaSMA'+str(Lag),Lag)
#         precision = self.CalcularPrecision(NombreEstrategia+str(Lag))
#         return self.Base
    
#     def PipelineEstrategiaSMATest(self,Lag,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['SMA'+str(Lag)]  = ta.SMA(self.Base['Adj Close'],Lag)
#         self.Base = Est.GenerarestrategiaSMA(shiftParaComparar,'EstrategiaSMA'+str(Lag))
#         return self.Base
    
#     def PipelineEstrategiaEMA(self,Lags,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         for i in Lags:
#             self.Base['EMA'+str(i)] = ta.EMA(self.Base['Adj Close'],i)
#         self.Base = Est.GenerarestrategiaEMA(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base['EstrategiaEMA']==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
#         return self.Base
    
#     def PipelineEstrategiaRSI(self,timeperiodEstrategy,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['RSI'] = ta.RSI(np.array(self.Base['Adj Close']),timeperiod=timeperiodEstrategy)
#         self.Base = Est.GenerarEstrategiaRSI(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base['EstrategiaRSI']==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
#         return self.Base
    
#     def PipelineEstrategiaSar(self,acceleration,maximum,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['SAR'] = ta.SAR(np.array(self.Base['High']),
#                                   np.array(self.Base['Low']),
#                                   acceleration=acceleration,
#                                   maximum=maximum)
#         self.Base = Est.GenerarEstrategiaSar(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
#         return self.Base
    
#     def PipelineEstrategiaROC(self,timeperiod,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['ROC'] = ta.ROC(np.array(self.Base['Adj Close']),timeperiod=timeperiod)
#         self.Base = Est.GenerarEstrategiaROC(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
#         return self.Base
    
#     def PipelineEstrategiaBB(self,timeperiod,nbdevup,nbdevdn,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['Upper'],self.Base['Middle'],self.Base['Lower'] = ta.BBANDS(np.array(self.Base['Adj Close']),
#                                                                timeperiod = timeperiod,
#                                                                nbdevup=nbdevup,
#                                                                nbdevdn=nbdevdn,
#                                                                matype=0)
#         self.Base = Est.GenerarEstrategiaBB(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
        
#         return self.Base
    
#     def PipelineEstrategiaCCI(self,timeperiod,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['CCI'] = ta.CCI(np.array(self.Base['High']),np.array(self.Base['Low']),np.array(self.Base['Adj Close']),
#                      timeperiod=timeperiod)
#         self.Base = Est.GenerarEstrategiaCCI(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
#         return self.Base
    
#     def PipelineEstrategiaMACD(self,fastperiod,slowperiod,signalperiod,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['MACD'],self.Base['MACDMA'],self.Base['MACDHIST'] = ta.MACD(np.array(self.Base['Adj Close']),
#                                                                fastperiod=fastperiod,
#                                                                slowperiod=slowperiod,
#                                                                signalperiod=signalperiod)
#         self.Base = Est.GenerarEstrategiaMACD(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
        
#         return self.Base
    
#     def PipelineEstrategiaSTOCH(self,fastk_period,slowk_period,slowd_period,
#                                 shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['SLOWWK'],self.Base['SLOWWD'] = ta.STOCH(np.array(self.Base['High']),
#                                                  np.array(self.Base['Low']),
#                                                  np.array(self.Base['Adj Close']),
#                                                  fastk_period=fastk_period,
#                                                  slowk_period=slowk_period,
#                                                  slowk_matype=0,
#                                                  slowd_period=slowd_period,
#                                                  slowd_matype=0)
#         self.Base = Est.GenerarEstrategiaSTOCH(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
        
#         return self.Base
    
#     def PipelineEstrategiaADX(self,timeperiod,shiftParaComparar,NombreEstrategia):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['ADX'] = ta.ADX(np.array(self.Base['High']),
#                              np.array(self.Base['Low']),
#                              np.array(self.Base['Adj Close']),
#                              timeperiod=timeperiod)
#         self.Base = Est.GenerarEstrategiaADX(shiftParaComparar,NombreEstrategia)
#         e = self.Base[self.Base[NombreEstrategia]==self.Base['Target']].shape[0]
#         n = self.Base.shape[0]
#         precision = e/n
#         print(precision)
        
#         return self.Base
    
#     def PipelineEstrategiaOBV(self,shiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice):
#         Est = PreParacionDatosModelo(self.Base)
#         self.Base['OBV'] = ta.OBV(np.array(self.Base['Adj Close']),
#                              np.array(self.Base['Volume']).astype(float))
#         self.Base = Est.GenerarEstrategiaOBV(shiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice)
#         precision = self.CalcularPrecision(NombreEstrategia)
        
#         return self.Base

        
        