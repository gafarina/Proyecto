import numpy as np
class PreParacionDatosModelo():
    def __init__(self,Base):
        self.Base = Base
        
    def TargetPrice(self,days,Por,Roll,RollMeanOption,*args):
        self.Base = self.Base.dropna()
        if RollMeanOption:
            self.Base.loc[args[0]+'Roll'] = self.Base[args[0]].rolling(Roll).mean()
            Return = (self.Base[args[0]+'Roll'].shift(days)-self.Base[args[0]+'Roll'])/self.Base[args[0]+'Roll']
        else:
            Return = (self.Base[args[0]].shift(days)-self.Base[args[0]])/self.Base[args[0]]
        self.Base['Return'] = Return
        self.Base = self.Base.dropna()
        self.Base['Target'] = 0
        self.Base.loc[self.Base['Return']>=Por[0],'Target'] = 1
        return self.Base
    
    def CompletarEstrategia(self,NombreEstrategia):
        estrategia = [0]
        for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
            if self.Base[NombreEstrategia].iloc[j] == 1:
                estrategia.append(1)
            elif self.Base[NombreEstrategia].iloc[j] == -1:
                estrategia.append(0)
            else:
                estrategia.append(estrategia[j-1])
        self.Base.is_copy = None
        self.Base[NombreEstrategia] = estrategia
        return self.Base
    
    def GenerarestrategiaSMA(self,Shift,NombreEstrategia,Lag):
        self.Base[NombreEstrategia]=0
        AntesCompra = (self.Base['Adj Close'].shift(Shift)<self.Base['SMA'+str(Lag)].shift(Shift))
        DespuesCompra = (self.Base['Adj Close'].shift(1)>self.Base['SMA'+str(Lag)].shift(1))
        
        AntesVenta = (self.Base['Adj Close'].shift(Shift)>self.Base['SMA'+str(Lag)].shift(Shift))
        DespuesVenta = (self.Base['Adj Close'].shift(1)<self.Base['SMA'+str(Lag)].shift(1))
        
        self.Base = self.Base.dropna()
        self.Base.loc[((AntesCompra) & (DespuesCompra)),NombreEstrategia]=1
        self.Base.loc[((AntesVenta) & (DespuesVenta)),NombreEstrategia]=-1
        self.Base = self.CompletarEstrategia(NombreEstrategia)
        return self.Base
        
    
    def GenerarEstrategiaOBV(self,ShiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice):
        self.Base['OBVROllMean'+str(RollMeanOBV)] = self.Base['OBV'].rolling(RollMeanOBV).mean()
        self.Base['Adj Close'+str(RollMeanPrice)] = self.Base['Adj Close'].rolling(RollMeanPrice).mean()
        self.Base[NombreEstrategia] = 0
        
        AntesCompra = (self.Base['Adj Close'].shift(ShiftParaComparar)<self.Base['Adj Close'+str(RollMeanPrice)].shift(ShiftParaComparar)) & (self.Base['OBV'].shift(ShiftParaComparar)<self.Base['OBVROllMean'+str(RollMeanOBV)].shift(ShiftParaComparar))
        
        DespuesCompra = (self.Base['OBV'].shift(1)>self.Base['OBVROllMean'+str(RollMeanOBV)].shift(1)) & (self.Base['Adj Close'].shift(1)>self.Base['Adj Close'+str(RollMeanPrice)].shift(1))
        
        AntesVenta = (self.Base['Adj Close'].shift(ShiftParaComparar)>self.Base['Adj Close'+str(RollMeanPrice)].shift(ShiftParaComparar)) & (self.Base['OBV'].shift(ShiftParaComparar)>self.Base['OBVROllMean'+str(RollMeanOBV)].shift(ShiftParaComparar))
        
        DespuesVenta = (self.Base['OBV'].shift(1)<self.Base['OBVROllMean'+str(RollMeanOBV)].shift(1)) & (self.Base['Adj Close'].shift(1)<self.Base['Adj Close'+str(RollMeanPrice)].shift(1))
        self.Base.loc[(AntesCompra) & (DespuesCompra),NombreEstrategia] = 1
        self.Base.loc[(AntesVenta) & (DespuesVenta),NombreEstrategia] = -1
        self.Base = self.CompletarEstrategia(NombreEstrategia)
        return self.Base
    
    def GenerarEstrategiaWPR(self,VectorLimits,Shift,NombreEstrategia):
        self.Base[NombreEstrategia] = 0
        AntesCompra = (self.Base['WPR'].shift(Shift)<VectorLimits[0][0])
        DespuesCompra = (self.Base['WPR'].shift(1)>VectorLimits[0][1])
        AntesVenta = (self.Base['WPR'].shift(Shift)>VectorLimits[1][0])
        DespuesVenta = (self.Base['WPR'].shift(1)<VectorLimits[1][1])
        self.Base.loc[(AntesCompra) & (DespuesCompra),NombreEstrategia] = 1
        self.Base.loc[(AntesVenta) & (DespuesVenta) ,NombreEstrategia] = -1
        self.Base = self.CompletarEstrategia(NombreEstrategia)
        return self.Base
    
    def GenerarEstrategiaROC(self,ShiftParaComparar,ParametroReturn,NombreEstrategia):
            self.Base[NombreEstrategia] = 0
            AntesCompra = (self.Base['ROC'].shift(ShiftParaComparar)<-ParametroReturn)
            DespuesCompra = (self.Base['ROC'].shift(1)>-ParametroReturn)
            AntesVenta = (self.Base['ROC'].shift(ShiftParaComparar)>ParametroReturn)
            DespuesVenta = (self.Base['ROC'].shift(1)<ParametroReturn)
            
            self.Base.loc[(AntesCompra) & (DespuesCompra),NombreEstrategia] = 1
            self.Base.loc[(AntesVenta) & (DespuesVenta) ,NombreEstrategia] = -1
            self.Base = self.CompletarEstrategia(NombreEstrategia)
            return self.Base
    def GenerarestrategiaEMA(self,Lags,Shift,NombreEstrategia):
        self.Base[NombreEstrategia]=0
        AntesCompra = self.Base['EMA'+str(Lags[0])].shift(Shift)<self.Base['EMA'+str(Lags[1])].shift(Shift)
        DespuesCompra = self.Base['EMA'+str(Lags[0])].shift(1)>self.Base['EMA'+str(Lags[1])].shift(1)
        AntesVenta = self.Base['EMA'+str(Lags[0])].shift(Shift)>self.Base['EMA'+str(Lags[1])].shift(Shift)
        DespuesVenta = self.Base['EMA'+str(Lags[0])].shift(1)<self.Base['EMA'+str(Lags[1])].shift(1)
        self.Base.loc[(AntesCompra) & (DespuesCompra),NombreEstrategia]=1
        self.Base.loc[(AntesVenta) & (DespuesVenta),NombreEstrategia]=-1
        self.Base = self.CompletarEstrategia(NombreEstrategia)
        return self.Base
    
# #     def GenerarestrategiaSMA(self,Shift,NombreEstrategia,Lag):
# #         self.Base[NombreEstrategia]=0
# #         AntesCompra = (self.Base['Adj Close'].shift(Shift)<self.Base['SMA'+str(Lag)].shift(Shift))
# #         DespuesCompra = (self.Base['Adj Close'].shift(1)>self.Base['SMA'+str(Lag)].shift(1))
        
# #         AntesVenta = (self.Base['Adj Close'].shift(Shift)>self.Base['SMA'+str(Lag)].shift(Shift))
# #         DespuesVenta = (self.Base['Adj Close'].shift(1)<self.Base['SMA'+str(Lag)].shift(1))
        
# #         self.Base = self.Base.dropna()
# #         self.Base.loc[((AntesCompra) & (DespuesCompra)),NombreEstrategia]=1
# #         self.Base.loc[((AntesVenta) & (DespuesVenta)),NombreEstrategia]=-1
# #         self.Base = self.CompletarEstrategia(NombreEstrategia)
# #         return self.Base
    
#     def GenerarestrategiaEMA(self,Shift,NombreEstrategia):
#         self.Base[NombreEstrategia]=0
#         self.Base[NombreEstrategia].loc[(self.Base['EMA5'].shift(Shift)<self.Base['EMA21'].shift(Shift)) & (self.Base['EMA5'].shift(1)>self.Base['EMA21'].shift(1))]=1
#         self.Base[NombreEstrategia].loc[(self.Base['EMA5'].shift(Shift)>self.Base['EMA21'].shift(Shift)) & (self.Base['EMA5'].shift(1)<self.Base['EMA21'].shift(1))]=-1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaRSI(self,Shift,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(Shift)<self.Base['SMA5'].shift(Shift)) & (self.Base['Adj Close'].shift(1)>self.Base['SMA5'].shift(1))&(self.Base['RSI'].shift(1)<45)] = 1
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(Shift)>self.Base['SMA5'].shift(Shift)) & (self.Base['Adj Close'].shift(1)<self.Base['SMA5'].shift(1))&(self.Base['RSI'].shift(1)>55)] = -1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaSar(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(ShiftParaComparar)<self.Base['SAR'].shift(ShiftParaComparar)) & (self.Base['Adj Close'].shift(1)>self.Base['SAR'].shift(1))]=1
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(ShiftParaComparar)>self.Base['SAR'].shift(ShiftParaComparar)) & (self.Base['Adj Close'].shift(1)<self.Base['SAR'].shift(1))]=-1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaROC(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['ROC'].shift(ShiftParaComparar)<-4) & (self.Base['ROC'].shift(1)>-4)]=1
#         self.Base[NombreEstrategia].loc[(self.Base['ROC'].shift(ShiftParaComparar)<4) & (self.Base['ROC'].shift(1)>4)]=-1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaBB(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(ShiftParaComparar)<self.Base['Lower'].shift(ShiftParaComparar)) & (self.Base['Adj Close'].shift(1)>self.Base['Lower'].shift(1))]=1
        
#         self.Base[NombreEstrategia].loc[(self.Base['Adj Close'].shift(ShiftParaComparar)<self.Base['Upper'].shift(ShiftParaComparar)) & (self.Base['Adj Close'].shift(1)>self.Base['Upper'].shift(1))]=-1 
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaCCI(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['CCI'].shift(ShiftParaComparar)<-100) & (self.Base['CCI'].shift(1)>-100)]=1
#         self.Base[NombreEstrategia].loc[(self.Base['CCI'].shift(ShiftParaComparar)<100) & (self.Base['CCI'].shift(1)>100)]=-1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaMACD(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['MACD'].shift(ShiftParaComparar) < self.Base['MACDMA'].shift(ShiftParaComparar)) & (self.Base['MACD'].shift(1) > self.Base['MACDMA'].shift(1))]=1
#         self.Base[NombreEstrategia].loc[(self.Base['MACD'].shift(ShiftParaComparar) > self.Base['MACDMA'].shift(ShiftParaComparar)) & (self.Base['MACD'].shift(1) < self.Base['MACDMA'].shift(1))]=-1 
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#     def GenerarEstrategiaSTOCH(self,ShiftParaComparar,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base[NombreEstrategia].loc[(self.Base['SLOWWD'].shift(ShiftParaComparar)<20) & (self.Base['SLOWWD'].shift(1)>20)]=1
#         self.Base[NombreEstrategia].loc[(self.Base['SLOWWD'].shift(ShiftParaComparar)<80) & (self.Base['SLOWWD'].shift(1)>80)]=-1
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
        
#         return self.Base
    
#     def GenerarEstrategiaADX(self,ShiftParaComparar,NombreEstrategia):
#         self.Base['ADX1'] = 0
#         self.Base['ADX2'] = 0
#         self.Base['ADX1'].loc[(self.Base['ADX'].shift(ShiftParaComparar)>25)]=1
#         self.Base['ADX2'].loc[(self.Base['ADX'].shift(ShiftParaComparar)<20)]=-1
#         self.Base[NombreEstrategia] = self.Base['ADX1']+self.Base['ADX2']
#         estrategia = [1]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
#      def GenerarEstrategiaWPR(self,VectorLimits,Shift,NombreEstrategia):
#         self.Base[NombreEstrategia] = 0
#         self.Base.loc[(self.Base['WPR'].shift(Shift)<VectorLimits[0][0]) & (self.Base['WPR'].shift(1)>VectorLimits[0][1]),NombreEstrategia] = 1
#         self.Base.loc[(self.Base['WPR'].shift(Shift)>VectorLimits[1][0]) & (self.Base['WPR'].shift(1)<VectorLimits[1][1]),NombreEstrategia] = -1
#         estrategia = [0]
#         for j in np.arange(1,self.Base[NombreEstrategia].shape[0]):
#             if self.Base[NombreEstrategia].iloc[j] == 1:
#                 estrategia.append(1)
#             elif self.Base[NombreEstrategia].iloc[j] == -1:
#                 estrategia.append(0)
#             else:
#                 estrategia.append(estrategia[j-1])
#         self.Base[NombreEstrategia] = estrategia
#         return self.Base
    
# #     def GenerarEstrategiaOBV(self,ShiftParaComparar,NombreEstrategia,RollMeanOBV,RollMeanPrice):
# #         self.Base['OBVROllMean'+str(RollMeanOBV)] = self.Base['OBV'].rolling(RollMeanOBV).mean()
# #         self.Base['Adj Close'+str(RollMeanPrice)] = self.Base['Adj Close'].rolling(RollMeanPrice).mean()
# #         self.Base[NombreEstrategia] = 0
        
# #         AntesCompra = (self.Base['Adj Close'].shift(ShiftParaComparar)<self.Base['Adj Close'+str(RollMeanPrice)].shift(ShiftParaComparar)) & (self.Base['OBV'].shift(ShiftParaComparar)<self.Base['OBVROllMean'+str(RollMeanOBV)].shift(ShiftParaComparar))
        
# #         DespuesCompra = (self.Base['OBV'].shift(1)>self.Base['OBVROllMean'+str(RollMeanOBV)].shift(1)) & (self.Base['Adj Close'].shift(1)>self.Base['Adj Close'+str(RollMeanPrice)].shift(1))
        
# #         AntesVenta = (self.Base['Adj Close'].shift(ShiftParaComparar)>self.Base['Adj Close'+str(RollMeanPrice)].shift(ShiftParaComparar)) & (self.Base['OBV'].shift(ShiftParaComparar)>self.Base['OBVROllMean'+str(RollMeanOBV)].shift(ShiftParaComparar))
        
# #         DespuesVenta = (self.Base['OBV'].shift(1)<self.Base['OBVROllMean'+str(RollMeanOBV)].shift(1)) & (self.Base['Adj Close'].shift(1)<self.Base['Adj Close'+str(RollMeanPrice)].shift(1))
# #         self.Base.loc[(AntesCompra) & (DespuesCompra),NombreEstrategia] = 1
# #         self.Base.loc[(AntesVenta) & (DespuesVenta),NombreEstrategia] = -1
        
        
# #         self.Base = self.CompletarEstrategia(NombreEstrategia)
# #         return self.Base
        
        
        
# # Base['ADX1'] = 0
# # Base['ADX2'] = 0
# # Base['ADX1'].loc[(Base['ADX'].shift(1)>25)]=1
# # Base['ADX2'].loc[(Base['ADX'].shift(1)<20)]=-1
# # Base['Strategia5'] = Base['ADX1']+Base['ADX2']


# # for i in [5]:
# #     Base['SMA'+str(i)] = ta.SMA(Base['Adj Close'],i)
# # for i in [5,21]:
# #     Base['EMA'+str(i)] = ta.EMA(Base['Adj Close'],i)

# # Base['Upper'],Base['Middle'],Base['Lower'] = ta.BBANDS(np.array(Base['Adj Close']),
# #                                                       timeperiod = 20,
# #                                                       nbdevup=2,
# #                                                       nbdevdn=2,
# #                                                       matype=0)
# # Base['SAR'] = ta.SAR(np.array(Base['High']),np.array(Base['Low']),acceleration=0.02,maximum=0.2)
# # Base['ADX'] = ta.ADX(np.array(Base['High']),np.array(Base['Low']),np.array(Base['Adj Close']),
# #                      timeperiod=14)
# # Base['DIPOS'] = ta.PLUS_DI(np.array(Base['High']),np.array(Base['Low']),np.array(Base['Adj Close']),
# #                      timeperiod=14)
# # Base['DINEG'] = ta.MINUS_DI(np.array(Base['High']),np.array(Base['Low']),np.array(Base['Adj Close']),
# #                      timeperiod=14)
# # Base['CCI'] = ta.CCI(np.array(Base['High']),np.array(Base['Low']),np.array(Base['Adj Close']),
# #                      timeperiod=20)

# # Base['MACD'],Base['MACDMA'],Base['MACDHIST'] = ta.MACD(np.array(Base['Adj Close']),
# #                                                       fastperiod=12,
# #                                                       slowperiod=26,
# #                                                       signalperiod=9)
# # Base['ROC'] = ta.ROC(np.array(Base['Adj Close']),
# #                      timeperiod=21)
# # Base['RSI'] = ta.RSI(np.array(Base['Adj Close']),timeperiod=14)

# # Base['SLOWWK'],Base['SLOWWD'] = ta.STOCH(np.array(Base['High']),np.array(Base['Low']),np.array(Base['Adj Close']),
# #                        fastk_period=14,slowk_period=3,slowk_matype=0,slowd_period=3,
# #                       slowd_matype=0)

# #a = 1
# # for i in ['SMA5']:
# #     Base['Strategia'+str(a)]=0
# #     Base['Strategia'+str(a)].loc[(Base['Adj Close'].shift(2)<Base[i].shift(2)) & (Base['Adj Close'].shift(1)>Base[i].shift(1))]=1
# #     Base['Strategia'+str(a)].loc[(Base['Adj Close'].shift(2)>Base[i].shift(2)) & (Base['Adj Close'].shift(1)<Base[i].shift(1))]=-1
# #     a += 1
# # Base['Strategia'+str(a)]=0
# # Base['Strategia'+str(a)].loc[(Base['EMA5'].shift(2)<Base['EMA21'].shift(2)) & (Base['EMA5'].shift(1)>Base['EMA21'].shift(1))]=1
# # Base['Strategia'+str(a)].loc[(Base['EMA5'].shift(2)>Base['EMA21'].shift(2)) & (Base['EMA5'].shift(1)<Base['EMA21'].shift(1))]=-1

    
# # Base['Strategia3'] = 0
# # Base['Strategia3'].loc[(Base['Adj Close'].shift(2)<Base['Lower'].shift(2)) & (Base['Adj Close'].shift(1)>Base['Lower'].shift(1))]=1
# # Base['Strategia3'].loc[(Base['Adj Close'].shift(2)<Base['Upper'].shift(2)) & (Base['Adj Close'].shift(1)>Base['Upper'].shift(1))]=-1

# # Base['Strategia4'] = 0
# # Base['Strategia4'].loc[(Base['Adj Close'].shift(2)<Base['SAR'].shift(2)) & (Base['Adj Close'].shift(1)>Base['SAR'].shift(1))]=1
# # Base['Strategia4'].loc[(Base['Adj Close'].shift(2)>Base['SAR'].shift(2)) & (Base['Adj Close'].shift(1)<Base['SAR'].shift(1))]=-1

# # Base['ADX1'] = 0
# # Base['ADX2'] = 0
# # Base['ADX1'].loc[(Base['ADX'].shift(1)>25)]=1
# # Base['ADX2'].loc[(Base['ADX'].shift(1)<20)]=-1
# # Base['Strategia5'] = Base['ADX1']+Base['ADX2']

# # Base['Strategia6'] = 0
# # Base['Strategia6'].loc[(Base['MACD'].shift(2) < Base['MACDMA'].shift(2)) & (Base['MACD'].shift(1) > Base['MACDMA'].shift(1))]=1
# # Base['Strategia6'].loc[(Base['MACD'].shift(2) > Base['MACDMA'].shift(2)) & (Base['MACD'].shift(1) < Base['MACDMA'].shift(1))]=-1

# # Base['Strategia7'] = 0
# # Base['Strategia7'].loc[(Base['CCI'].shift(2)<-100) & (Base['CCI'].shift(1)>-100)]=1
# # Base['Strategia7'].loc[(Base['CCI'].shift(2)<100) & (Base['CCI'].shift(1)>100)]=-1

# # Base['Strategia8'] = 0
# # Base['Strategia8'].loc[(Base['ROC'].shift(2)<-5) & (Base['ROC'].shift(1)>-5)]=1
# # Base['Strategia8'].loc[(Base['ROC'].shift(2)<5) & (Base['ROC'].shift(1)>5)]=-1

# # Base['Strategia9'] = 0
# # Base['Strategia9'].loc[(Base['RSI'].shift(2)<30) & (Base['RSI'].shift(1)>30)]=1
# # Base['Strategia9'].loc[(Base['RSI'].shift(2)<70) & (Base['RSI'].shift(1)>70)]=-1

# # Base['Strategia10'] = 0
# # Base['Strategia10'].loc[(Base['SLOWWD'].shift(2)<20) & (Base['SLOWWD'].shift(1)>20)]=1
# # Base['Strategia10'].loc[(Base['SLOWWD'].shift(2)<80) & (Base['SLOWWD'].shift(1)>80)]=-1