'''
Clases
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm._tqdm_notebook import tqdm_notebook
import pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import os
import time
import itertools as it
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
import datetime

class LecturaYOperacionesDeExcel():
    
    def __init__(self,Ruta,NombreOutput):
        self.Ruta = Ruta
        self.NombreOutput = NombreOutput
   
    def GenerarDatosFormateados(self,**kwargs):
        dfNuevo = pd.DataFrame()
        for key,value in kwargs.items():
            df = pd.read_excel(self.Ruta+key+'.xlsx')
            df.columns = value
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index(['Timestamp'])
            dfNuevo = pd.concat([dfNuevo,df],axis=1)
        pkl.dump(dfNuevo,open(self.Ruta+self.NombreOutput,'wb'))
        return dfNuevo
    
    
    def ConcatenarDf(self,TipoConcatenacion,NombreOutput,*args):
        for i in range((len(args)-1)):
            Base = pd.concat([args[i],args[i+1]],axis = TipoConcatenacion)
        pkl.dump(Base,open(self.Ruta+NombreOutput,'wb'))
        return Base
    
    def LeerDatos(self,Ruta):
        Base = pkl.load(open(self.Ruta + self.NombreOutput,'rb'))
        return Base
    
class ModificacionYLimpiezaDeDataFrames():
    
    def __init__(self,Base):
        self.Base = Base
        
    def NombrarColumnas(self,colnames):
        assert self.Base.shape[1] == len(colnames),'El Largo no es el mismo'
        try:
            self.Base.columns = colnames
        except:
            pass 
        else:
            return self.Base
        
    def ClipValoresExtremos(self,**kwargs):
        for key,value in kwargs.items():
            assert key in list(self.Base.columns),'Puede que uno de los inputs no esta en las columnas'
            try:
                self.Base[key].loc[(self.Base[key]<=value[0])] = value[0]
                self.Base[key].loc[(self.Base[key]>=value[1])] = value[1]
            except:
                print('Existe un Error al hacer clip en los datos')
            
        return self.Base
            
    def CortarValoresExtremos(self,**kwargs):
        for key,value in kwargs.items():
            assert key in list(self.Base.columns),'Puede que uno de los inputs no esta en las columnas'
            try:
                self.Base = self.Base[(self.Base[key]>=value[0]) & (self.Base[key]<=value[1])]
            except:
                print('Existe un Error al cortar los datos')
            else:
                return self.Base
            
    
class Graficos(ModificacionYLimpiezaDeDataFrames):
    
    def __init__(self,Base):
        super().__init__(Base)
    
    def TimeSeriesGraphAndBoxPlot(self,*args):
        cont = 0
        for i in args:
            try:
                fig = plt.figure(figsize=(20,5))
                grid = plt.GridSpec(1,3,hspace=0.2,wspace=0.2)
                main_ax = fig.add_subplot(grid[:,:-1])
                y_hist = fig.add_subplot(grid[:,2], xticklabels=[])
                main_ax.plot(self.Base[i],'.',alpha=1.0)
                main_ax.set_title(i,fontsize=15)
                main_ax.set_xlabel('Fecha',fontsize = 15)
                main_ax.set_ylabel(list(self.Base.columns)[cont],fontsize = 15)
                sns.boxplot(x=self.Base[i],ax=y_hist,orient='v')
                cont += 1
            except:
                print('La columna no esta definida para {}'.format(i))
            else:
                plt.figure()            
    
    def GraficosDePoblacion(self,*args):
        try:
            x = np.array(self.Base[args[0]])
            y = np.array(self.Base[args[1]])
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            fig,ax = plt.subplots()
            f = ax.scatter(x,y,c=z, s=100, edgecolor='',cmap='RdYlGn')
            plt.ylabel(args[1],fontsize=20)
            plt.xlabel(args[0],fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(args[0]+ ' Vs ' + args[1],size=18)
            CB = plt.colorbar(f)
            CB.ax.tick_params(labelsize=16)
        except:
            print('La columna {} o {} no esta definida'.format(args[0],args[1]))
    
    def GraficoTimeSeriesEnLaMismaEscala(self,Parameters,**kwargs):
        for key,value in kwargs.items():
            try:
                plt.plot(self.Base[key],color=value[0],label = value[1],alpha=value[2],linewidth=value[3],linestyle=value[4])
            except:
                print('Error en graficar')
        try:        
            plt.title(Parameters['Titulo'][0],fontsize=Parameters['Titulo'][1])
            plt.xlabel(Parameters['xlabel'][0],fontsize=Parameters['xlabel'][1])
            plt.ylabel(Parameters['ylabel'][0],fontsize=Parameters['ylabel'][1])
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.legend()
        except:
            print('Existió un error en los parametros')
            
    def GraficoTimeSeriesEnDosEscalas(self,Parameters,**kwargs):
        '''
        Parametros
        Parameters : dict con key Titulo:[nombre del titulo,fontsize]
                                  xtime: [xlabelname,fontsize,color]
                                  xlabel: [y1labelname,fontsize,color]
                                  ylabel: [y2labelname,fontsize,color]
        Dict : Name of the column = [color,'labelname',alpha,linewidth,state:1 o 0,legend_name]
        '''
        fig,ax1 = plt.subplots()
        try:
            ax1.set_xlabel(Parameters['xtime'][0],color = Parameters['xtime'][2])
            ax1.set_ylabel(Parameters['xlabel'][0],color=Parameters['xlabel'][2])
            cont = 0
            for key,value in kwargs.items():
                if value[5] == 0:
                   ax1.plot(self.Base[key],'.',color=value[0],alpha=value[2],linewidth=value[3],label=value[6])
                   ax1.tick_params(labelcolor=Parameters['xlabel'][2])
            ax2 = ax1.twinx()
            ax2.set_ylabel(Parameters['ylabel'][0], color=Parameters['ylabel'][2])
            for key,value in kwargs.items():
                if value[5] == 1:
                    ax2.plot(self.Base[key],color=value[0],alpha=value[2],linewidth=value[3],linestyle=value[4],label=value[6])
                    ax2.tick_params(axis=value[1],labelcolor=value[0])
            fig.tight_layout()
            fig.legend(bbox_to_anchor=(0.9,0.89), loc="lower right",  bbox_transform=fig.transFigure)
            plt.title(Parameters['Titulo'][0],fontsize = Parameters['Titulo'][1])
        except:
            print('Existe un error en los parametros')
    
    def Distribution(self,*args,**kwargs):
        try:
            for i in args:
                fig,ax = plt.subplots()
                sns.distplot(self.Base[i], kde=True,kde_kws={"color":kwargs['color'],"alpha":kwargs['alpha'],"linewidth":kwargs['linewidth'],"shade":kwargs['shade']},label = i)
                plt.title(i,fontsize = 24)
                plt.legend()
                ax.tick_params(labelsize=15)
                plt.xlabel(i,fontsize=18)
                plt.figure()
        except:
            print('Revisar las columnas o los parametros')
        finally:
            pass
    
class LSTMClass(ModificacionYLimpiezaDeDataFrames):
    def __init__(self,Base,batch_size,epochs,lr,time_steps,MinMaxScaler):
        super().__init__(Base)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.time_steps = time_steps
        self.min_max_scaler = MinMaxScaler
        
    def Train_Test(self,Traincols,PropTrain_set,MinMax=True):
        train_cols = Traincols
        df_train,df_test = train_test_split(self.Base, train_size=PropTrain_set, test_size=1-PropTrain_set, shuffle=False)
        x = df_train.loc[:,train_cols].values
        if MinMax:
            x_train = self.min_max_scaler.fit_transform(x)
            x_test = self.min_max_scaler.transform(df_test.loc[:,train_cols])
        return x_train,x_test
    
    def Get_all_combinations(self,params):
        all_names = params.keys()
        combinations = it.product(*(params[name] for name in all_names))
        return list(combinations)
    
    def BuildTimeSeries(self,Mat,Y_index):
        dim_0 = Mat.shape[0] - self.time_steps
        dim_1 = Mat.shape[1]
        x = np.zeros((dim_0, self.time_steps, dim_1))
        y = np.zeros((dim_0,))
        for i in np.arange(dim_0):
            x[i] = Mat[i:self.time_steps+i,]
            y[i] = Mat[self.time_steps+i,Y_index]
        #print("length of time-series i/o",x.shape,y.shape)
        return x,y
    
    def TrimData(self,Mat):
        no_of_rows_drop = Mat.shape[0]%self.batch_size
        if(no_of_rows_drop > 0):
            return Mat[:-no_of_rows_drop]
        else:
            return Mat
    
    def Model(self,x_train):
        lstm_model = Sequential()
        lstm_model.add(LSTM(200, batch_input_shape=(self.batch_size, self.time_steps, x_train.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(Dense(1,activation='relu'))
        optimizer = optimizers.RMSprop(lr=self.lr)
        # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return lstm_model
    
    def TrainModel(self,x_train,x_test,TrainModel=True,file=1):
        model = None
        try:
            model = pkl.load(open("lstm_model", 'rb'))
            #print("Loaded saved model...")
        except FileNotFoundError:
            print("Model not found")
        x_t,y_t = self.BuildTimeSeries(x_train,0)
        x_t = self.TrimData(x_t)
        y_t = self.TrimData(y_t)
       # print("Batch trimmed size",x_t.shape, y_t.shape)
        x_temp, y_temp = self.BuildTimeSeries(x_test,0)
        x_val, x_test_t = np.split(self.TrimData(x_temp),2)
        y_val, y_test_t = np.split(self.TrimData(y_temp),2)
        #print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)
        is_update_model = TrainModel
        if model is None or is_update_model:
            from keras import backend as K
            #print("Building model...")
            #print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
            model = self.Model(x_t)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=40, min_delta=0.0001)
            mcp = ModelCheckpoint(os.path.join('C:/Users/farig/OneDrive - BHP/Projects/Datoscyc/',"best_model_"+str(file)+".h5"), monitor='val_loss', verbose=1,
                                               save_best_only=True, save_weights_only=False, mode='min', period=1)
            r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
#             csv_logger = CSVLogger(os.path.join("C://Users/farig/OneDrive - BHP/Projects/Datoscyc/", 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)
            history = model.fit(x_t, y_t, epochs=self.epochs, verbose=2, batch_size=self.batch_size,
                                          shuffle=False, validation_data=(self.TrimData(x_val),
                                          self.TrimData(y_val)), callbacks=[es,mcp])
            #print("saving model...")
            plt.figure()
            Plot = Graficos(history.history)
            Plot.GraficoTimeSeriesEnLaMismaEscala({'Titulo':['Loss vs Val Loss',25],'xlabel':['Epocas',20],'ylabel':['Error',20]},loss=['red','Train_Loss',1.0,2,'-'],val_loss=['blue','Validation_Loss',0.5,2,'-'])
            pkl.dump(model, open('C:/Users/farig/OneDrive - BHP/Projects/Datoscyc/lstm_model.h5', "wb"))
        plt.figure()
        y_pred = model.predict(self.TrimData(x_test_t), batch_size=self.batch_size)
        y_pred = y_pred.flatten()
        y_test_t = self.TrimData(y_test_t)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[0]) + self.min_max_scaler.data_min_[0]
        y_test_t_org = (y_test_t * self.min_max_scaler.data_range_[0]) + self.min_max_scaler.data_min_[0]
        BasePred = pd.DataFrame({'Real':y_test_t_org,'Prediccion':y_pred_org})
        
        Plot = Graficos(BasePred)
        Plot.GraficoTimeSeriesEnLaMismaEscala({'Titulo':['Prediccion vs TPH Real ',25],'xlabel':['Tiempo',20],'ylabel':['TPH',20]},Real=['red','Real',1.0,2,'-'],Prediccion=['blue','Prediccion',0.5,2,'-'])
        
        
        
        return model,BasePred
    
class RandomForestModel(ModificacionYLimpiezaDeDataFrames):
    def __init__(self,Base,n_estimator,max_depth,random_state):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=self.n_estimator,max_depth=self.max_depth,random_state=self.random_state)
        super().__init__(Base)
    
    def TrainTest(self,test_size,ListX,ListY):
        X = np.array(self.Base[ListX])
        Y = np.array(self.Base[ListY])
        train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=test_size,random_state=self.random_state)
        return train_X,test_X,train_Y,test_Y
    
    def Fit(self,XTrain,YTrain):
        fitted_model = self.model.fit(XTrain,YTrain)
        return fitted_model
    
    def GridSearch(self,X,Y,param_grid):
        gsc = GridSearchCV(
           estimator = self.model,
           param_grid = param_grid)

        grid_result = gsc.fit(X,Y)
        best_params = grid_result.best_params_
        return best_params
    
    def CV(self,scoring,X,Y,cv,return_train_score):
        scores = cross_validate(self.model,X,Y,cv=cv,scoring=scoring,return_train_score=return_train_score)
        return scores
    
class CalculoDeImpactoDeChancadores():
    def __init__(self,Base):
        self.Base = Base
    
    def CalculoDeTiempoDetenido(self,Chanc,Manual,FechasDetenciones,*args):
        ### Input : Nombre del chancador como Ch5, args = nombre de la columna de el funcionando del chancador
        ### Cuando el equipo esta funcionando y el minuto anterior estaba funcionando es cero,sino el numero de minutos que estuvo detenido
        DictDetenciones = dict()
        self.Base['BaseTiempoDetenido'+Chanc] = 0
        self.Base['BaseTiempoDetenido'+Chanc].loc[self.Base[args[0]] > 0.5]=list(pd.Series(self.Base[self.Base[args[0]]>0.5].index).diff().dt.total_seconds()/60)
        self.Base = self.Base.dropna()
        self.Base['BaseTiempoDetenido'+Chanc].loc[self.Base[args[0]] > 0.5] = (self.Base['BaseTiempoDetenido'+Chanc].loc[self.Base[args[0]] > 0.5]-1).astype('int')
        if Manual == False:
            FechasDeDetencion = list(self.Base['BaseTiempoDetenido'+Chanc][self.Base['BaseTiempoDetenido'+Chanc] > 0].index)
            Duraciones = list(self.Base['BaseTiempoDetenido'+Chanc][self.Base['BaseTiempoDetenido'+Chanc] > 0])
            for i,j in zip(FechasDeDetencion,Duraciones):
                DictDetenciones[str(i-datetime.timedelta(minutes=j))+'-'+str(i)] = self.Base[str(i-datetime.timedelta(minutes=j)):str(i-datetime.timedelta(minutes=1))]
        if Manual == True:
            DictDetenciones = dict()
            for Inicio,Fin in FechasDetenciones.items():
                DictDetenciones[Inicio + '-' + Fin] = self.Base[Inicio:Fin]
        return self.Base,DictDetenciones
    
    def Indices(self,Dict,*args):
        FechaInicio = list()
        FechaFin = list()
        Duracion = list()
        DiferenciaDeNivel = list()
        MeanTPHSag = list()
        MeanNivel = list()
        DiferenciaTPHCorreaAntesSP = list()
        NivelInicio = list()
        TiempoNivelMinimo = list()
        MeanTPHCV104 = list()
        SumFunCh3 = list()
        SumFunCh2 = list()
        SumFunCh5 = list()
        SumFunCh1 = list()
        TPHTotalDescChanInicio = list()
        TPHTotalDescChanMean = list()
        TPHCorreaAntesSPInicio = list()
        for value in Dict.values():
            if value.shape[0] > 0:
                FechaInicio.append(value[args[0]].index[0])
                FechaFin.append(value[args[0]].index[(value.shape[0]-1)])
                Duracion.append(value.shape[0])
                DiferenciaDeNivel.append(value[args[0]].iloc[0]-value[args[0]].min())
                MeanTPHSag.append(value[args[1]].mean())
                DiferenciaTPHCorreaAntesSP.append(value[args[2]].iloc[0]-value[args[2]].min())
                TPHCorreaAntesSPInicio.append(value[args[2]].iloc[0])
                NivelInicio.append(value[args[0]].iloc[0])
                TiempoNivelMinimo.append((value[args[0]].idxmin()-value[args[0]].index[0]).seconds/60)
                MeanNivel.append(value[args[0]].mean())
                SumFunCh3.append(np.sum(value[args[3]]))
                SumFunCh2.append(np.sum(value[args[4]]))
                SumFunCh5.append(np.sum(value[args[5]]))
                SumFunCh1.append(np.sum(value[args[6]]))
                MeanTPHCV104.append(value[args[2]].mean())
                TPHTotalDescChanInicio.append(value[args[7]].iloc[0])
                TPHTotalDescChanMean.append(value[args[7]].mean())
        Df = pd.DataFrame({'FechaInicio':FechaInicio,'FechaFin':FechaFin,'DiferenciaDeNivel':DiferenciaDeNivel,'Duracion':Duracion,'MeanTPHSag':MeanTPHSag,'MeanTPHCV104Inicio':TPHCorreaAntesSPInicio,'MeanTPHCV104':MeanTPHCV104,'DiferenciaTPHCorreaAntesSP':DiferenciaTPHCorreaAntesSP,
                          'NivelInicio':NivelInicio,'TiempoNivelMinimo':TiempoNivelMinimo,'MeanNivel':MeanNivel,'SumFunCh3':SumFunCh3,'SumFunCh2':SumFunCh2,
                          'SumFunCh5':SumFunCh5,'SumFunCh1':SumFunCh1,'TPHTotalDescChanInicio':TPHTotalDescChanInicio,'TPHTotalDescChanMean':TPHTotalDescChanMean})    
        return Df
    
    def AnalisisIndices(self,Indices,**args):
        BaseCortada = Indices[(Indices[list(args.keys())[0]]>=list(args.values())[0][0]) & (Indices[list(args.keys())[0]]<=list(args.values())[0][1]) & (Indices[list(args.keys())[1]]>=list(args.values())[1][0]) & (Indices[list(args.keys())[1]]<=list(args.values())[1][1]) & (Indices[list(args.keys())[2]]>=list(args.values())[2][0]) & (Indices[list(args.keys())[2]]<=list(args.values())[2][1])]
        return BaseCortada
        
        
    
    
    
    
#     def ImpactoEnConcentradora(self,Base,TiempoMinimo,Chanc,TiempoDespuesDeVolver,*args):
        ## Input Base,Tiempo Minimo que se analizara,string chancador,Tiempo que se deja despues de empezar a funcionar
        ## Se analizaran solo los tiempos mayores que el tiempo minimo
        ## La variable Tiempos es un vector de fechas desde que empezo la detencion cada 1 minuto hasta TiempoDespuesDeVolver minutos
#         ## retorna un diccionario donde cada valor es un data frame con id: numero de detencion,FechaIn:fecha real de detencion,FechaFin:FechaReal de Detencion,FechaIn2:fecha de detencion cada 1 minutos,FechaFin2 es la fecha de detencion cada 1 minutos. el maximo tiempo despues de una detencion son 1+TiempoDespuesDeVolver
#         TiempoDeDetencionMinima = TiempoMinimo
#         TDetencion = Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc]
#         n = Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc].shape[0]
#         print(n)
#         Niveles = dict()
#         for j in np.arange(10):
#             print(j)
#             FechaFin = Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc].index[j]-datetime.timedelta(minutes=1)
#             Fechaindet = Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc].index[j]-datetime.timedelta(minutes=Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc][j])
#             Fechaindet0 = Fechaindet
#             Tiempos = np.arange(0,Base[Base['BaseTiempoDetenido'+Chanc]>=TiempoDeDetencionMinima]['BaseTiempoDetenido'+Chanc][j]+TiempoDespuesDeVolver,1)
#             print('Detencion numero {} de {} hasta {}'.format(str(j),str(Fechaindet),str(FechaFin)))
#             print('-------------------------------------')
#             CorreaSp = list()
#             NivelProm = list()
#             TPHSag = list()
#             TiempoDeDetencion = list()
#             Id = list()
#             FechaIn = list()
#             FechaIn2 = list()
#             fecha_Fin = list()
#             fecha_Fin2 = list()
#             for i in np.arange(Tiempos.shape[0]):
#                 Fecha1Hora = Fechaindet+datetime.timedelta(minutes=1)
#                 FechaIn2.append(Fechaindet)
#                 fecha_Fin2.append(Fecha1Hora)
#                 CorSP = Base[str(Fechaindet):str(Fecha1Hora)][list(args)].describe().ix['mean'][args[0]]
#                 NProm = Base[str(Fechaindet):str(Fecha1Hora)][list(args)].describe().ix['mean'][args[1]]
#                 Tph = Base[str(Fechaindet):str(Fecha1Hora)][list(args)].describe().ix['mean'][args[2]]
#                 Fechaindet = Fecha1Hora
#                 CorreaSp.append(CorSP)
#                 NivelProm.append(NProm)
#                 TPHSag.append(Tph)
#                 TiempoDeDetencion.append(TDetencion[j])
#                 Id.append(j)
#                 FechaIn.append(Fechaindet0)
#                 fecha_Fin.append(FechaFin)
#             Niveles[j] = pd.DataFrame({'id':Id,'FechaIn':FechaIn,'FechaFin':FechaFin,
#                                        'FechaIn2':FechaIn2,'FechaFin2':fecha_Fin2,
#                                         args[0]:CorreaSp,
#                                         args[1]:NivelProm,args[2]:TPHSag,'TDet':TiempoDeDetencion})
#         #print(Niveles)
        return Niveles


    
    
    def AnalisisDeImpactosConcentradora(self,Niveles,*args):
        DiferenciaDeTPHCV104 = list()
        DiferenciaDeNivelLS1 = list()
        DiferenciaDeTPHLS1 = list()
        MinLugarTPHCV104 = list()
        MinLugarNivelLS1 = list()
        MinLugarTPHLS1 = list()
        MaxLugarTPHCV104 = list()
        MaxLugarNivelLS1 = list()
        MaxLugarTPHLS1 = list()
        InicioTPHCV104 = list()
        InicioNivelLS1 = list()
        InicioTPHLS1 = list()
        MaxTPHCV104 = list()
        MaxNivelLS1 = list()
        MaxTPHLS1 = list()
        MinTPHCV104 = list()
        MinNivelLS1 = list()
        MinTPHLS1 = list()
        MeanTPHCV104 = list()
        MeanNivelLS1 = list()
        MeanTPHLS1 = list()
        TiempoTotalDeDetencion = list()
        for k in np.arange(len(Niveles)):
            DiferenciaDeTPHCV104.append(Niveles[k][args[0]][0]-Niveles[k][args[0]][Niveles[k].shape[0]-1])
            DiferenciaDeNivelLS1.append(Niveles[k][args[1]][0]-Niveles[k][args[1]][Niveles[k].shape[0]-1])
            DiferenciaDeTPHLS1.append(Niveles[k][args[2]][0]-Niveles[k][args[2]][Niveles[k].shape[0]-1])
            MinLugarTPHCV104.append(Niveles[k][args[0]].idxmin())
            MinLugarNivelLS1.append(Niveles[k][args[1]].idxmin())
            MinLugarTPHLS1.append(Niveles[k][args[2]].idxmin())
            TiempoTotalDeDetencion.append(Niveles[k].shape[0])
            InicioTPHCV104.append(Niveles[k][args[0]][0])
            InicioNivelLS1.append(Niveles[k][args[1]][0])
            InicioTPHLS1.append(Niveles[k][args[2]][0])
            MinTPHCV104.append(Niveles[k][args[0]].min())
            MinNivelLS1.append(Niveles[k][args[1]].min())
            MinTPHLS1.append(Niveles[k][args[2]].min())
            MeanTPHCV104.append(Niveles[k][args[0]].mean())
            MeanNivelLS1.append(Niveles[k][args[1]].mean())
            MeanTPHLS1.append(Niveles[k][args[2]].mean())


        ChAnalisis = pd.DataFrame({'DiferenciaDeTPHCV104':DiferenciaDeTPHCV104,'DiferenciaDeNivelLS1':DiferenciaDeNivelLS1,
                                    'DiferenciaDeTPHLS1':DiferenciaDeTPHLS1,'MinLugarTPHCV104':MinLugarTPHCV104,
                                    'MinLugarNivelLS1':MinLugarNivelLS1,'MinLugarTPHLS1':MinLugarTPHLS1,
                                    'TiempoTotalDeDetencion':TiempoTotalDeDetencion,'InicioTPHCV104':InicioTPHCV104,
                                   'InicioNivelLS1':InicioNivelLS1,'InicioTPHLS1':InicioTPHLS1,'MinTPHCV104':MinTPHCV104,
                                   'MinNivelLS1':MinNivelLS1,'MinTPHLS1':MinTPHLS1,'MeanTPHCV104':MeanTPHCV104,
                                    'MeanNivelLS1':MeanNivelLS1,'MeanTPHLS1':MeanTPHLS1})
        
        return ChAnalisis
    
    def DescribeChancadoras(self,Base,LimitesTPHSag,LimitesTiempoDetencionChanc):
        BaseAnalisis = Base[(Base['MeanTPHLS1']>=LimitesTPHSag[0])&(Base['MeanTPHLS1']<=LimitesTPHSag[1])&(Base['TiempoTotalDeDetencion']<=LimitesTiempoDetencionChanc[1])&(Base['TiempoTotalDeDetencion']>LimitesTiempoDetencionChanc[0])].describe()
        return BaseAnalisis
        


        
        
