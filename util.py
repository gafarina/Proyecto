import pandas as pd
import yfinance as yf
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
from keras import backend as K
class ReadData():
    def GetDataFromYahoo(self,Ticker,start,end,interval):
        #try:
        data = yf.download(Ticker,start=start,end=end,interval=interval)
        return data
        #except:
        #print('Error El Ticker no existe')
    
    def GetDataFromYahooSeveral(self,start,end,interval,columna,*args):
            Base = pd.DataFrame()
            for i in args:
                assert isinstance(i,str),'The value is not string'
                #try:
                data = self.GetDataFromYahoo(i,start,end,interval)[columna]
                Base = pd.merge(Base,data,how='outer',left_index=True,right_index=True)
                Base = Base.fillna(method='ffill')
                #except:
                #print("El Ticker no existe o no exsten datos en el Ticker %s"%(i))
            Base.columns = args


            return Base

class LSTMClass():
    def __init__(self,Base,batch_size,epochs,lr,time_steps,MinMaxScaler):
        self.Base = Base
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.time_steps = time_steps
        self.min_max_scaler = MinMaxScaler
        
    def Train_Test(self,Traincols,PropTrain_set,MinMax=True):
        train_cols = Traincols
        df_train,df_test = train_test_split(self.Base, train_size=PropTrain_set, 
                                            test_size=1-PropTrain_set,shuffle=False)
        x_train = df_train.loc[:,train_cols].values
        x_test = df_test.loc[:,train_cols].values
        
        
        if MinMax:
            x = df_train.loc[:,train_cols].values
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
    
    def Model(self,x_train,NumNeu,DropOut):
        lstm_model = Sequential()
        lstm_model.add(LSTM(NumNeu,batch_input_shape=(self.batch_size, self.time_steps,
                                                   x_train.shape[2]), dropout=0.0, recurrent_dropout=0.0,
                                                   stateful=True,kernel_initializer='random_uniform',
                                                   return_sequences=True))
        lstm_model.add(LSTM(NumNeu,batch_input_shape=(self.batch_size, self.time_steps,
                                                   x_train.shape[2]), dropout=0.0, recurrent_dropout=0.0,
                                                   stateful=True,kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(DropOut))
        lstm_model.add(Dense(1,activation='relu'))
        optimizer = optimizers.RMSprop(lr=self.lr)
        # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return lstm_model
    
    def SplitTrainValTest(self,x_train,x_test,pos_Y):
        x_t,y_t = self.BuildTimeSeries(x_train,pos_Y)
        #x_t = self.TrimData(x_t)
        #y_t = self.TrimData(y_t)
        print("Train size",x_t.shape, y_t.shape)
        x_temp,y_temp = self.BuildTimeSeries(x_test,pos_Y)
        x_temp = x_temp[x_temp.shape[0]%self.batch_size:,]
        y_temp = y_temp[y_temp.shape[0]%self.batch_size:]
        #x_val, x_test_t = np.split(self.TrimData(x_temp),2)
        #x_val = np.split(self.TrimData(x_temp),2)
        #y_val, y_test_t = np.split(self.TrimData(y_temp),2)
        #y_val = np.split(self.TrimData(y_temp),2)
        #print("Test size", x_temp.shape, y_temp.shape, x_val.shape, y_val.shape)
        print("Test size",x_temp.shape,y_temp.shape)
        #print("Val size",len(x_val), len(y_val))
        #return x_t,y_t,x_val,x_test_t,y_val,y_test_t
        return x_t,y_t,x_temp,y_temp
    
    def TrainModel(self,x_t,y_t,x_val,y_val,
                   NumNeuron,Dropout,OutputPath,TrainingModel=True,file=1):
        x_t = x_t[x_t.shape[0]%self.batch_size:,]
        print(x_t.shape)
        y_t = y_t[y_t.shape[0]%self.batch_size:]
        print(y_t.shape)
        
        model = None
        try:
            model = pkl.load(open("lstm_model", 'rb'))
            print("Loaded saved model...")
        except FileNotFoundError:
            print("Model not found")
        
        is_update_model = TrainingModel
        if model is None or is_update_model:
            print("Building model...")
            print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
            model = self.Model(x_t,NumNeuron,Dropout)
            es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=70,min_delta=0.0001)
            mcp = ModelCheckpoint(os.path.join('C:/Users/gasto/OneDrive/Trabajo2/',"best_model_"+str(file)+".h5"), 
                                  monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False, mode='min', period=1)
            #r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
            #                              patience=15, verbose=0, mode='auto',min_delta=0.0001,
            #                              cooldown=0, min_lr=0)
            history = model.fit(x_t,y_t,epochs=self.epochs, verbose=2, batch_size=self.batch_size,
                                          shuffle=False, validation_data=(x_val,
                                          y_val), callbacks=[es])
            print("saving model...")
            plt.figure()
            Plot = Graficos(history.history)
            Plot.GraficoTimeSeriesEnLaMismaEscala({'Titulo':['Loss vs Val Loss',25],
                                                   'xlabel':['Epocas',20],'ylabel':['Error',20]},
                                                   loss=['red','Train_Loss',1.0,2,'-'],
                                                   val_loss=['blue','Validation_Loss',0.5,2,'-'])
            pkl.dump(model, open(OutputPath, "wb"))
        return model
    
    def Prediction(self,model,x_test_t,y_test_t):
        y_pred = model.predict(self.TrimData(x_test_t),batch_size=self.batch_size)
        y_pred = y_pred.flatten()
        y_test_t = self.TrimData(y_test_t)
        error = mean_squared_error(y_test_t,y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        y_pred_orig = (y_pred * self.min_max_scaler.data_range_[5]) + self.min_max_scaler.data_min_[5]
        y_test_t_orig = (y_test_t * self.min_max_scaler.data_range_[5]) + self.min_max_scaler.data_min_[5]
        BasePred = pd.DataFrame({'Real':y_test_t_orig,'Prediccion':y_pred_orig})
        
        Plot = Graficos(BasePred)
        Plot.GraficoTimeSeriesEnLaMismaEscala({'Titulo':['Prediccion vs TPH Real ',25],
                                               'xlabel':['Tiempo',20],'ylabel':['TPH',20]},
                                               Real=['red','Real',1.0,2,'-'],
                                               Prediccion=['blue','Prediccion',0.5,2,'-'])
        return model,BasePred
    
    def PredictionNewData(self,model,BaseTest,TrainCols,pos_Y,batch_size):
        x = BaseTest.loc[:,TrainCols].values
        self.min_max_scaler.fit_transform(x)
        x_test = self.min_max_scaler.transform(BaseTest.loc[:,TrainCols])
        x_t,y_t = self.BuildTimeSeries(x_test,pos_Y)
        y_pred = model.predict(x_t,batch_size=batch_size)
        error = mean_squared_error(y_t,y_pred)
        print("Error is", error)
        y_pred_orig = (y_pred * self.min_max_scaler.data_range_[5]) + self.min_max_scaler.data_min_[5]
        y_test_t_orig = (y_t * self.min_max_scaler.data_range_[5]) + self.min_max_scaler.data_min_[5]
        print(y_pred_orig.flatten())
        print(y_test_t_orig)
        Datos = pd.DataFrame({'y_pred':y_pred_orig.flatten(),'y_real':y_test_t_orig.flatten()})
        Datos.index = list(BaseTest.index)[10:110]
        Datos = Datos['2020-01-01':'2020-01-31']
        Plot = Graficos(Datos)
        Plot.GraficoTimeSeriesEnLaMismaEscala({'Titulo':['Prediccion vs Precio Real ',25],
                                               'xlabel':['Tiempo',20],'ylabel':['Precio',20]},
                                               y_real=['red','Real',1.0,2,'-'],
                                               y_pred =['blue','Prediccion',0.5,2,'-'])
        return Datos
        
        
    
    
        
        
#         #'C:/Users/gasto/OneDrive/Trabajo2/lstm_model.h5'

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

    
