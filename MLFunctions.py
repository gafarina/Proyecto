import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

class PrincipalComponentAnalysis():
    def __init__(self,Base):
        self.Base = Base
    def PcaStand(self,Componentes,VarX,VarY):
        BaseX = np.array(self.Base[VarX])
        BaseY = np.array(self.Base[VarY])
        scaler =  StandardScaler()
        scaler.fit(BaseX)
        TrainX = scaler.transform(BaseX)
        pca = PCA(Componentes)
        pca.fit(TrainX)
        TrainX = pca.transform(TrainX)
        BaseNueva = pd.concat([pd.DataFrame(TrainX),pd.DataFrame(BaseY)],axis=1)
        BaseNueva['Date'] = list(self.Base.index)
        BaseNueva = BaseNueva.set_index('Date')
        comp = []
        for i in np.arange(Componentes):
            comp.append('Comp'+str(i))
        comp.append('Target')
        BaseNueva.columns = comp
        ExplainedVariance = pca.explained_variance_ratio_
        SumExplainedVariance = np.sum(pca.explained_variance_ratio_)
        return BaseNueva,ExplainedVariance,SumExplainedVariance
        
        
class RandomForestModel():
    def __init__(self,Base,n_estimator,max_depth,random_state,Classifier):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.random_state = random_state
        if Classifier == False:
            self.model = RandomForestRegressor(n_estimators=self.n_estimator,max_depth=self.max_depth,random_state=self.random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=self.n_estimator,max_depth=self.max_depth,random_state=self.random_state)
            
        self.Base=Base
    
    def TrainTest(self,Base,test_size,ListX,ListY):
        X = np.array(Base[ListX])
        Y = np.array(Base[ListY])
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

# scaler = StandardScaler()
# scaler.fit(train_X)
# TrainX = scaler.transform(train_X)
# pca = PCA(6)
# pca.fit(TrainX)
# TrainX = pca.transform(TrainX)
# BaseNueva = pd.concat([pd.DataFrame(TrainX),pd.DataFrame(train_Y)],axis=1)