import pandas as pd
import yfinance as yf
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
