from modelProducers.data import Data
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import talib
from talib import abstract

class featureExtractor(object):
    def __init__(self, bool_PCA=False, nPCA=5):
        self.Data = Data()
        self.PCA = bool_PCA
        self.nPCA = nPCA

    def techAssemble(self, stockId, startTime, endTime):
        # 我們想要用歷史的價格去預測明天的漲跌符號，如果是上漲(標記為+1)，如果是下跌(標記為-1)。<br>
        # 所以這是個**二元分類**問題。我們有許多模型可以使用，linear models/SVM models/tree-based models/KNNs.
        def talib2df(talib_output):
            if type(talib_output) == list:
                df = pd.DataFrame(talib_output).transpose()
            else:
                df = pd.Series(talib_output)
            df.index = self.tsmc['close'].index
            return df

        def get_RANGE_label(series, upper, lower):
            temp_label = series.copy()
            for i, t in enumerate(series.index):
                if np.isnan(series.iloc[i]) :
                    temp_label.iloc[i] = np.nan
                elif series.iloc[i] > upper:
                    temp_label.iloc[i] = -1
                elif series.iloc[i] < lower:
                    temp_label.iloc[i] = + 1
                else:
                    if temp_label.iloc[i] - temp_label.iloc[i-1] > 0:
                        temp_label.iloc[i] = +1
                    else:
                        temp_label.iloc[i] = -1
            return temp_label
        # ### OHLCV

        ohlcv_df = self.Data.get_dailyOHLCV(stockId, startTime, endTime)
        ohlcv_df = ohlcv_df.shift(1).dropna()

        self.tsmc = {
            'close':ohlcv_df["close"],
            'open':ohlcv_df["open"],
            'high': ohlcv_df["high"],
            'low':  ohlcv_df["low"],
            'volume': ohlcv_df["volume"],
        }

        # ## 特徵工程

        # 在這一步，我們要創造出features，好的feature帶模型上天堂，壞feature帶你的模型...?<br>
        # 以下我們實作這篇[論文](https://www.sciencedirect.com/science/article/pii/S0957417414004473) 的方法：利用技術分析的指標作為特徵：技術分析指標可以作為是買進或賣出的訊號，我們將買進標記為+1，賣出標記為-1。<br>

        features=pd.DataFrame()

        # ### 移動平均指標

        for day in [5, 10, 20, 30, 60]:
            MA = talib2df(abstract.MA(self.tsmc, timeperiod=day))
            WMA = talib2df(abstract.WMA(self.tsmc, timeperiod=day))
            features["X_SMA_%d"%day] = (ohlcv_df["close"] > MA).apply(lambda x: 1.0 if x else -1.0) #創造出特徵
            features["X_WMA_%d"%day] = (ohlcv_df["close"] > WMA).apply(lambda x: 1.0 if x else -1.0) #創造出特徵

        # ### 動量指標

        MOM = talib2df(abstract.MOM(self.tsmc, timeperiod=10))
        STOCH = talib2df(abstract.STOCH(self.tsmc))
        RSI = talib2df(abstract.RSI(self.tsmc))
        STOCHRSI = talib2df(abstract.STOCHRSI(self.tsmc))
        MACD = talib2df(abstract.MACD(self.tsmc))
        WILLR = talib2df(abstract.WILLR(self.tsmc))
        CCI = talib2df(abstract.CCI(self.tsmc))
        RSI = talib2df(abstract.RSI(self.tsmc))

        # #### 轉換成特徵

        features["X_MOM"] = MOM.apply(lambda x: 1.0 if x > 0  else -1.0)
        features["X_WILLR"]  = (WILLR - WILLR.shift()).apply(np.sign)

        features["X_STOCH_0"] = (STOCH[0] -  STOCH[0].shift()).apply(np.sign)
        features["X_STOCH_1"] = (STOCH[1] -  STOCH[1].shift()).apply(np.sign)

        features["X_MACD_0"] = (MACD[0] -  MACD[0].shift()).apply(np.sign)
        features["X_MACD_1"] = (MACD[1] -  MACD[1].shift()).apply(np.sign)
        features["X_MACD_2"] = (MACD[2] -  MACD[2].shift()).apply(np.sign)

        # #### 震盪指標需要特殊處理

        features["X_STOCHRSI_0"] = get_RANGE_label(STOCHRSI[0], upper = 70, lower = 30)
        features["X_STOCHRSI_1"] = get_RANGE_label(STOCHRSI[1], upper = 70, lower = 30)
        features["X_CCI"] = get_RANGE_label(CCI, upper=200, lower=-200)
        features["X_RSI"] = get_RANGE_label(RSI, upper=70, lower=30)

        # ### 交易量指標

        ADOSC = talib2df(abstract.ADOSC(self.tsmc))
        OBV = talib2df(abstract.OBV(self.tsmc))
        features["X_ADOSC"]  = (ADOSC - ADOSC.shift()).apply(np.sign)
        features["X_OBV"]  = (OBV - OBV.shift()).apply(np.sign)

        features = features.dropna() #把nan的資料丟掉

        # ### create multilevel column

        cols = list(features.columns)
        stockIds = [stockId]*len(cols)
        features.columns = pd.MultiIndex.from_tuples(zip(stockIds, cols))

        return self.__classMethod(features)

    def __PCA(self, features):
        stockId = features.columns[0][0]
        X = features[stockId]
        pca = PCA(n_components=self.nPCA)
        X = pd.DataFrame(pca.fit_transform(X), index=features.index)

        # ### create multilevel column

        cols = list(X.columns)
        stockIds = [stockId]*len(X)
        X.columns = pd.MultiIndex.from_tuples(zip(stockIds, cols))

        return X

    def __classMethod(self, features):
        if self.PCA:
            return self.__PCA(features)
        else:
            return features
