#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import talib
from talib import abstract
from datetime import datetime, timedelta
from pymongo import MongoClient

class modelProducer(object):
    def __init__(self, stockId, startTime, endTime, period=5):
        self.db = self.__connect_db()
        self.stockId = stockId
        self.startTime = startTime
        self.endTime = endTime
        self.period = period # label period
    def get_model(self):
        self.model = self.__get_model()
        return self.model

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        return MongoClient(mongo_uri)[dbName]

    def __get_model(self):
        # 我們想要用歷史的價格去預測明天的漲跌符號，如果是上漲(標記為+1)，如果是下跌(標記為-1)。<br>
        # 所以這是個**二元分類**問題。我們有許多模型可以使用，linear models/SVM models/tree-based models/KNNs.

        # ## 讀取資料
        self.price_df = pd.DataFrame(list(self.db["dailyPrice"].find({"timestamp": {"$gt": self.startTime, "$lt": self.endTime}, "stockId":self.stockId},{"timestamp": 1, "成交股數": 1, '最高價': 1, '最低價': 1, '收盤價': 1, '開盤價':1 }))).drop(columns="_id").drop_duplicates("timestamp").set_index("timestamp")

        # ## 清理資料

        self.price_df = self.price_df[~self.price_df.index.duplicated()] # 將重複的時間戳拿掉
        for c in ['成交股數', '最高價', '最低價', '收盤價', '開盤價']:
            self.price_df[c] = self.price_df[c].apply(self.__parse_to_numeric)

        # ### 整理出OHLCV

        tsmc_ohlcv_df = self.price_df[[ '成交股數', '最高價', '最低價', '收盤價',  '開盤價']]
        tsmc_ohlcv_df.columns = ["volume", "high", "close", "low", "open"]
        tsmc_ohlcv_df = tsmc_ohlcv_df.append(pd.Series(name=tsmc_ohlcv_df.index[-1] + timedelta(1)))

        tsmc_ohlcv_df = tsmc_ohlcv_df.shift(1).dropna()
        self.tsmc = {
            'close':tsmc_ohlcv_df["close"],
            'open':tsmc_ohlcv_df["open"],
            'high': tsmc_ohlcv_df["high"],
            'low':  tsmc_ohlcv_df["low"],
            'volume': tsmc_ohlcv_df["volume"],
        }

        # ## 造出標籤 (labels)
        tsmc_feature = tsmc_ohlcv_df.copy().dropna() # deep copy
        tsmc_feature["Y"] = tsmc_feature["close"].pct_change(self.period).apply(np.sign).shift(-1)

        # ## 特徵工程

        # 在這一步，我們要創造出features，好的feature帶模型上天堂，壞feature帶你的模型...?<br>
        # 以下我們實作這篇[論文](https://www.sciencedirect.com/science/article/pii/S0957417414004473) 的方法：利用技術分析的指標作為特徵：技術分析指標可以作為是買進或賣出的訊號，我們將買進標記為+1，賣出標記為-1。<br>

        # ### 移動平均指標

        for day in [5, 10, 20, 30, 60]:
            tsmc_MA = self.__talib2df(abstract.MA(self.tsmc, timeperiod=day))
            tsmc_WMA = self.__talib2df(abstract.WMA(self.tsmc, timeperiod=day))
            tsmc_feature["X_SMA_%d"%day] = (tsmc_ohlcv_df["close"] > tsmc_MA).apply(lambda x: 1.0 if x else -1.0) #創造出特徵
            tsmc_feature["X_WMA_%d"%day] = (tsmc_ohlcv_df["close"] > tsmc_WMA).apply(lambda x: 1.0 if x else -1.0) #創造出特徵

        # ### 動量指標

        tsmc_MOM = self.__talib2df(abstract.MOM(self.tsmc, timeperiod=10))
        tsmc_STOCH = self.__talib2df(abstract.STOCH(self.tsmc))
        tsmc_RSI = self.__talib2df(abstract.RSI(self.tsmc))
        tsmc_STOCHRSI = self.__talib2df(abstract.STOCHRSI(self.tsmc))
        tsmc_MACD = self.__talib2df(abstract.MACD(self.tsmc))
        tsmc_WILLR = self.__talib2df(abstract.WILLR(self.tsmc))
        tsmc_CCI = self.__talib2df(abstract.CCI(self.tsmc))
        tsmc_RSI = self.__talib2df(abstract.RSI(self.tsmc))

        # #### 轉換成特徵

        tsmc_feature["X_MOM"] = tsmc_MOM.apply(lambda x: 1.0 if x > 0  else -1.0)
        tsmc_feature["X_WILLR"]  = (tsmc_WILLR - tsmc_WILLR.shift()).apply(np.sign)

        tsmc_feature["X_STOCH_0"] = (tsmc_STOCH[0] -  tsmc_STOCH[0].shift()).apply(np.sign)
        tsmc_feature["X_STOCH_1"] = (tsmc_STOCH[1] -  tsmc_STOCH[1].shift()).apply(np.sign)

        tsmc_feature["X_MACD_0"] = (tsmc_MACD[0] -  tsmc_MACD[0].shift()).apply(np.sign)
        tsmc_feature["X_MACD_1"] = (tsmc_MACD[1] -  tsmc_MACD[1].shift()).apply(np.sign)
        tsmc_feature["X_MACD_2"] = (tsmc_MACD[2] -  tsmc_MACD[2].shift()).apply(np.sign)

        # #### 震盪指標需要特殊處理

        tsmc_feature["X_STOCHRSI_0"] = self.__get_RANGE_label(tsmc_STOCHRSI[0], upper = 70, lower = 30)
        tsmc_feature["X_STOCHRSI_1"] = self.__get_RANGE_label(tsmc_STOCHRSI[1], upper = 70, lower = 30)
        tsmc_feature["X_CCI"] = self.__get_RANGE_label(tsmc_CCI, upper=200, lower=-200)
        tsmc_feature["X_RSI"] = self.__get_RANGE_label(tsmc_RSI, upper=70, lower=30)

        # ### 交易量指標

        tsmc_ADOSC = self.__talib2df(abstract.ADOSC(self.tsmc))
        tsmc_OBV = self.__talib2df(abstract.OBV(self.tsmc))
        tsmc_feature["X_ADOSC"]  = (tsmc_ADOSC - tsmc_ADOSC.shift()).apply(np.sign)
        tsmc_feature["X_OBV"]  = (tsmc_OBV - tsmc_OBV.shift()).apply(np.sign)

        # ## 建模 Modeling

        tsmc_feature = tsmc_feature.dropna() #把nan的資料丟掉
        tsmc_feature = tsmc_feature[tsmc_feature["Y"] != 0] #只選取return有變動的

        timestamps = tsmc_feature.index

        N = 2800
        n = 100
        fea = [x for x in tsmc_feature.columns if "X" in x] #選取feature的columns

        X = tsmc_feature[fea]
        y = tsmc_feature["Y"]

        RANDOM_STATE = 2300
        # RANDOM_STATE = 23000

        # ----------------
        pca = PCA(n_components = 3)
        X = pca.fit_transform(X)

        X_train = X[:N]
        y_train = y[:N]

        X_test = X[N+n:]
        y_test = y[N+n:]

        # ----------------

        n_estimator = 1000
        # n_estimator = 10000
        clf = RandomForestClassifier(max_depth=4, n_estimators=n_estimator,
                                    criterion='entropy', random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        return clf, proba, pred, timestamps


    def __parse_to_numeric(self, x):
        try: return pd.to_numeric(x.replace(',', ''), downcast='float')
        except: return np.nan

    def __talib2df(self, talib_output):
        if type(talib_output) == list:
            df = pd.DataFrame(talib_output).transpose()
        else:
            df = pd.Series(talib_output)
        df.index = self.tsmc['close'].index
        return df

    def __get_RANGE_label(self, series, upper, lower):
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
