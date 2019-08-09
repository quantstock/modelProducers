"""
data query modulus
author: wenping lo
last updated: 2019/8/6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import datetime
import sys

class Data(object):
    def __init__(self):
        self.__connect_db()

    def get_df_from_db(self, collection_name, stockId, startTime, endTime):
        temp_df = pd.DataFrame(list(self.db[collection_name].find(
             {"timestamp":{"$gte":startTime, "$lte":endTime},
              "stockId":stockId})))  # selection criterion
        temp_df = temp_df.drop(columns=["_id", "stockId"]).drop_duplicates("timestamp", keep="first").set_index("timestamp")
        temp_df = self.__parse_df_to_float(temp_df)
        try:
            temp_df = temp_df.drop(columns=["證券名稱"])
            return temp_df
        except:
            temp_df = temp_df.drop(columns=["股票名稱"])
            return temp_df
        else:
            return temp_df

    def get_singleDailyBrokerPoints_df(self, stockId, startTime, endTime):
        temp_df = pd.DataFrame(list(self.db["dailyBrokerPoints"].find({"stockId":stockId}))).drop(columns="_id")
        temp_df[['均價', '買價', '買賣超', '買量', '賣價', '賣量']] = temp_df[['均價', '買價', '買賣超', '買量', '賣價', '賣量']].astype("float")

        if not self.foreign_broker_list:
            broker_name_df = pd.DataFrame(list(self.db["券商代號表"].find()))
            self.foreign_broker_list = broker_name_df.loc[broker_name_df["類別"] != "本土券商", "券商名稱"].dropna().to_list()+ ["台灣巴克萊"]
        temp_df["類別"] = temp_df["券商名稱"].apply(lambda x: "外資" if x in self.foreign_broker_list else "台資")
        temp_df = temp_df.set_index("timestamp")

        return df

    # def get_multiDailyOHLCV_df(self, stockIdList, startTime, endTime):
        stockIdDictList = [{"stockId": s} for s in stockIdList]
        temp_df = pd.DataFrame(list(self.db["dailyPrice"].find(
                 {"$or":stockIdDictList, "timestamp": {"$gte": startTime, "$lte": endTime}},  # selection criterion
                 {"timestamp": "-1",
                  "stockId": "1",
                  "成交股數": "1",
                  "收盤價": "1",
                  "最低價": "1",
                  "最高價": "1",
                  "開盤價": "1"})))
        # 處理dataframe
        temp_df = temp_df.drop(columns="_id")#.set_index("timestamp")
        dfs = []
        for stockId in stockIdList:
            dfs.append(temp_df.loc[temp_df["stockId"] == stockId].drop_duplicates("timestamp", keep="first"))
        temp_df = pd.concat(dfs, axis=0)

        # 重新命名OHLCV
        temp_df = temp_df.rename(columns = {
            "成交股數": "volume",
            "收盤價": "close",
            "最低價": "low",
            "最高價": "high",
            "開盤價": "open"})
        # # 將df命名為stockId
        # temp_df.name = stockId
        # 將str轉為float
        # temp_df["volume"] = temp_df["volume"].apply(lambda x: x.replace(',', '')).astype(float)
        temp_df = self.__parse_df_to_float(temp_df)
        return temp_df

    def get_DailyOHLCV_df(self, stockIdList, startTime, endTime):
        """ fn: 獲取標的(們)的日資料的開高低收與交易量
            input:
                stockIdList: 可為string(例如"2330")，或是list(例如["2330", "2317", ...])
                startTime: python datetime格式
                endTime: python datetime格式
            output:
                pandas DataFrame; index為時間(timestamp)，columns為multilevleultilevle
                第一個column為stockId, 第二個column為open, high, low, close, volume
            """
        if type(stockIdList) is str:
            stockId = stockIdList
            return self.__get_singleDailyOHLCV_df(stockId, startTime, endTime)
        elif type(stockIdList) is list:
            sub_dfs = [self.__get_singleDailyOHLCV_df(stockId, startTime, endTime) for stockId in stockIdList]
            return pd.concat(sub_dfs, axis=1)
        else:
            raise TypeError("the variable stockIdList should be a list (or a string is also acceptable)")

    def get_DailyChips_df(self, stockIdList, startTime, endTime):
        """ fn: 獲取標的日資料的籌碼面資訊
            input:
                stockIdList: 可為string(例如"2330")，或是list(例如["2330", "2317", ...])
                startTime: python datetime格式
                endTime: python datetime格式
            output:
                pandas DataFrame; index為時間(timestamp)，columns為multilevleultilevle
                第一個column為stockId, 第二個column為籌碼面的各項資訊
            """
        if type(stockIdList) is str:
            stockId = stockIdList
            return self.__get_singleDailyChips_df(stockId, startTime, endTime)
        elif type(stockIdList) is list:
            sub_dfs = [self.__get_singleDailyChips_df(stockId, startTime, endTime) for stockId in stockIdList]
            return pd.concat(sub_dfs, axis=1)
        else:
            raise TypeError("the variable stockIdList should be a list (or a string is also acceptable)")

    def __get_singleDailyOHLCV_df(self, stockId, startTime, endTime):
        temp_df = pd.DataFrame(list(self.db["dailyPrice"].find(
                 {"timestamp": {
                     "$gte": startTime, "$lte": endTime},
                     "stockId":stockId},  # selection criterion
                 {"timestamp": "-1",
                  "成交股數": "1",
                  "收盤價": "1",
                  "最低價": "1",
                  "最高價": "1",
                  "開盤價": "1"})))
        # 處理dataframe
        temp_df = temp_df.drop(columns="_id").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        # 重新命名OHLCV
        temp_df = temp_df.rename(columns = {
            "成交股數": "volume",
            "收盤價": "close",
            "最低價": "low",
            "最高價": "high",
            "開盤價": "open"})
        # 將str轉為float
        temp_df = self.__parse_df_to_float(temp_df)
        ### create multilevel column
        cols = list(temp_df.columns)
        stockIds = [stockId]*len(cols)
        temp_df.columns = pd.MultiIndex.from_tuples(zip(stockIds, cols))
        return temp_df

    def __get_singleDailyChips_df(self, stockId, startTime, endTime):
        chip_collections = [
            'dailyCreditTrading',
            'dailyFundTrading',
            'dailyDayTrading',
            'dailyOddLots',
            'dailyStockLending']
        temp_dfs = []
        for collections in chip_collections:
            temp_df = self.get_df_from_db(collections, stockId, startTime, endTime)
            temp_dfs.append(temp_df)
        temp_df = pd.concat(temp_dfs, axis=1)

        temp_df = temp_df.drop(columns=
            ['借券賣出',
            '借券賣出今日餘額',
            '借券賣出可使用額度',
            '借券賣出庫存異動'])
        #處理三大法人資料不連續
        if "dailyFundTrading" in chip_collections:
            temp_df = self.__merge_df_columns(temp_df, "自營商賣出股數", '自營商賣出股數自行買賣', '自營商賣出股數避險')
            temp_df = self.__merge_df_columns(temp_df, "自營商買進股數", '自營商買進股數自行買賣', '自營商買進股數避險')
            temp_df = self.__merge_df_columns(temp_df, "外資賣出股數", '外資自營商賣出股數', '外陸資賣出股數不含外資自營商')
            temp_df = self.__merge_df_columns(temp_df, "外資買進股數", '外資自營商買進股數', '外陸資買進股數不含外資自營商')
            temp_df = self.__merge_df_columns(temp_df, "外資買賣超股數", '外資自營商買賣超股數', '外陸資買賣超股數不含外資自營商')
        ### create multilevel column
        cols = list(temp_df.columns)
        stockIds = [stockId]*len(cols)
        temp_df.columns = pd.MultiIndex.from_tuples(zip(stockIds, cols))

        return temp_df

    def __parse_close(self, x):
        try: return pd.to_numeric(x["收盤價"].replace(",", ""))
        except ValueError:
            try: return (pd.to_numeric(x["最後揭示賣價"].replace(",", "")) + pd.to_numeric(x["最後揭示賣價"].replace(",", "")))/2
            except ValueError:
                try: return pd.to_numeric(x["最後揭示賣價"].replace(",", ""))
                except ValueError:
                    try: return pd.to_numeric(x["最後揭示買價"].replace(",", ""))
                    except ValueError:
                        return np.nan

    def __parse_df_to_float(self, df):
        # make all dataframe into float; nan will remain
        def __parse_to_float(x):
            try:
                return float(x.replace(",", ""))
            except:
                return x
        for c in df.columns:
            df[c] = df[c].apply(__parse_to_float)
        return df

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

    def __merge_df_columns(self, df, target_c, root_c1, root_c2):
        df[target_c] = pd.concat(
            [df[target_c].dropna(),
            df[root_c1].dropna().replace(np.nan, 0) +
            df[root_c2].dropna().replace(np.nan, 0)],
            axis=0)
        return df

def plot_columns_time(temp_df):
    #畫出每個columns的時間軸
    df_c = temp_df.columns
    temp_df.columns= [str(s) for s in range(len(df.columns))]
    # create a (constant) Series for each sensor
    for i, sym in enumerate(df.columns):
        t_range = df[[sym]].dropna().index
        dff = t_range.to_series().apply(lambda x: i if x >= t_range.min() and x <= t_range.max() else numpy.NaN)

        p = dff.plot(ylim=[0, len(df.columns)], legend=False)
        p.set_yticks(range(len(df.columns)))
        p.set_yticklabels(df.columns)
    return df_c


if __name__ == '__main__':
    data = Data()
    stockId = "2330"
    stockIdList = ["2330", "2317", "0050", "0056", "1101", "2412"]
    startTime = datetime.datetime(2016, 1, 1)
    endTime = datetime.datetime(2019, 8, 6)

    # df = data.get_DailyOHLCV_df(stockId, startTime, endTime)
    # df = data.get_DailyOHLCV_df(stockIdList, startTime, endTime)
    # df = data.get_DailyChips_df(stockIdList, startTime, endTime)
    # df = data.get_singleDailyBrokerPoints_df(stockId, startTime, endTime)
