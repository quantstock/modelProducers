from modelProducers.data import Data
from datetime import datetime
import pandas as pd
import numpy as np

class labelProducer(object):
    def __init__(self):
        self.Data = Data()

    def pct_change(self, stockId, startTime, endTime, period=5):

        # get data
        ohlcv_df = self.Data.get_dailyOHLCV(stockId, startTime, endTime)

        # ## 造出標籤 (labels)
        label = ohlcv_df.copy().dropna() # deep copy
        label = label[["close"]].pct_change(period).apply(np.sign).shift(-1)

        # ### create multilevel column

        label.columns = pd.MultiIndex.from_tuples([(stockId, 'Y')])

        return label
