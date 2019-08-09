import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/home/wenping/wp_stock/backTesters')
sys.path.append('/home/wenping/wp_stock/stgyProducers')
sys.path.append('/home/wenping/wp_stock/modelProducers')
from data import Data


class FeatureExtractor(object):
    def __init__(self):
        self.data = Data()

    def get_multiTechnicalFeatures_df


if __name__ == '__main__':
    fe = FeatureExtractor()
    dic = fe.get_multiTechnicalFeatures_dict(stockIdList, startTime, endTime)
