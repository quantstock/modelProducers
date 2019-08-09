from tqdm import tqdm, tqdm_notebook
import pymongo
from pymongo import MongoClient
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from multiprocessing import cpu_count, pool
import time
import sys
sys.path.append('/home/wenping/wp_stock/backTesters')
sys.path.append('/home/wenping/wp_stock/stgyProducers')
sys.path.append('/home/wenping/wp_stock/modelProducers')
from backtest import BackTest
from data import Data


class Label(object):

    def __init__(self):
        self.data = Data()

    def get_TripleLabeling_df(self,stockId, startTime, endTime, dailyVolSpan0=100,VerticalBarrierDays=5,minRet=0.01,labelColumn="close",cpus=8,ptSl=[1,1]):
        df = self.data.get_singleDailyOHLCV_df(stockId, startTime, endTime)
        series = df[labelColumn]
        target = self.getDailyVol(series, span0=dailyVolSpan0) #done
        tEvents = self.__getTEvents(series, h=0)#,h=dailyVol.mean()) #done
        t1 = self.__addVerticalBarrier(tEvents, series, numDays=VerticalBarrierDays)
        events = self.__getEvents(series,tEvents,ptSl,target,minRet,cpus,t1=t1)
        labels = self.__getBins(events, series)
        labels.columns = ["target_return", "label"]
        return labels

    def getDailyVol(self,series,span0=100):
        """ daily vol reindexed to series
            Daily Volatility Estimator [3.1]"""
        df0=series.index.searchsorted(series.index-pd.Timedelta(days=1))
        df0=df0[df0>0]
        df0=(pd.Series(series.index[df0-1],
                       index=series.index[series.shape[0]-df0.shape[0]:]))
        try:
            df0=series.loc[df0.index]/series.loc[df0.values].values-1 # daily rets
        except Exception as e:
            print('error: {}\nplease confirm no duplicate indices'.format(e))
        df0=df0.ewm(span=span0).std().rename('dailyVol')
        return df0

    def __getTEvents(self,series,h):
        """Symmetric CUSUM Filter"""
        tEvents, sPos, sNeg = [], 0, 0
        diff = np.log(series).diff().dropna()
        for i in tqdm(diff.index[1:]):
            try:
                pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
            except Exception as e:
                print(e)
                print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
                print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
                break
            sPos, sNeg=max(0., pos), min(0., neg)
            if sNeg<-h:
                sNeg=0;tEvents.append(i)
            elif sPos>h:
                sPos=0;tEvents.append(i)
        return pd.DatetimeIndex(tEvents)

    def __applyPtSlOnT1(self,series,events,ptSl,molecule):
        # apply stop loss/profit taking, if it takes place before t1 (end of event)
        events_=events.loc[molecule]
        out=events_[['t1']].copy(deep=True)
        if ptSl[0]>0: pt=ptSl[0]*events_['trgt']
        else: pt=pd.Series(index=events.index) # NaNs
        if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']
        else: sl=pd.Series(index=events.index) # NaNs
        for loc,t1 in events_['t1'].fillna(series.index[-1]).iteritems():
            df0=series[loc:t1] # path prices
            df0=(df0/series[loc]-1)*events_.at[loc,'side'] # path returns
            out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
            out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
        return out

    def __getEvents(self,series,tEvents,ptSl,trgt,minRet,numThreads,t1=False,side=None):
        # Gettting Time of First Touch (getEvents) [3.3], [3.6]
        #1) get target
        trgt=trgt.loc[tEvents]
        trgt=trgt[trgt>minRet] # minRet
        #2) get t1 (max holding period)
        if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)
        #3) form events object, apply stop loss on t1
        if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
        else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
        events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
                .dropna(subset=['trgt']))
        df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
                        numThreads=numThreads,close=series,events=events,
                        ptSl=ptSl_)
        events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
        if side is None:events=events.drop('side',axis=1)
        return events

    def __addVerticalBarrier(self, tEvents, series, numDays=1):
        # Adding Vertical Barrier [3.4]
        t1=series.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
        t1=t1[t1<series.shape[0]]
        t1=(pd.Series(series.index[t1],index=tEvents[:t1.shape[0]]))
        return t1

    def __getBinsOld(self, events, series):
        # Labeling for side and size [3.5]
        #1) prices aligned with events
        events_=events.dropna(subset=['t1'])
        px=events_.index.union(events_['t1'].values).drop_duplicates()
        px=series.reindex(px,method='bfill')
        #2) create out object
        out=pd.DataFrame(index=events_.index)
        out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
        out['bin']=np.sign(out['ret'])
        # where out index and t1 (vertical barrier) intersect label 0
        try:
            locs = out.query('index in @t1').index
            out.loc[locs, 'bin'] = 0
        except:
            pass
        return out

    # Expanding getBins to Incorporate Meta-Labeling [3.7]
    def __getBins(self, events, series):
        '''
        Compute event's outcome (including side information, if provided).
        events is a DataFrame where:
        -events.index is event's starttime
        -events['t1'] is event's endtime
        -events['trgt'] is event's target
        -events['side'] (optional) implies the algo's position side
        Case 1: ('side' not in events): bin in (-1,1) <-label by price action
        Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
        '''
        #1) prices aligned with events
        events_=events.dropna(subset=['t1'])
        px=events_.index.union(events_['t1'].values).drop_duplicates()
        px=series.reindex(px,method='bfill')
        #2) create out object
        out=pd.DataFrame(index=events_.index)
        out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
        if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
        out['bin']=np.sign(out['ret'])
        if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
        return out



# Linear Partitions [20.4.1]
def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts

#  multiprocessing snippet [20.7]
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)

    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0

#  single-thread execution for debugging [20.8]
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

# Example of async call to multiprocessing lib [20.9]

#________________________________
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(datetime.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return
#________________________________
def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool = multiprocessing.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

# Unwrapping the Callback [20.10]
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

# Pickle Unpickling Objects [20.11]
def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj,cls)
#________________________________
def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)
#________________________________

copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)


if __name__ == '__main__':
    label = Label()
    stockId = "2330"
    startTime = datetime.datetime(2010, 1, 1)
    endTime = datetime.datetime(2019, 8, 7)

    df = label.get_TripleLabeling_df(stockId, startTime, endTime, ptSl=[1,1])
    df["bin"].value_counts()
