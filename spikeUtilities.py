import os
import random
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
import pandas as pd
import mat73
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
import scipy.signal
import re
from multiprocessing import Pool
from itertools import chain

import warnings

def loadBehavMat(Date_temp,Pathway):
    allmatfiles = os.listdir(Pathway)
    behavefilename = [file for file in allmatfiles if Date_temp in file 
                      and '_bahviorInfo' in file][0]
    behavefilePathway = os.path.join(Pathway,behavefilename)
    behavdata = mat73.loadmat(behavefilePathway)
    eventsTimDF = pd.DataFrame(behavdata['taskTime_shift']) # time (ms) of events in each trial relative to the soundOn
    eventsTimDF.rename(columns={'TestStimOn':'TestStimOnTDT'},inplace=True)
    eventsTimDF['LED_On'] = eventsTimDF.apply(lambda row: [e for e in row[['Bld_On', 'Gld_On', 'Rld_On']] if pd.notnull(e)][0] 
                                           if not row[['Bld_On', 'Gld_On', 'Rld_On']].isnull().all() else np.nan, axis=1)#combine all led lights into 1 column
    eventsTimDF['soundOnFlag'] = np.where(eventsTimDF['soundOn'].isna(),0,1)
    eventsTimDF['LEDFlag'] = np.where(eventsTimDF['LED_On'].isna(),0,1)
    labelDF = pd.DataFrame(behavdata['taskVar'])
    labelDF.rename(columns={'TestStimOn':'TestStimOnLabv'},inplace=True)
    labelDF['isHigh']=labelDF['isHigh'].replace({0:'Low',1:'High'})
    #flattern list to string for this columns
    labelDF['prior'] = labelDF['prior'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
    labelDF['choiceName'] = labelDF['choiceName'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
    labelDF['pretone'] = labelDF['pretone'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
    # convert tonelevel to positive int
    labelDF['toneLevel'] = labelDF['toneLevel'].abs().astype(int)
    labelDF['prior']=labelDF['prior'].replace({'H':'High','L':'Low','N':'Neutral','X':'XNone'})
    labeltimcombDF = pd.concat([eventsTimDF,labelDF],axis=1)
    columnsUseful = ['monkeyID', 'Date','trialID','LED_On','LEDFlag','bLED','gLED','rLED','TestStimOnTDT','Stm_On','Stm_Off','Trl_On','Trl_Off',
                     'soundOn','targOn','soundOnFlag',
                     'Rwd_On','rewardTTL','RewardDur','JoystickReach','JoystickStartMove','RT',
                       'cumCorrect', 'cumError','cumWrong',
                     'choiceName', 'isHigh','pretoneLength','pretone','prior','toneLevel','correct','wrong','freqHi','freqLo']
    return labeltimcombDF[columnsUseful]

def loadPreprocessMat(Date_temp,Pathway,area='AC'):
    allmatfiles = os.listdir(Pathway)
    spikefilename = [file for file in allmatfiles if Date_temp in file 
                      and '_'+area+'_' in file 
                      and '_preprocSpiketimeseries' in file][0]
    spikefilePathway = os.path.join(Pathway,spikefilename)
    spikedata = mat73.loadmat(spikefilePathway)
    spikeTimeDict = spikedata['Spikepreproc'] 
    spikefs = spikeTimeDict['fs']
    timeSamp2stimOn = np.arange(int(spikeTimeDict['timeRange'][0]*spikefs),int(spikeTimeDict['timeRange'][1]*spikefs),1)
    return spikeTimeDict, timeSamp2stimOn,spikefs

def convertspikeTime(spikeTime_tt,spktimshift,spikefs,timeRange2save):
    #convert spikeTime_tt(samples relative to stim onset) 
    # to time (miliseconds relative to target onset)
    if len(spikeTime_tt[0].shape)==0 :
        spikeTime_tt = [spikeTime_tt.copy()]  
    spikTime_array_temp = 1000*(spikeTime_tt[0]+spktimshift)/spikefs

    # get the data within the time window of interest 
    indwin = np.where((spikTime_array_temp>=timeRange2save[0])&(spikTime_array_temp<=timeRange2save[1]))       
    spikTime_list_temp = spikTime_array_temp[indwin].copy()
    ## trials without spikes will have one row of 100000 spiketime in the dataframe
    if len(spikTime_list_temp)>=1:           
        spikTime_df_temp =pd.DataFrame({'spktim':spikTime_list_temp})
    else:
        spikTime_df_temp =pd.DataFrame({'spktim':[100000]}) 
    return spikTime_df_temp
    
def getClsSPK(spikeTime_tt,rowseries,eventsIndref_tt,eventsIndtarg_tt,timeRange2save,spikefs):
    # spikeTime_tt in samples, eventsIndref_tt & eventsIndtarg_tt in ms, 
    rowdf = rowseries.to_frame().transpose()
        
    # for trials with valid target and reference events 
    if not np.isnan(eventsIndref_tt) and not np.isnan(eventsIndtarg_tt):
        # shift time (relative to stim onset) to the targetevent of each trial
        spktimshift = -spikefs*(eventsIndtarg_tt-eventsIndref_tt)/1000 #in the unit of samples
        spikTime_df_temp = convertspikeTime(spikeTime_tt,spktimshift,spikefs,timeRange2save)          
    # for all other trials will have one row of nan spiketime       
    else:
        spikTime_df_temp = pd.DataFrame({'spktim':[100000]})

    ## add labels info to spiketim
    for cc in rowdf.columns:
        val_temp = rowdf[cc].values[0]
        spks = spikTime_df_temp.shape[0]
        if isinstance(val_temp,float) or isinstance(val_temp,int) or isinstance(val_temp,str):            
            colval_temp = [val_temp]*spks
        elif len(val_temp)==1:
            colval_temp = val_temp*spks
        spikTime_df_temp[cc]=colval_temp
    return spikTime_df_temp
    
def getClsRasterMultiprocess(spikeTime,spikefs,eventsRef,eventsTarg,timeRange2save,label_df,cpus=2):
# get raster dataframe for this cluster, temporally cut according to timeTange2save 
# spikeTime originally align to the stim onset
# will temporally align the raster relative to the eventsIndtarg  
    spikTime_df = pd.DataFrame()

    # for (spikeTime_tt,(_,rowseries),eventsIndref_tt,eventsIndtarg_tt) in zip(spikeTime,label_df.iterrows(),label_df[eventsRef],label_df[eventsTarg]):
    #     spikTime_df_temp = getClsSPK(spikeTime_tt,rowseries,eventsIndref_tt,eventsIndtarg_tt,timeRange2save,spikefs)
    #     spikTime_df=pd.concat([spikTime_df,spikTime_df_temp])

    argItems = [(spikeTime_tt,rowseries,eventsIndref_tt,eventsIndtarg_tt,timeRange2save,spikefs)\
                for (spikeTime_tt,(_,rowseries),eventsIndref_tt,eventsIndtarg_tt) in \
                zip(spikeTime,label_df.iterrows(),label_df[eventsRef],label_df[eventsTarg])]
    with Pool(processes=cpus) as p:
        # spikeTime is a list: trails X spikeNum
        for spikTime_df_temp in p.starmap(getClsSPK,argItems):
            # print(spikTime_df_temp)
            spikTime_df=pd.concat([spikTime_df,spikTime_df_temp])
            # print(str(spikTime_df.shape[0])+'trials done') 
    p.close()
    p.join()

    return spikTime_df

def SortGroupDF(neuraldata_df_list,labelDict,filterlable = {'respLabel':0},condlabel=['trialMod','snr']):
    #neuraldata_df_list are list of dataframes, eahc dataframe is trials X feature
    neuraldata_df_grouped_list = []
    for neuraldata_df in neuraldata_df_list:
        # add condition columns to combined dataframe
        for cc in condlabel:
            neuraldata_df[cc] = labelDict[cc]
    # apply filter condition
        ind = np.arange(0,neuraldata_df.shape[0])
        for ii,(kk,vv) in enumerate(filterlable.items()):
            ind_temp = np.where(labelDict[kk]==vv)[0]
            ind = np.intersect1d(ind,ind_temp)
        neuraldata_df_filetered = neuraldata_df.iloc[ind,:]
    # # neuraldata_df_filetered shouldn't has any nan in datacolumns after conditional filtering
        colsubset = list(neuraldata_df_filetered.columns)[0:-len(condlabel)]
        mask = neuraldata_df_filetered[colsubset].isna().any(axis=1)
        NaNrow_index = mask.index[mask].tolist()
        if len(NaNrow_index)>=1:
            print('WARNING: nan still exist in data after applying conditional filtering')
        # neuraldata_df_filetered_clean = neuraldata_df_filetered.dropna(subset=colsubset )
    # #group by modalities    
        neuraldata_df_grouped_list.append(neuraldata_df_filetered.groupby(condlabel[0]))

    return neuraldata_df_grouped_list


def SortfilterDF(neuraldata_df,filterlable = {'respLabel':[0]}):
    #neuraldata_df is dataframe: trials X feature
    # apply filter condition
    ind = np.arange(0,neuraldata_df.shape[0])
    for ii,(kk,vv) in enumerate(filterlable.items()):        
        ind_temp = np.where(np.isin(neuraldata_df[kk],vv))[0]
        ind = np.intersect1d(ind,ind_temp)
    neuraldata_df_filetered = neuraldata_df.iloc[ind,:]
    return neuraldata_df_filetered,ind


def getPSTH(spikeTimelist,timpnts,bintimeLen,totaltrials=1,totaltrialsYaxis=1, scaleFlag='on', kernelFlag='on',triInfo=None):
    def gaussian_kernel(size, sigma):
        size = int(size) // 2
        x = np.arange(-size, size+1)
        kernel = np.exp(-(x**2) / (2*sigma**2))
        return kernel / sum(kernel)

    psth = []
    for tt in timpnts:
        psth.append(sum(1 for x in spikeTimelist if tt <= x < tt+bintimeLen)/totaltrials)
    # scale value for visual effect
    if scaleFlag=='on':    
        psth = psth.copy()
     
    # smooth psth
    if kernelFlag=='on':
        kernel = gaussian_kernel(50, 25)
        psth = scipy.signal.convolve(psth, kernel, mode='same')
    #set output
    if triInfo is not None:
        triaOutput = triInfo
    else:
        triaOutput = None
    return psth,triaOutput

def getPSTHTbyT(spikeTimedf_temp_filtered,timpnts, bintimeLen,cpus=2):
    #get num of spks in each slide window of each trial + trialinfo, saved in a dataframe
    PSTHTbyT = pd.DataFrame()
    timpntsColname = ['*'+str(tt)+'*' for tt in timpnts]
    argItems = []

    for tri in list(spikeTimedf_temp_filtered.trialID.unique()):
        spktim_temp = spikeTimedf_temp_filtered[spikeTimedf_temp_filtered['trialID']==tri].reset_index(drop=True)
        param_df_temp = spktim_temp.iloc[[0]].drop('spktim',axis=1,inplace=False)
        argItems = argItems+[(spktim_temp.spktim.values,timpnts,bintimeLen,1,1,'off','off',param_df_temp)]
        # psth_temp,triaOutput = getPSTH(spktim_temp.spktim.values,timpnts,bintimeLen,1,1,'off','off',param_df_temp)
        # PSTHTbyT_temp = pd.DataFrame([psth_temp],columns=timpntsColname)
        # PSTHTbyT=pd.concat([PSTHTbyT,pd.concat([PSTHTbyT_temp,param_df_temp],axis=1)],axis=0)

    with Pool(processes=cpus) as p:
        for psth_temp,triInfo in p.starmap(getPSTH,argItems):
            PSTHTbyT_temp = pd.DataFrame([psth_temp],columns=timpntsColname)
            PSTHTbyT=pd.concat([PSTHTbyT,pd.concat([PSTHTbyT_temp,triInfo],axis=1)],axis=0)
    p.close()
    p.join()
        
    return PSTHTbyT

def glmfit(spikenum_df_raw,DVcol,IVcolList,familystr,seedt):
    # balance samples within groups, main effect group need to be strictly balanced
    spikenum_df,_ = sampBalanceGLM(spikenum_df_raw.reset_index(drop=True),IVcolList,seeds=seedt,method='upsample')

    # add intercet in linear regression formula
    spikenum_df['const'] = np.ones((spikenum_df.shape[0],1))
    # #dtype object to float
    # for col in IVcolList:
    #     # if spikenum_df[col].dtype=='object':
    #     spikenum_df[col] = pd.factorize(spikenum_df[col],sort=True)[0]
    
    try:   
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # Convert RuntimeWarnings to exceptions

            # Instantiate a poisson family model with the default (log) link function.
            if familystr=='poisson':
                # print(spikenum_df[[DVcol]+IVcolList].to_string())
                glm_model = sm.GLM(spikenum_df[DVcol], spikenum_df[['const']+IVcolList] , family=sm.families.Poisson(),missing='drop')
            if familystr == 'gaussian':
                glm_model = sm.GLM(spikenum_df[DVcol], spikenum_df[['const']+IVcolList] , family=sm.families.Gaussian(link=sm.families.links.Identity()),missing='drop')    
            glm_results = glm_model.fit(max_iter=1000, tol=1e-6, tol_criterion='params')
            # print(glm_results.summary())
            coef = pd.DataFrame(glm_results.params).transpose()
            colnames = ['coef_'+col for col in coef.columns]
            coef = coef.rename(columns=dict(zip(coef.columns,colnames)))
            pval = pd.DataFrame(glm_results.pvalues).transpose()
            colnames = ['pval_'+col for col in pval.columns]
            pval = pval.rename(columns=dict(zip(pval.columns,colnames))) 
            evalparam = pd.DataFrame({'aic':glm_results.aic,'bic':glm_results.bic_llf},index=[0])
            # evalparam = pd.DataFrame({'aic':[0],'bic':[0]},index=[0])
            return coef, pval, evalparam,DVcol
    except (RuntimeWarning, ValueError) as e:
        # print(f"An error occurred: {e}")
        # print(spikenum_df[DVcol].values)
        return [DVcol]

def getglmfitovertime(psth_win,familystr,GLM_IV_list,cpus,Monkey,Date_temp,keys,area,fitdfnamestr):
    glmfitres_all = pd.DataFrame()
    IVnum = len(GLM_IV_list)
    timcol = [col for col in list(psth_win.columns) if "*" in col]
    othercol = list(set(list(psth_win.columns))-set(timcol))    
    
    # for (timcol_temp,seedt) in (zip(timcol,np.random.choice(range(10000), size=len(timcol), replace=False))):
    #     fitres = glmfit(psth_win[[timcol_temp]+othercol],timcol_temp,GLM_IV_list,familystr,seedt)
    #     if len(fitres)==1:
    #         print('fail to fit glm model in'+Monkey+Date_temp+' unit:'+keys)
    #         coeff_temp = pd.DataFrame([[np.nan]*IVnum],columns=['coef_'+iv for iv in GLM_IV_list])
    #         pval_temp = pd.DataFrame([[np.nan]*IVnum],columns=['pval_'+iv for iv in GLM_IV_list])
    #         evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])
    #     else:
            # coeff_temp = fitres[0]
            # pval_temp = fitres[1]
            # evalparam = fitres[2]
            # timcol_temp = fitres[3]
    #         glmfitres_temp = pd.DataFrame.from_dict({'Monkey':[Monkey]*IVnum,'session_cls':[Date_temp[-6:]+'_'+keys]*IVnum,\
    #                                                 'time':[extract_between_tts(timcol_temp)]*IVnum,'iv':GLM_IV_list,\
    #                                                 'slope':[coeff_temp['coef_'+iv].values[0] for iv in GLM_IV_list],\
    #                                                 'pval':[pval_temp['pval_'+iv].values[0] for iv in GLM_IV_list],\
    #                                                 'aic':[evalparam.aic.values[0]]*IVnum,'bic':[evalparam.bic.values[0]]*IVnum,\
    #                                                 'GLMmod':[str(GLM_IV_list)]*IVnum})                        
    #     glmfitres_all = pd.concat([glmfitres_all,glmfitres_temp])
    
    
    argItems = [(psth_win[[timcol_temp]+othercol],timcol_temp,GLM_IV_list,familystr,seedt) for (timcol_temp,seedt) in (zip(timcol,np.random.choice(range(10000), size=len(timcol), replace=False)))]
    with Pool(processes=cpus) as p:
        for fitres in p.starmap(glmfit,argItems):       
            if len(fitres)==1:
                timcol_temp = fitres[0]
                print('FAIL to fit glm model in '+Monkey+Date_temp+' unit:'+keys +' at '+timcol_temp)
                coeff_temp = pd.DataFrame([[np.nan]*IVnum],columns=['coef_'+iv for iv in GLM_IV_list])
                pval_temp = pd.DataFrame([[np.nan]*IVnum],columns=['pval_'+iv for iv in GLM_IV_list])
                evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])               
            else:
                coeff_temp = fitres[0]
                pval_temp = fitres[1]
                evalparam = fitres[2]
                timcol_temp = fitres[3]                  
                # print('suceed at glmfit in '+Monkey+Date_temp+' unit:'+keys +' at '+timcol_temp)

            glmfitres_temp = pd.DataFrame.from_dict({'Monkey':[Monkey]*IVnum,'session_cls':[Date_temp[-6:]+'_'+keys]*IVnum,'Region':[area]*IVnum,'session_cls_Region':[Date_temp[-6:]+'_'+keys+'_'+area]*IVnum,\
                                                    'time':[np.float64(extract_between_tts(timcol_temp))]*IVnum,'iv':GLM_IV_list,\
                                                    'slope':[coeff_temp['coef_'+iv].values[0] for iv in GLM_IV_list],\
                                                    'pval':[pval_temp['pval_'+iv].values[0] for iv in GLM_IV_list],\
                                                    'aic':[evalparam.aic.values[0]]*IVnum,'bic':[evalparam.bic.values[0]]*IVnum,\
                                                    'GLMmod':[str(GLM_IV_list)]*IVnum,'fitdfnamestr':[fitdfnamestr]*IVnum})  
            # print(glmfitres_temp.to_string())                      
            glmfitres_all = pd.concat([glmfitres_all,glmfitres_temp])
    p.close()
    p.join()

    return glmfitres_all

def sampBalanceGLM(spikeNumdf_temp_raw,GLM_IV_list,seeds=42,method='upsample'):
    #spikeNumdf_temp_raw need to have sequentially unique row index
    spikeNumdf_temp_raw = spikeNumdf_temp_raw.reset_index(drop=True)
    indexpick = []
    grouped = spikeNumdf_temp_raw.groupby(by=GLM_IV_list)
    group_indices = {key: grouped.groups[key].tolist() for key in grouped.groups}
    # upsample group trials by adding randomsample with replacement to the original trials
    if method=='upsample':
        maxtrials = max(len(lst) for lst in group_indices.values())   
        spikeNumdf_temp = pd.DataFrame()
        for gg, (gkey,glist) in enumerate(group_indices.items()):
            rng = np.random.default_rng(seeds+gg*10)
            addsampleInd = rng.choice(glist,size=maxtrials-len(glist),replace=True)
            # print(addsampleInd)
            spikeNumdf_temp_sub = pd.concat((spikeNumdf_temp_raw.iloc[glist,:],spikeNumdf_temp_raw.iloc[addsampleInd,:]))
            spikeNumdf_temp = pd.concat((spikeNumdf_temp,spikeNumdf_temp_sub))
            indexpick = indexpick+glist+list(addsampleInd)
    elif method=='downsample':
        maxtrials = min(len(lst) for lst in group_indices.values())
        spikeNumdf_temp = pd.DataFrame()
        for gg, (gkey,glist) in enumerate(group_indices.items()):
            rng = np.random.default_rng(seeds+gg*10)
            addsampleInd = rng.choice(glist,size=maxtrials,replace=True)
            spikeNumdf_temp = pd.concat((spikeNumdf_temp,spikeNumdf_temp_raw.iloc[addsampleInd,:])) 
            indexpick = indexpick+list(addsampleInd)        
    return spikeNumdf_temp.reset_index(drop=True),indexpick

def countTrialSPKs(spikeTimedf,estwin='off',winTim=[0,300],conswin = 600):
    spikenum_df = pd.DataFrame()

    for tt in spikeTimedf['trialID'].unique(): 
        group = spikeTimedf[spikeTimedf['trialID']==tt]
        spikenum_df_temp = pd.DataFrame(group.iloc[0,:].copy()).transpose()
        if estwin=='off':# num of spikes in this trial
            spikenum_df_temp['spknum'] = group[group['spktim']<100000].shape[0] 
        elif estwin=='BlNSig':
            spikenum_df_temp['spkRate_baseline'] = group[(group['spktim']>=winTim[0]) & (group['spktim']<=0)].shape[0]/np.abs(winTim[0]/1000)
            spikenum_df_temp['spkRate_sig'] = group[(group['spktim']>0) & (group['spktim']<=winTim[1])].shape[0]/np.abs(winTim[1]/1000)
        elif estwin=='subwinoff':
            spikenum_df_temp['spknum'] = group[(group['spktim']>=winTim[0]) &(group['spktim']<winTim[1])].shape[0]

        spikenum_df_temp.drop('spktim',axis=1,inplace=True)
        spikenum_df = pd.concat([spikenum_df,spikenum_df_temp])
    return spikenum_df.reset_index(drop=True)


def sampBalanCond(df,cond):
    df_group_counts = df.groupby(cond)[cond].size().reset_index(name='count')
    sampNum = df_group_counts['count'].values.min()
    selected_rows = df.groupby(cond).apply(lambda x: x.sample(sampNum)).reset_index(level=list(range(len(cond))),drop=True).index.tolist()
    return selected_rows

def addIVcol(AllSUspk_df_raw):
    def prior_pretone_targ(row,col1,col2,col3):
        # rank the interaction effect as numbers
        if col3 == None:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2]]):
                if row[col1][0] == row[col2][0]:
                    return 1 #[prior/pretone, isHigh]:HH/LL 
                elif row[col1][0] == 'L' and row[col2][0] == 'H':
                    return -1 #[prior/pretone, isHigh]:HL/LH 
                elif row[col1][0] == 'H' and row[col2][0] == 'L':
                    return -1
                elif row[col1][0] == 'N' or row[col2][0] == 'N':
                    return 0  #[prior/pretone, isHigh]NH/NL     
                else:
                    return np.nan
            else:
                return np.nan            
        else:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2],row[col3]]):
                PPparaList = [row[col1][0],row[col2][0]] 
                if PPparaList.count(row[col3][0])==2:
                    return 3 #[prior,pretone,isHigh]:[HHH/LLL] 
                elif PPparaList.count(row[col3][0])==1:
                    if [vv=='N' for vv in PPparaList].count(True)==1:
                        return 2 #[prior,pretone,isHigh]:[NHH,NLL,HNH,LNL] 
                    else:
                        return 1 #[prior,pretone,isHigh]:[HL(L/H),LH(L/H)] 
                elif PPparaList.count(row[col3][0])==0:
                    if [vv=='N' for vv in PPparaList].count(True)==2:
                        return 0 #[prior,pretone,isHigh]:[NNH,NNL]  
                    elif [vv=='N' for vv in PPparaList].count(True)==1:
                        return -1 #[prior,pretone,isHigh]:[NHL,NLH,HNL,LNH] 
                    elif [vv=='N' for vv in PPparaList].count(True)==0:
                        return -2 #[prior,pretone,isHigh]:HHL/LLH
            else:
                return np.nan
    
    AllSUspk_df = AllSUspk_df_raw.copy()                    
    AllSUspk_df.loc[:,'prior'] = AllSUspk_df_raw['prior'].replace({'None':'XNone'})
    #prior:[L,H,N,X],  pretone:[High, Low, None], isHigh:[High,Low],  Choice:[High,Low,noChoice]     
    AllSUspk_df['prior-Targ'] = AllSUspk_df.apply(prior_pretone_targ,col1='prior', col2='isHigh',col3=None,axis=1)#HH/LL:1,HL/LH:-1,NH/NL:0
    AllSUspk_df['pretone-Targ'] = AllSUspk_df.apply(prior_pretone_targ,col1='pretone', col2='isHigh',col3=None,axis=1)#HH/LL:1,HL/LH:-1,NH/NL:0
    AllSUspk_df['prior-pretone-Targ'] = AllSUspk_df.apply(prior_pretone_targ,col1='prior',col2='pretone', col3='isHigh',axis=1)

    AllSUspk_df['prior-Choice'] = AllSUspk_df.apply(prior_pretone_targ,col1='prior', col2='choiceName',col3=None,axis=1) 
    AllSUspk_df['pretone-Choice'] = AllSUspk_df.apply(prior_pretone_targ,col1='pretone', col2='choiceName',col3=None,axis=1) 
    #[HHH/LLL]:3,[NHH,NLL,HNH,LNL]:2,[HL(L/H),LH(L/H)]:1,[NNH,NNL]:0,[NHL,NLH,HNL,LNH]:-1,HHL/LLH:-2
    AllSUspk_df['prior-pretone-Choice'] = AllSUspk_df.apply(prior_pretone_targ,col1='prior',col2='pretone', col3='choiceName',axis=1)
    #keep the string name also
    #rank string factor to number for interpretable glm slope of these factors
    AllSUspk_df.loc[:,'prior'] = AllSUspk_df_raw['prior'].replace({'High':1,'Low':-1,'Neutral':0,'XNone':np.nan})
    AllSUspk_df.loc[:,'pretone'] = AllSUspk_df_raw['pretone'].replace({'High':1,'Low':-1,'None':0})
    AllSUspk_df.loc[:,'isHigh'] = AllSUspk_df_raw['isHigh'].replace({'High':1,'Low':-1})
    AllSUspk_df.loc[:,'choiceName'] = AllSUspk_df_raw['choiceName'].replace({'High':1,'Low':-1,'noChoice':np.nan})
    AllSUspk_df.loc[:,'correct'] = AllSUspk_df_raw['correct'].astype(int)
    AllSUspk_df['session_cls_Region'] = AllSUspk_df['sess']+'_'+AllSUspk_df['cls']+'_'+AllSUspk_df['Region']
    return AllSUspk_df

def addIVcol2(AllSUspk_df_raw,FreqPref):
    def prior_pretone_targ_pref2(row,col1,col2,col3,FreqPref):
        # rank the interaction effect as numbers
        if col3 == None:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2]]) \
                and sorted(np.unique([row[col1][0],row[col2][0],FreqPref[0]])).isin(['H','L','N']):
                if row[col1][0]=='N':
                    if FreqPref[0]==row[col2][0]:
                        return 2 # [neuron,prior/pretone,isHigh]:HNH,LNL
                    if FreqPref[0]!=row[col2][0]:
                        return -2 # [neuron,prior/pretone,isHigh]:HNL,LNH
                elif row[col1][0]!='N':                
                    if [vv==FreqPref[0] for vv in [row[col1][0],row[col2][0]]].count(True)==2:
                        return 3 # [neuron,prior/pretone,isHigh]: HHH,LLL
                    elif [vv==FreqPref[0] for vv in [row[col1][0],row[col2][0]]].count(True)==0:
                        return -3 # [neuron,prior/pretone,isHigh]: HLL,LHH             
                    elif FreqPref[0]==row[col2][0] and FreqPref[0]!=row[col1][0]:
                        return 1 # [neuron,prior/pretone,isHigh]: HLH,LHL
                    elif FreqPref[0]!=row[col2][0] and FreqPref[0]==row[col1][0]:
                        return -1 # [neuron,prior/pretone,isHigh]: HHL,LLH                         
            else:
                return np.nan            
        else:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2],row[col3]])\
                and sorted(np.unique([row[col1][0],row[col2][0],row[col3][0],FreqPref[0]])).isin(['H','L','N']):
                PPparaList = [row[col1][0],row[col2][0]] 
                if row[col1][0] == row[col2][0] == row[col3][0] == FreqPref[0]:
                    return 3
                elif row[col3][0] == FreqPref[0] and [vv==FreqPref[0] for vv in PPparaList].count(True)==1:
                    return 2
                elif row[col3][0] == FreqPref[0] and [vv==FreqPref[0] for vv in PPparaList].count(True)==0:
                    return 1
                elif row[col3][0] != FreqPref[0] and [vv==FreqPref[0] for vv in PPparaList].count(True)==2:
                    return -1
                elif row[col3][0] != FreqPref[0] and [vv==FreqPref[0] for vv in PPparaList].count(True)==1:
                    return -2
                elif row[col3][0] != FreqPref[0] and [vv==FreqPref[0] for vv in PPparaList].count(True)==0:
                    return -3
            else:
                return np.nan

    def prior_pretone_targ_pref3(row,col1,col2,col3,FreqPref):
        # rank the interaction effect as numbers
        if col3 == None:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2]]):
                if row[col2][0]==FreqPref[0]:# only factorize when isHigh(col2) is neuron's preferred frequency
                    if row[col2][0]==row[col1][0]:
                        return 3 # [neuron,prior/pretone,isHigh]:HHH,LLL
                    elif row[col1][0]=='N':
                        return 2 # [neuron,prior/pretone,isHigh]:HNH,LNL
                    else:
                        return 1 # [neuron,prior/pretone,isHigh]:HLH,LHL
                else:
                    return 0                      
            else:
                return np.nan            
        else:
            if not any(s in ['noChoice','XNone'] for s in [row[col1],row[col2],row[col3]]):
                PPparaList = [row[col1][0],row[col2][0]] 
                if row[col3][0]==FreqPref[0]:# only factorize when isHigh(col3) is neuron's preferred frequency
                    if row[col1][0]=='N':
                        if row[col2][0] == row[col3][0]:
                            return 4 # [neuron,prior,pretone,isHigh]:HNHH,LNLL
                        elif row[col2][0] != row[col3][0]:
                            return 2 # [neuron,prior,pretone,isHigh]:HNLH,LNHL
                    else:
                        if [vv==FreqPref[0] for vv in PPparaList].count(True)==2:
                            return 5 # [neuron,prior,pretone,isHigh]:HHHH,LLLL
                        elif [vv==FreqPref[0] for vv in PPparaList].count(True)==1:
                            return 3 # [neuron,prior,pretone,isHigh]:HLHH,HHLH,LHLL,LLHL
                        elif [vv==FreqPref[0] for vv in PPparaList].count(True)==0:
                            return 1 # [neuron,prior,pretone,isHigh]:HLLH,LHHL
                else:
                    return 0               
            else:
                return np.nan

    AllSUspk_df = AllSUspk_df_raw.copy()                    
    AllSUspk_df.loc[:,'prior'] = AllSUspk_df_raw['prior'].replace({'None':'XNone'})
    #prior:[L,H,N,X],  pretone:[High, Low, None], isHigh:[High,Low],  Choice:[High,Low,noChoice]     
    AllSUspk_df['prior-Targ'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='prior', col2='isHigh',col3=None,FreqPref=FreqPref,axis=1)
    AllSUspk_df['pretone-Targ'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='pretone', col2='isHigh',col3=None,FreqPref=FreqPref,axis=1)
    AllSUspk_df['prior-pretone-Targ'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='prior',col2='pretone', col3='isHigh',FreqPref=FreqPref,axis=1)

    AllSUspk_df['prior-Choice'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='prior', col2='choiceName',col3=None,FreqPref=FreqPref,axis=1) 
    AllSUspk_df['pretone-Choice'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='pretone', col2='choiceName',col3=None,FreqPref=FreqPref,axis=1) 
    AllSUspk_df['prior-pretone-Choice'] = AllSUspk_df.apply(prior_pretone_targ_pref3,col1='prior',col2='pretone', col3='choiceName',FreqPref=FreqPref,axis=1)
    #rank string factor to number for interpretable glm slope of these factors
    AllSUspk_df.loc[:,'prior'] = AllSUspk_df_raw['prior'].replace({'High':1,'Low':-1,'Neutral':0,'XNone':np.nan})
    AllSUspk_df.loc[:,'pretone'] = AllSUspk_df_raw['pretone'].replace({'High':1,'Low':-1,'None':0})
    AllSUspk_df.loc[:,'isHigh'] = AllSUspk_df_raw['isHigh'].replace({'High':1,'Low':-1})
    AllSUspk_df.loc[:,'choiceName'] = AllSUspk_df_raw['choiceName'].replace({'High':1,'Low':-1,'noChoice':np.nan})
    return AllSUspk_df

def getcorrcoef(spk1,spk2,rdmseed):
    rng = np.random.default_rng(rdmseed)
    NC = np.ma.corrcoef(rng.choice(spk1,size=len(spk1),replace=False),spk2)[0][1]
    return NC 

def estNoiseCorr(spkdf1,spkdf2,unitpairs,spkCol,cond=['trialMod','snr'],cpus=20):
    # Suppress specific warnings from NumPy about masked elements
    warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")
    NCdf = pd.DataFrame()
    spkdf1_group = spkdf1.groupby(cond)
    spkdf2_group = spkdf2.groupby(cond)
    grou_keys = spkdf1_group.groups.keys()
    for key in grou_keys:
        group1 = spkdf1_group.get_group(key).sort_values(['trialID'],kind='mergesort')
        group2 = spkdf2_group.get_group(key).sort_values(['trialID'],kind='mergesort')
        if list(group1['trialID'].values) == list(group2['trialID'].values):
            for tim,col in enumerate(spkCol):
                NCdfz_temp = pd.DataFrame()
                spk1_temp_all = group1[col].values
                spk2_temp_all = group2[col].values
                # handle nans in noise fr 
                spk1_temp = np.ma.masked_array(spk1_temp_all, mask=np.isnan(spk1_temp_all))
                spk2_temp = np.ma.masked_array(spk2_temp_all, mask=np.isnan(spk2_temp_all))

                for cc in cond:
                    NCdfz_temp[cc]= [group1[cc].values[0]]
                NC = np.ma.corrcoef(spk1_temp,spk2_temp)[0][1]

                NCdfz_temp['time'] = col
                NCdfz_temp['corrcoef']=NC                
                ## get shuffles nc for 95 percentile
                NC_shuffle = []    
                bst = 200   
                for bts in range(bst):
                    NC_shuffle.append(np.ma.corrcoef(random.sample(list(spk1_temp),len(spk1_temp)),\
                                                  list(spk2_temp))[0][1])                            
                # argItems = [(list(spk1_temp),list(spk2_temp),rdmseed) for rdmseed in np.random.choice(range(10000), size=bst, replace=False)]
                # with Pool(processes=cpus) as p:
                #     for NC_shuffle_temp in p.starmap(getcorrcoef,argItems):
                #         NC_shuffle.append(NC_shuffle_temp)                
                # print(NC_shuffle)

                NCdfz_temp['95percentile'] = np.percentile(NC_shuffle,95,axis=0) 
                NCdfz_temp['5percentile'] = np.percentile(NC_shuffle,5,axis=0) 
                if NC>NCdfz_temp['95percentile'].values or NC<NCdfz_temp['5percentile'].values:
                    NCdfz_temp['sig'] = 1
                else:
                    NCdfz_temp['sig'] = -1
                NCdfz_temp['NeuPairs'] = unitpairs[0]+'&'+unitpairs[1]
                NCdf = pd.concat((NCdf,NCdfz_temp))   
    return NCdf

# def estNoiseCorr(spkdf1,spkdf2,unitpairs,spkCol,cond=['trialMod','snr']):
#     NCdf = pd.DataFrame()
#     spkdf1_group = spkdf1.groupby(cond)
#     spkdf2_group = spkdf2.groupby(cond)
#     grou_keys = spkdf1_group.groups.keys()
#     for key in grou_keys:
#         group1 = spkdf1_group.get_group(key).sort_values(['trialNum'],kind='mergesort')
#         group2 = spkdf2_group.get_group(key).sort_values(['trialNum'],kind='mergesort')
#         if list(group1['trialNum'].values) == list(group2['trialNum'].values):
#             for tim,col in enumerate(spkCol):
#                 NCdfz_temp = pd.DataFrame()
#                 spk1_temp_all = group1[col].values
#                 spk2_temp_all = group2[col].values
#                 # handle nans in noise fr 
#                 spk1_temp = np.ma.masked_array(spk1_temp_all, mask=np.isnan(spk1_temp_all))
#                 spk2_temp = np.ma.masked_array(spk2_temp_all, mask=np.isnan(spk2_temp_all))

#                 for cc in cond:
#                     NCdfz_temp[cc]= [group1[cc].values[0]]
#                 NC = np.ma.corrcoef(spk1_temp,spk2_temp)[0][1]
#                 NCdfz_temp['time'] = col
#                 NCdfz_temp['corrcoef']=NC                
#                 ## get shuffles nc for 95 percentile
#                 NC_shuffle = []
#                 for bts in range(500):
#                     NC_shuffle.append(np.ma.corrcoef(random.sample(list(spk1_temp),len(spk1_temp)),\
#                                                   list(spk2_temp))[0][1])
#                 NCdfz_temp['95percentile'] = np.percentile(NC_shuffle,95,axis=0) 
#                 NCdfz_temp['5percentile'] = np.percentile(NC_shuffle,5,axis=0) 
#                 if NC>NCdfz_temp['95percentile'].values or NC<NCdfz_temp['5percentile'].values:
#                     NCdfz_temp['sig'] = 1
#                 else:
#                     NCdfz_temp['sig'] = -1
#                 NCdfz_temp['NeuPairs'] = unitpairs[0]+'&'+unitpairs[1]
#                 NCdf = pd.concat((NCdf,NCdfz_temp))   
#     return NCdf

def extract_between_tts(s):
    match = re.search(r'\*(.*?)\*', s) # extract content bewteen * in a string
    return match.group(1) if match else None

def getdPrime(fitacc_nNueron_temp,trialLab,printflag,snrsep):
    def dprimeCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if hitRate==0:
            hitRate = 0.0001 
        if FARate==1:
            FARate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        dPrime = norm.ppf(hitRate)-norm.ppf(FARate)
        return dPrime 
    
    def ProbCorctCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if hitRate==0:
            hitRate = 0.0001 
        if FARate==1:
            FARate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        probCorct = norm.cdf((norm.ppf(hitRate)-norm.ppf(FARate))/2)
        return probCorct 
    
    behavdf = pd.DataFrame()
    ylabelStrlist = [extract_between_tts(ii) for ii in trialLab]
    sig_index = [index for index,value in enumerate(ylabelStrlist) if 'sig' in value]
    noise_index = [index for index,value in enumerate(ylabelStrlist) if 'noise' in value]
    # a trials then av trials
    for mod in [['a_'],['av_','v_']]:
        behavdf_temp = pd.DataFrame()
        mod_index = [index for index,value in enumerate(ylabelStrlist) if any(s in value for s in mod)]
        sig_index_mod = list(set(sig_index)&set(mod_index))
        noise_index_mod= list(set(noise_index)&set(mod_index))

        fitacc_nNueron_sig = fitacc_nNueron_temp[sig_index_mod]
        fitacc_nNueron_noise = fitacc_nNueron_temp[noise_index_mod]
        ylabelSig = [ylabelStrlist[kk] for kk in sig_index_mod]

        if printflag==1:
            print('mod:'+mod[0]+' signan: '+str(len(np.where(np.isnan(fitacc_nNueron_sig))[0]))+'/'+str(len(fitacc_nNueron_sig))
                  +' noisenan:'+str(len(np.where(np.isnan(fitacc_nNueron_noise))[0]))+'/'+str(len(fitacc_nNueron_noise)))

        # remove nan from fitacc if there is any
        fitacc_nNueron_sig_nonan = fitacc_nNueron_sig[~np.isnan(fitacc_nNueron_sig)]
        fitacc_nNueron_noise_nonan = fitacc_nNueron_noise[~np.isnan(fitacc_nNueron_noise)]            
        fa_temp = len(np.where(fitacc_nNueron_noise_nonan==0)[0])/len(fitacc_nNueron_noise_nonan)   
            
        if len(snrsep)==0:
            hitrate_temp = len(np.where(fitacc_nNueron_sig_nonan==1)[0])/len(fitacc_nNueron_sig_nonan)
            behavdf_temp['mod'] = [mod[0][0:-1].upper()]
            behavdf_temp['hitrate0'] = [hitrate_temp]
            behavdf_temp['hitrate'] = [ProbCorctCal(hitrate_temp,fa_temp)]
            behavdf_temp['dprime'] = [dprimeCal(hitrate_temp,fa_temp)]
            behavdf_temp['nonantrials_sig'] = [len(fitacc_nNueron_sig_nonan)]
            behavdf_temp['nonantrials_noise'] = [len(fitacc_nNueron_noise_nonan)]
            behavdf = pd.concat((behavdf,behavdf_temp))
        if len(snrsep)>0:
            for cc in list(set(ylabelSig)):
                behavdf_temp_temp = pd.DataFrame()
                ccind = [ind for ind,val in enumerate(ylabelSig) if cc in val]
                fitacc_sig_temp = fitacc_nNueron_sig[ccind]
                hitrate_temp = len(np.where(fitacc_sig_temp[np.where(~np.isnan(fitacc_sig_temp))[0]]==1)[0])/np.sum(~np.isnan(fitacc_sig_temp))
                behavdf_temp_temp['mod'] = [mod[0][0:-1].upper()]
                behavdf_temp_temp['snr'] = [[float(re.search(r'_(\d+\.\d+)_', item).group(1)) for item in [cc]][0]-20]
                behavdf_temp_temp['hitrate0'] = hitrate_temp
                behavdf_temp_temp['hitrate'] = [ProbCorctCal(hitrate_temp,fa_temp)]
                behavdf_temp_temp['dprime'] = [dprimeCal(hitrate_temp,fa_temp)]
                behavdf_temp_temp['nonantrials_sig'] = [len(fitacc_sig_temp[~np.isnan(fitacc_sig_temp)])]
                behavdf_temp_temp['nonantrials_noise'] = [str(len(fitacc_nNueron_noise_nonan))+'/'+str(len(list(set(ylabelSig))))]              
                behavdf = pd.concat((behavdf,behavdf_temp_temp))
    
    return behavdf
