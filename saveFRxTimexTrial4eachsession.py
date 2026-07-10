import time
from datetime import timedelta
start_time = time.monotonic()
import os
import numpy as np
import pandas as pd
from spikeUtilities import getClsRasterMultiprocess,loadPreprocessMat,loadBehavMat, getPSTHTbyT,countTrialSPKs
from sharedparam import getMonkeyDate_all
cpus = 20

# save spknums aligned to an event in a set window for all clusters have fr>1

MonkeyDate_all = getMonkeyDate_all() #{'MrCassius':['MrCassius-190324']} #
Pathway='/Users/caihuaizhen1991gmail.com/Documents/LalittaProj/data/preprocNeuralMatfiles_KS3'
ResPathway = '/Users/caihuaizhen1991gmail.com/Documents/LalittaProj/results/PSTH_KS3'

# MonkeyDate_all = getMonkeyDate_all()
# Pathway='/home/huaizhen/Documents/LalittaProj/data/preprocNeuralMatfiles/'
# ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/PSTH/' 

figformat = 'png'

# get psth in non overlap windows for dimension reduction analysis
bintimeLen = 50 #ms
binmethod = 'nonoverlaptimwin'  
winstepsize=bintimeLen # 

# #### dont change the parameters for each run, if change need to rerun all sessionss
# ## get fr in slidewindows for decoding 
# bintimeLen = 100 #ms
# binmethod = 'overlaptimwin'  
# winstepsize=50 # temporal resolution: stepsize of slidewindow

timwin_all = [[-2000,1500+bintimeLen]] #ms # window for getcluster
aligncond = [['targOn','TargOn']]

# timwin_all = [[-2500,1000+bintimeLen]] #ms # window for getcluster
# aligncond = [['JoystickStartMove','JSOn']]

if __name__ == '__main__':
    # loop through different alignment conditions
    for aligncond_temp,timwin in zip(aligncond,timwin_all):
        alignkeys = aligncond_temp[0]
        x0str = aligncond_temp[1]
        timpnts = np.arange(timwin[0],timwin[1]-bintimeLen,winstepsize) # 
        
        #start go through session-cluster 
        for Monkey,Date in MonkeyDate_all.items():    
            for Date_temp in Date:
                sess_start_time = time.monotonic()
                AllSUspk_df = pd.DataFrame()
                # load behavioral data
                labeltimcombDF = loadBehavMat(Date_temp,Pathway)

                for area in ['AC','PFC']:
                    # load spike times 
                    spikeTimeDict,timeSamp2stimOn,spikefs= loadPreprocessMat(Date_temp,Pathway,area)           
                    # get trial by trial raster in each cluster, align2stimOn
                    for keys in list(spikeTimeDict.keys()):
                        if 'cls' in keys:
                            spktotal_temp = [item for sublist in spikeTimeDict[keys] for item in (sublist[0] if isinstance(sublist[0],np.ndarray) and sublist[0].ndim > 0 else [sublist[0]])]
                            fr_ave = np.count_nonzero(~np.isnan(spktotal_temp))/(len(spikeTimeDict[keys])*20) # estimate average firing rate of this cluster
                            if any(substr in keys for substr in ['good','mua']) and fr_ave>1:                            
                                print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
                                # # # get trial by trial raster in each cluster
                                # sttrials = 1200
                                # ntrials = 400               
                                # labeltimcombDF = labeltimcombDF.iloc[sttrials:(sttrials+ntrials),:] #a 2nd cluster will error
                                # spikeTime_temp = spikeTimeDict[keys][sttrials:(sttrials+ntrials)]
                                                                                               
                                spikeTime_temp = spikeTimeDict[keys]

                                # get baseline spknum before led on
                                spikeTimedf_temp_led = getClsRasterMultiprocess(spikeTime_temp,spikefs, 'soundOn','LED_On',[-300,0],labeltimcombDF,cpus) # list spike times in the 300ms window before LED onset for each trial, original spike series align to soundOn, 
                                spikeNumdf_temp_LED=countTrialSPKs(spikeTimedf_temp_led,estwin='off') #get number of spikes in the 300ms window before LED onset for each trial                         
                                spikeNumdf_temp_LED = spikeNumdf_temp_LED.rename(columns={'spknum':'spknum_300ms_b4led'})
                                spikeNumdf_temp_LED2=countTrialSPKs(spikeTimedf_temp_led,estwin='subwinoff',winTim=[-bintimeLen,0]) #get number of spikes in the bintimeLen ms window before LED onset for each trial                           
                                spikeNumdf_temp_LED2 = spikeNumdf_temp_LED2.rename(columns={'spknum':'spknum_'+str(bintimeLen)+'ms_b4led'})

                                # AllSUspk_df_ori = pickle.load(open(ResPathway+Date_temp+'_ACnPFC_allcls_align2TargOn_overlaptimwin_binlen100ms_binstep50ms_PSTH_df.pkl','rb'))
                                # AllSUspk_df_ori = pickle.load(open(ResPathway+Date_temp+'_ACnPFC_allcls_align2JSOn_overlaptimwin_binlen100ms_binstep50ms_PSTH_df.pkl','rb'))

                                # spikeNumdf_temp_LED = AllSUspk_df_ori[(AllSUspk_df_ori['cls']==keys)&(AllSUspk_df_ori['Region']==area)].sort_values('trialID')
                                # spikeNumdf_temp_LED2 = spikeNumdf_temp_LED.copy()
                                # print(spikeNumdf_temp_LED[['trialID','spknum_300ms_b4led','spknum_100ms_b4led']])

                                # get spknum before+after alignkeys
                                spikeTimedf_temp = getClsRasterMultiprocess(spikeTime_temp,spikefs,'soundOn',alignkeys,timwin,labeltimcombDF,cpus)   
                                psth_win_df = getPSTHTbyT(spikeTimedf_temp,timpnts,bintimeLen,cpus)# get psth trial by trial 
                                psth_win_df.drop(['monkeyID','Date'],axis=1,inplace=True)
                                # concatenate PSTH of each trial of each cluster
                                trials = psth_win_df.shape[0]
                                info_temp_df = pd.DataFrame({'Monkey':[Monkey]*trials,'sess':[Date_temp[-6:]]*trials,'cls':[keys]*trials,'Region':[area]*trials})  
                                AllSUspk_df_temp = pd.concat([psth_win_df.reset_index(drop=True),info_temp_df],axis=1) 
                                if (spikeNumdf_temp_LED.trialID.values==psth_win_df.trialID.values).all() and (spikeNumdf_temp_LED2.trialID.values==psth_win_df.trialID.values).all():
                                    AllSUspk_df_temp = pd.concat([AllSUspk_df_temp,spikeNumdf_temp_LED[['spknum_300ms_b4led']]],axis=1)
                                    AllSUspk_df_temp = pd.concat([AllSUspk_df_temp,spikeNumdf_temp_LED2[['spknum_'+str(bintimeLen)+'ms_b4led']]],axis=1)
                                else:
                                    print('trial mismatch, missing baseline spknum')    
                                AllSUspk_df = pd.concat([AllSUspk_df,AllSUspk_df_temp],axis=0) 
                # print(AllSUspk_df.shape)
                # print(list(AllSUspk_df.columns)[:30])
                print('time spend to get psth for this session')
                print(timedelta(seconds= time.monotonic()- sess_start_time)) 
                save_file = os.path.join(ResPathway, Date_temp+'_ACnPFC_allcls_align2'+x0str+'_'+binmethod+'_binlen'+str(bintimeLen)+'ms_binstep'+str(winstepsize)+'ms_PSTH_df.parquet')
                AllSUspk_df.to_parquet(save_file, index=False)
                # pd.read_parquet(file_path)

        print('total time spend to get all indivisual PSTH from 2 monkeys:')
        print(timedelta(seconds= time.monotonic()- start_time))        
