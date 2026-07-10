import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import os
import pickle
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt 
from spikeUtilities import loadBehavMat,getClsRasterMultiprocess,SortfilterDF,loadPreprocessMat,sampBalanceGLM,countTrialSPKs
from sharedparam import getMonkeyDate_all
cpus = 10

############# 
# estimate auditory/reward/led/choice selectivity using wilcoxon signed-rank ttest

# MonkeyDate_all = {'MrCassius':['MrCassius-190414'],'MrMiyagi':['MrMiyagi-190417']} #
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/FigureOutput/AVmodIndex/'
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/preprocNeuralMatfiles/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/AVmodIndex/'
# excelpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/pythonAnalysis/'

MonkeyDate_all = getMonkeyDate_all()
figSavePath = '/home/huaizhen/Documents/LalittaProj/data/FigureOutput/AVmodIndex/'
Pathway='/home/huaizhen/Documents/LalittaProj/data/preprocNeuralMatfiles/'
ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/AVmodIndex/' 
excelpathway = '/home/huaizhen/Documents/LalittaProj/pythonAnalysis/'
figformat = 'png'

start_time = time.monotonic()
df_avMod_all = pd.DataFrame()
df_avMod_all1 = pickle.load(open(ResPathway+'AVmodTTestDF1.pkl','rb'))
# print(df_avMod_all.to_string())
if __name__ == '__main__':
  for Monkey,Date in MonkeyDate_all.items():    
      for Date_temp in Date:
          # load behavioral data
          labeltimcombDF = loadBehavMat(Date_temp,Pathway)

          for area in ['AC','PFC']:
              # load spike times 
              spikeTimeDict,timeSamp2stimOn,spikefs= loadPreprocessMat(Date_temp,Pathway,area)           
              # get trial by trial raster in each cluster,  
              for keys in list(spikeTimeDict.keys()):
                  if 'cls' in keys:
                      spktotal_temp = [item for sublist in spikeTimeDict[keys] for item in (sublist[0] if isinstance(sublist[0],np.ndarray) and sublist[0].ndim > 0 else [sublist[0]])]
                      fr_ave = np.count_nonzero(~np.isnan(spktotal_temp))/(len(spikeTimeDict[keys])*20) # estimate average firing rate of this cluster, only neurons with fr larger than 1 are included
                      if any(substr in keys for substr in ['good','mua']) and fr_ave>1:                            
                          print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
                          # # # get trial by trial raster in each cluster
                          # sttrials = 1200
                          # ntrials = 400               
                          # labeltimcombDF = labeltimcombDF.iloc[sttrials:(sttrials+ntrials),:]
                          # spikeTime_temp = spikeTimeDict[keys][sttrials:(sttrials+ntrials)]

                          spikeTime_temp = spikeTimeDict[keys]  

                          # extrastr = 'LedOn'
                          # timwin = [-500,500]
                          # spikeTimedf_temp_led = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                          #                                     'soundOn','LED_On',\
                          #                                       timwin,labeltimcombDF,cpus)  
                          # spikeNumdf_temp_LED=countTrialSPKs(spikeTimedf_temp_led,estwin='BlNSig',winTim=timwin) #fr estimate before&after led onset                          
                          # spikeNumdf_temp_filtered_LED = SortfilterDF(spikeNumdf_temp_LED,filterlable = {'soundOnFlag':[1],'LEDFlag':[1]})[0].sort_values('trialID')
                          # try:
                          #   L_stats,L_pval = stats.wilcoxon(spikeNumdf_temp_filtered_LED['spkRate_sig'].values,
                          #                                   spikeNumdf_temp_filtered_LED['spkRate_baseline'].values,alternative='two-sided')
                          #   L_diff = np.median(spikeNumdf_temp_filtered_LED['spkRate_sig'].values-spikeNumdf_temp_filtered_LED['spkRate_baseline'].values)
                          #   trialsA = len(spikeNumdf_temp_filtered_LED['spkRate_sig'].values)
                          #   trialsB = len(spikeNumdf_temp_filtered_LED['spkRate_baseline'].values)
                          # except Exception as e:
                          #   print('stats test error in '+extrastr+f" error message: {e}")
                          #   L_stats = np.nan
                          #   L_pval = np.nan
                          #   L_diff = np.nan
                          #   trialsA = np.nan
                          #   trialsB = np.nan
                          # df_avMod_all = pd.concat((df_avMod_all,pd.DataFrame({'Monkey':Monkey,'sess':Date_temp[-6:],'session_cls':Date_temp[-6:]+'_'+keys,\
                          #                                                       'iv':'LED','stats':L_stats,'pval':L_pval,'inhibition/excitation':[-1 if L_diff<0 else 1][0],\
                          #                                                         'Region':area,'trailNumA':trialsA,'trailNumB':trialsB},index=[0])))

                          # extrastr = 'AudOn'
                          # timwin = [-150,150]
                          # spikeTimedf_temp_aud = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                          #                                     'soundOn','soundOn',\
                          #                                       timwin,labeltimcombDF,cpus)                         
                          # spikeNumdf_temp_A=countTrialSPKs(spikeTimedf_temp_aud,estwin='BlNSig',winTim=timwin) #fr estimate before&after sound onset
                          # spikeNumdf_temp_filtered_A = SortfilterDF(spikeNumdf_temp_A,filterlable = {'soundOnFlag':[1],'LEDFlag':[1],'correct':[1]})[0].sort_values('trialID')
                          # spikeNumdf_temp_filtered_LED_crct = SortfilterDF(spikeNumdf_temp_filtered_LED,filterlable = {'soundOnFlag':[1],'LEDFlag':[1],'correct':[1]})[0].sort_values('trialID')
                          # if (spikeNumdf_temp_filtered_A.trialID.values==spikeNumdf_temp_filtered_LED_crct.trialID.values).all():
                          #   try:
                          #     A_stats,A_pval = stats.wilcoxon(spikeNumdf_temp_filtered_A['spkRate_sig'].values,
                          #                                     spikeNumdf_temp_filtered_LED_crct['spkRate_baseline'].values,alternative='two-sided')
                          #     A_diff = np.median(spikeNumdf_temp_filtered_A['spkRate_sig'].values-spikeNumdf_temp_filtered_LED_crct['spkRate_baseline'].values)      
                          #     trialsA = len(spikeNumdf_temp_filtered_A['spkRate_sig'].values)
                          #     trialsB = len(spikeNumdf_temp_filtered_LED_crct['spkRate_baseline'].values)                            
                          #   except Exception as e:
                          #     print('stats test error in '+extrastr+f" error message: {e}")
                          #     A_stats = np.nan
                          #     A_pval = np.nan 
                          #     A_diff = np.nan
                          #     trialsA = np.nan
                          #     trialsB = np.nan                              
                          #   df_avMod_all = pd.concat((df_avMod_all,pd.DataFrame({'Monkey':Monkey,'sess':Date_temp[-6:],'session_cls':Date_temp[-6:]+'_'+keys,\
                          #                                                         'iv':'A','stats':A_stats,'pval':A_pval,'inhibition/excitation':[-1 if A_diff<0 else 1][0],
                          #                                                         'Region':area,'trailNumA':trialsA,'trailNumB':trialsB},index=[0])))
                          # else:
                          #    print('mismatch trialID between baseline and signal responses!!!')    
                                        
                          # extrastr = 'RWDOn'
                          # timwin = [-300,300]
                          # # add fake JoystickStartMove time for noChoice trials, separately for with/no pretone trials
                          # labeltimcombDF_withpretone = labeltimcombDF[labeltimcombDF['pretone'].isin(['High','Low'])].copy()
                          # labeltimcombDF_withpretone['JoystickStartMove'] = labeltimcombDF_withpretone['JoystickStartMove'].fillna(labeltimcombDF_withpretone['JoystickStartMove'].mean())
                          # labeltimcombDF_nopretone = labeltimcombDF[labeltimcombDF['pretone'].isin(['None'])].copy()
                          # labeltimcombDF_nopretone['JoystickStartMove'] = labeltimcombDF_nopretone['JoystickStartMove'].fillna(labeltimcombDF_nopretone['JoystickStartMove'].mean())
                          # labeltimcombDF_JStimeadd = pd.concat([labeltimcombDF_withpretone,labeltimcombDF_nopretone],axis=0).sort_values('trialID')

                          # spikeTimedf_temp_JS = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                          #                                     'soundOn','JoystickStartMove',\
                          #                                       timwin,labeltimcombDF_JStimeadd,cpus)   
                          # spikeNumdf_temp_JS=countTrialSPKs(spikeTimedf_temp_JS,estwin='BlNSig',winTim=timwin) #fr estimate beofre+after js onset                                                   
                          # spikeNumdf_temp_filtered_JS = SortfilterDF(spikeNumdf_temp_JS,filterlable = {'soundOnFlag':[1],'LEDFlag':[1],'choiceName':['Low','High']})[0].sort_values('trialID')
                          # # balance rwd/norwd trials
                          # spikeNumdf_temp_rwd = spikeNumdf_temp_filtered_JS.copy()
                          # # spikeNumdf_temp_rwd = sampBalanceGLM(spikeNumdf_temp_filtered_JS,['rewardTTL'],method='upsample')[0]
                          # # spikeNumdf_temp_rwd.sort_values(by=['rewardTTL','trialID'],kind='mergesort',inplace=True)
                          # try:
                          #   rwd_stats,rwd_pval = stats.mannwhitneyu(spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==0]['spkRate_sig'].values,
                          #                                       spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==1]['spkRate_sig'].values,alternative='two-sided')
                          #   rwd_diff = np.median(spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==1]['spkRate_sig'].values)-np.median(spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==0]['spkRate_sig'].values)
                          #   trialsA = len(spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==1]['spkRate_sig'].values)
                          #   trialsB = len(spikeNumdf_temp_rwd[spikeNumdf_temp_rwd['rewardTTL']==0]['spkRate_sig'].values)                           
                          # except Exception as e:
                          #   print('stats test error in '+extrastr+f" error message: {e}")
                          #   rwd_stats = np.nan
                          #   rwd_pval = np.nan  
                          #   rwd_diff = np.nan
                          #   trialsA = np.nan
                          #   trialsB = np.nan
                          # df_avMod_all = pd.concat((df_avMod_all,pd.DataFrame({'Monkey':Monkey,'sess':Date_temp[-6:],'session_cls':Date_temp[-6:]+'_'+keys,\
                          #                                                       'iv':'RWD','stats':rwd_stats,'pval':rwd_pval,'inhibition/excitation':[-1 if rwd_diff<0 else 1][0],
                          #                                                       'Region':area,'trailNumA':trialsA,'trailNumB':trialsB},index=[0])))

                          # extrastr = 'JSdir' 
                          # # balance High/Low choice trials
                          # spikeNumdf_temp_ch = spikeNumdf_temp_filtered_JS.copy()
                          # # spikeNumdf_temp_ch = sampBalanceGLM(spikeNumdf_temp_filtered_JS,['choiceName'],method='upsample')[0]
                          # # spikeNumdf_temp_ch.sort_values(by=['choiceName','trialID'],kind='mergesort',inplace=True)
                          # try:
                          #   ch_stats,ch_pval = stats.mannwhitneyu(spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='Low']['spkRate_baseline'].values,
                          #                                     spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='High']['spkRate_baseline'].values,alternative='two-sided')
                          #   ch_diff = np.median(spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='Low']['spkRate_baseline'].values)-np.median(spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='High']['spkRate_baseline'].values)
                          #   trialsA = len(spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='Low']['spkRate_baseline'].values)
                          #   trialsB = len(spikeNumdf_temp_ch[spikeNumdf_temp_ch['choiceName']=='High']['spkRate_baseline'].values)                            
                          # except Exception as e:
                          #   print('stats test error in '+extrastr+f" error message: {e}")
                          #   ch_stats = np.nan
                          #   ch_pval = np.nan
                          #   ch_diff = np.nan
                          #   trialsA = np.nan
                          #   trialsB = np.nan
                          # df_avMod_all = pd.concat((df_avMod_all,pd.DataFrame({'Monkey':Monkey,'sess':Date_temp[-6:],'session_cls':Date_temp[-6:]+'_'+keys,\
                          #                                                       'iv':'CHOICE','stats':ch_stats,'pval':ch_pval,'inhibition/excitation':[-1 if ch_diff<0 else 1][0],
                          #                                                       'Region':area,'trailNumA':trialsA,'trailNumB':trialsB},index=[0])))
                          
                          # extrastr = 'JSmv' 
                          # # balance with/no choice trials
                          # spikeNumdf_temp_filtered_JS2 = SortfilterDF(spikeNumdf_temp_JS,filterlable = {'soundOnFlag':[1],'LEDFlag':[1],'choiceName':['Low','High','noChoice']})[0].sort_values('trialID')
                          # spikeNumdf_temp_filtered_JS2['choiceName'] = spikeNumdf_temp_filtered_JS2['choiceName'].replace({'High':'Choice','Low':'Choice'})
                          # spikeNumdf_temp_mv = spikeNumdf_temp_filtered_JS2.copy()
                          # # spikeNumdf_temp_mv = sampBalanceGLM(spikeNumdf_temp_filtered_JS2,['choiceName'],method='upsample')[0]
                          # # spikeNumdf_temp_mv.sort_values(by=['choiceName','trialID'],kind='mergesort',inplace=True)
                          # try:
                          #   mv_stats,mv_pval = stats.mannwhitneyu(spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='noChoice']['spkRate_baseline'].values,
                          #                                     spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='Choice']['spkRate_baseline'].values,alternative='two-sided')
                          #   mv_diff = np.median(spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='noChoice']['spkRate_baseline'].values)-np.median(spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='Choice']['spkRate_baseline'].values)
                          #   trialsA = len(spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='noChoice']['spkRate_baseline'].values)
                          #   trialsB = len(spikeNumdf_temp_mv[spikeNumdf_temp_mv['choiceName']=='Choice']['spkRate_baseline'].values)                            
                          # except Exception as e:
                          #   print('stats test error in '+extrastr+f" error message: {e}")
                          #   mv_stats =np.nan
                          #   mv_pval =np.nan
                          #   mv_diff = np.nan
                          #   trialsA = np.nan
                          #   trialsB = np.nan
                          # df_avMod_all = pd.concat((df_avMod_all,pd.DataFrame({'Monkey':Monkey,'sess':Date_temp[-6:],'session_cls':Date_temp[-6:]+'_'+keys,\
                          #                                                       'iv':'MOVE','stats':mv_stats,'pval':mv_pval,'inhibition/excitation':[-1 if mv_diff<0 else 1][0],
                          #                                                       'Region':area,'trailNumA':trialsA,'trailNumB':trialsB},index=[0])))

                          # add hi/lo frequency preference
                          df_avMod_all_temp = df_avMod_all1[(df_avMod_all1['Monkey']==Monkey)
                                                           &(df_avMod_all1['session_cls']==Date_temp[-6:]+'_'+keys)
                                                           &(df_avMod_all1['Region']==area)].reset_index(drop=True)                                                              
                          extrastr = 'TargOn'
                          timwin = [-150,150]
                          spikeTimedf_temp_aud = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                                                              'soundOn','targOn',\
                                                                timwin,labeltimcombDF,cpus)                         
                          spikeNumdf_temp_A=countTrialSPKs(spikeTimedf_temp_aud,estwin='BlNSig',winTim=timwin) #fr estimate before&after sound target onset
                          spikeNumdf_temp_filtered_T = SortfilterDF(spikeNumdf_temp_A,filterlable = {'soundOnFlag':[1],'LEDFlag':[1],'correct':[1],'toneLevel':[75,90,100,110]})[0].sort_values('trialID')
                          try:
                              T_stats,T_pval = stats.mannwhitneyu(spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='High']['spkRate_sig'].values,
                                                            spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='Low']['spkRate_sig'].values,alternative='two-sided')
                              T_diff = np.mean(spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='High']['spkRate_sig'].values)\
                                        -np.mean(spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='Low']['spkRate_sig'].values)      
                              trialsA = len(spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='High']['spkRate_sig'].values)
                              trialsB = len(spikeNumdf_temp_filtered_T[spikeNumdf_temp_filtered_T['isHigh']=='Low']['spkRate_sig'].values)                            
                          except Exception as e:
                            print('stats test error in '+extrastr+f" error message: {e}")
                            T_stats = np.nan
                            T_pval = np.nan 
                            T_diff = np.nan
                            trialsA = np.nan
                            trialsB = np.nan    
                            
                          df_avMod_all = pd.concat((df_avMod_all,pd.concat([df_avMod_all_temp,pd.DataFrame({'prefHi/Lo':['Low' if T_diff<0 else 'High' if T_diff>0 else 'noPref']*df_avMod_all_temp.shape[0],
                                                                                                            'prefPval':[T_pval]*df_avMod_all_temp.shape[0]})],axis=1)),axis=0)
      print(df_avMod_all.to_string())
      pickle.dump(df_avMod_all,open(ResPathway+'AVmodTTestDF.pkl','wb'))            
  
  # df_avMod_all = pickle.load(open(ResPathway+'AVmodTTestDF.pkl','rb')).reset_index()
  # datasig = df_avMod_all[df_avMod_all['pval']<0.001] 
  # # filter out drifted neurons
  # dfinspect = pd.read_excel(excelpathway+'AllClusters4inspectionSheet.xlsx')
  # nodriftedUnites = dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)'].isin([0])]
  # datasig_filter = pd.DataFrame()
  # for mm in ['MrCassius','MrMiyagi']:
  #   datasig_mm_temp = datasig[datasig['Monkey']==mm]
  #   for rr in ['AC','PFC']:
  #     datasig_mm_rr_temp = datasig_mm_temp[datasig_mm_temp['Region']==rr]
  #     nodriftedUnites_temp = nodriftedUnites[(nodriftedUnites['Monkey']==mm)&(nodriftedUnites['Region']==rr)]
  #     datasig_mm_rr_filter_temp = datasig_mm_rr_temp[datasig_mm_rr_temp['session_cls'].isin(nodriftedUnites_temp.session_cls.values)]
  #     datasig_filter = pd.concat([datasig_filter,datasig_mm_rr_filter_temp],axis=0)
  #     print(mm+': SigUnits in '+rr+' #'+str(len(datasig_mm_rr_filter_temp.session_cls.unique())))
  # print(datasig_filter.to_string())

  # ## plot number of trials used in each condition for each test 
  # fig, axess = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1,1]})
  # df_avMod_all_uniquesession = df_avMod_all.groupby(['Monkey','sess','iv'])[['trailNumA','trailNumB']].mean().reset_index()
  # sns.scatterplot(df_avMod_all_uniquesession[df_avMod_all_uniquesession['Monkey']=='MrCassius'],x='trailNumA',y='trailNumB',hue='iv',ax=axess[0])
  # sns.scatterplot(df_avMod_all_uniquesession[df_avMod_all_uniquesession['Monkey']=='MrMiyagi'],x='trailNumA',y='trailNumB',hue='iv',ax=axess[1])
  # axess[0].set_title('MrCassius')
  # axess[1].set_title('MrMiyagi')
  # plt.tight_layout()
  # plt.savefig(figSavePath+'AVmodIndexTest_trialInbalance4IVs.png')
  # plt.close()
 
  # ## plot number of inhib/excitory units in each IV and monkey and brain region condition
  # fig, axess = plt.subplots(2,2,figsize=(12,12),gridspec_kw={'width_ratios': [1,1]})
  # datasig_filter_temp = datasig_filter.copy()#dropna('inhibition/excitation')
  # sns.countplot(datasig_filter_temp[(datasig_filter_temp['Region']=='AC')&(datasig_filter_temp['Monkey']=='MrCassius')],x='iv',hue='inhibition/excitation',ax=axess[0,0])
  # sns.countplot(datasig_filter_temp[(datasig_filter_temp['Region']=='AC')&(datasig_filter_temp['Monkey']=='MrMiyagi')],x='iv',hue='inhibition/excitation',ax=axess[1,0])
  # axess[0,0].set_title('MrCassius-AC')
  # axess[1,0].set_title('MrMiyagi-AC')  
  # sns.countplot(datasig_filter_temp[(datasig_filter_temp['Region']=='PFC')&(datasig_filter_temp['Monkey']=='MrCassius')],x='iv',hue='inhibition/excitation',ax=axess[0,1])
  # sns.countplot(datasig_filter_temp[(datasig_filter_temp['Region']=='PFC')&(datasig_filter_temp['Monkey']=='MrMiyagi')],x='iv',hue='inhibition/excitation',ax=axess[1,1])
  # axess[0,1].set_title('MrCassius-PFC')
  # axess[1,1].set_title('MrMiyagi-PFC')  
  # plt.tight_layout()
  # plt.savefig(figSavePath+'AVmodIndexTest_inhORexi_UnitsNum4IVs.png')
  # plt.close()

  # ## plot num of sig units
  # fig, axess = plt.subplots(2,2,figsize=(12,12),gridspec_kw={'width_ratios': [1,1]})
  # sns.stripplot(datasig_filter[datasig_filter['Region']=='AC'],x='iv',y='stats',hue='session_cls',ax=axess[0,0],legend=False)
  # sns.countplot(datasig_filter[datasig_filter['Region']=='AC'],x='iv',hue='Monkey',ax=axess[0,1])
  # sns.stripplot(datasig_filter[datasig_filter['Region']=='PFC'],x='iv',y='stats',hue='session_cls',ax=axess[1,0],legend=False)
  # sns.countplot(datasig_filter[datasig_filter['Region']=='PFC'],x='iv',hue='Monkey',ax=axess[1,1])
  # axess[0,0].set_title('AC')
  # axess[1,0].set_title('PFC')
  # plt.tight_layout()
  # plt.savefig(figSavePath+'AVmodIndexTest_byCluster.png')
  # plt.close()


  end_time = time.monotonic()
  print(timedelta(seconds=end_time - start_time))


