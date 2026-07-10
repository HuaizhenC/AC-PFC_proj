import time
from datetime import timedelta
start_time = time.monotonic()
import numpy as np
import random
import seaborn as sns
import os
import pickle
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.ticker import FuncFormatter
from spikeUtilities import SortfilterDF, getPSTHTbyT,getglmfitovertime,addIVcol,addIVcol2,extract_between_tts
from sharedparam import getMonkeyDate_all,neuronfilterDF
from scipy.stats import zscore
from scipy import stats
cpus = 20

## plot slopes
def add0Str2xtick(xt,x0str):            
    xt = [vv for vv in xt if vv!=0]
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl

def plotsigSlp(axrow,monkey,glmfitres_monk,ycolumn,huecol):   
    if huecol == []:
        sns.boxplot(glmfitres_monk,x='iv',y=ycolumn,ax=axrow,palette='Pastel2')
        sns.swarmplot(glmfitres_monk,x='iv',y=ycolumn,ax=axrow,palette='Set2',size=4)
    else:     
        hue_order=sorted(glmfitres_monk[huecol].unique())         
        sns.boxplot(glmfitres_monk,x='iv',y=ycolumn,hue=huecol,hue_order=hue_order,ax=axrow,dodge=True,palette='Pastel2')
        sns.swarmplot(glmfitres_monk,x='iv',y=ycolumn,hue=huecol,hue_order=hue_order,ax=axrow,dodge=True,palette='Set2',size=1)
        # sns.lineplot(glmfitres_monk,x='iv',y=ycolumn,hue=huecol,hue_order=sorted(glmfitres_monk[huecol].unique()),ax=axrow,palette='hsv',style=huecol,markers=True,dashes=False,estimator=None,lw=0.5)
        # ttest 
        ttestres = ''
        for name,group in glmfitres_monk.groupby('iv'):
            stats_grp,pval_grp = stats.mannwhitneyu(group[group[huecol]==hue_order[0]][ycolumn].values,group[group[huecol]==hue_order[1]][ycolumn].values,alternative='two-sided',nan_policy='omit')
            ttestres = ttestres+' '+name+':'+str(np.round(pval_grp,decimals=3))
        print(monkey+'_pval    '+ttestres)
    axrow.legend(frameon=False, framealpha=0,fontsize=fontsizeNo-4,loc='upper left',bbox_to_anchor=(1, 1), borderaxespad=0., )
    axrow.set_title('monkey '+monkey[2])
    axrow.set_ylabel('Predictor Variable Weights',fontsize=fontsizeNo)
    yt = axrow.get_yticks() 
    axrow.set_yticks([np.round(yt.min(),decimals=1),0,np.round(yt.max(),decimals=1)])


# MonkeyDate_all = {'MrCassius':['MrCassius-190414'],'MrMiyagi':['MrMiyagi-190417']} #,'MrMiyagi':['MrMiyagi-190417']
figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/FigureOutput/glmfit/'
Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/PSTH/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/glmfit/'
excelpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/pythonAnalysis/'
AVResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/AVmodIndex/'

MonkeyDate_all = getMonkeyDate_all()
# figsavPathway = '/home/huaizhen/Documents/LalittaProj/data/FigureOutput/glmfit/'
# Pathway='/home/huaizhen/Documents/LalittaProj/data/Results/PSTH/'
# ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/glmfit/' 
# AVResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/AVmodIndex/' 
# excelpathway = '/home/huaizhen/Documents/LalittaProj/pythonAnalysis/'
figformat = 'png'

fontsizeNo = 12

# fileextrastr = '_slidewin'
# timwin_all = [[-800,700],[-1000,500]]
# aligncond = [['targOn','TargOn'],['JoystickStartMove','JSOn']]

fileextrastr = '_winATevent'
timwin_all = [[0]]
aligncond =[['targOn','TargOn']]# [['targOn','TargOn']]# ,[['JoystickStartMove','JSOn']]#
fitfilter_all = [{'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['High','Low'],'correct':[1],'toneLevel':[75,90,100,110]},
               'GLM_IV':['prior','pretone','isHigh','prior-Targ','pretone-Targ','prior-pretone-Targ','toneLevel'],
               'fitdfnamestr':'_crect-soundOnTri-withpretone-4IV3interactions'}, 
               {'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['None'],'correct':[1],'toneLevel':[75,90,100,110]},
               'GLM_IV':['prior','isHigh','prior-Targ','toneLevel'],
               'fitdfnamestr':'_crect-soundOnTri-nopretone-3IV1interaction'},                                        
               ]

if __name__ == '__main__':
    # filter out drifted neurons
    dfinspect = pd.read_excel(excelpathway+'AllClusters4inspectionSheet.xlsx')
    nodriftedUnites = dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)'].isin([0])]
    # filter out neurons based on ttest
    df_avMod_all = pickle.load(open(AVResPathway+'AVmodTTestDF.pkl','rb')).reset_index()
    df_avMod_all['session_cls_Region'] = df_avMod_all['session_cls']+'_'+df_avMod_all['Region']    
    # loop through different alignment conditions
    for aligncond_temp,timwin in zip(aligncond,timwin_all):
        alignkeys = aligncond_temp[0]
        x0str = aligncond_temp[1]

        # if alignkeys=='targOn':
        #     df_avMod_A = df_avMod_all[(df_avMod_all['iv']=='A')&(df_avMod_all['pval']<0.001)]
        #     df_avMod_prefFreqSig = df_avMod_A[(df_avMod_A['prefPval']<0.001)]#filter out neurons have preferred frequency (hi/low)
        # if alignkeys=='JoystickStartMove':
        #     df_avMod_prefFreqSig = df_avMod_all[(df_avMod_all['iv']=='CHOICE')&(df_avMod_all['pval']<0.001)]#filter out neurons have preferred move direction (hi/low)
        #     df_avMod_prefFreqSig['prefHi/Lo'] = np.where(df_avMod_prefFreqSig['inhibition/excitation']==-1,'High','Low') 

        # # prepare for glmfit
        # glmfitres_all = pd.DataFrame()  
        # #start go through session-cluster 
        # for Monkey,Date in MonkeyDate_all.items():    
        #     for Date_temp in Date:
        #         AllSUspk_df = pickle.load(open(Pathway+Date_temp+'_ACnPFC_allcls_align2'+x0str+'_overlaptimwin_binlen100ms_binstep50ms_PSTH_df.pkl','rb'))        
        #         cccount = 0
        #         for area in ['AC','PFC']:                    
        #             AllSUspk_df_regiontemp = AllSUspk_df[AllSUspk_df['Region']==area]
        #             nodriftedUnites_sub = nodriftedUnites[nodriftedUnites['Region']==area]
        #             # df_avMod_prefFreq_sub = df_avMod_prefFreqSig[df_avMod_prefFreqSig['Region']==area]
        #             # df_avMod_all_sub = df_avMod_all[df_avMod_all['Region']==area]
        #             for keys in list(AllSUspk_df_regiontemp.cls.unique()):
        #                 # if Date_temp[-6:]+'_'+keys in list(nodriftedUnites_sub.session_cls.values) \
        #                 #     and Date_temp[-6:]+'_'+keys in list(df_avMod_prefFreq_sub.session_cls.values): # only non-drifted units and units have frequency preference
        #                 if Date_temp[-6:]+'_'+keys in list(nodriftedUnites_sub.session_cls.values): # only non-drifted units and units have sig modulation
                             
        #                     if any(substr in keys for substr in ['good','mua']):                            
        #                         print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
        #                         AllSUspk_df_region_cls_temp = AllSUspk_df_regiontemp[AllSUspk_df_regiontemp['cls']==keys] 

        #                         # loop through different fitting factors or data filters 
        #                         for filter_info in fitfilter_all:                                   
        #                             filter_dict = filter_info['filter_dict']
        #                             GLM_IV_list = filter_info['GLM_IV']
        #                             fitdfnamestr = filter_info['fitdfnamestr']                                   
                                    
        #                             psth_win_df_all,_ = SortfilterDF(AllSUspk_df_region_cls_temp,filterlable = filter_dict)  
        #                             # add interaction IV columns and rank each factor as meaningful numbers for the following fit, noprior/nochoice trials are nans
        #                             psth_win_df_all = addIVcol(psth_win_df_all) 
        #                             # psth_win_df_all = addIVcol2(psth_win_df_all,df_avMod_prefFreq_sub[df_avMod_prefFreq_sub['session_cls']==Date_temp[-6:]+'_'+keys]['prefHi/Lo'].unique()[0]) 
 
        #                             # filter out timepnts want to use
        #                             frcolall = [s for s in list(psth_win_df_all.columns) if '*' in s]
        #                             if len(timwin) == 2:
        #                                 frcol= [element for element in frcolall if (float(extract_between_tts(element)) >= timwin[0]) and (float(extract_between_tts(element)) <= timwin[1])]
        #                                 timpnts = sorted([float(extract_between_tts(element)) for element in frcol])
        #                             if len(timwin) == 1:
        #                                 if alignkeys == 'targOn':
        #                                     timpntsdict = {'pretone0':[0,100,200,300,400],'pretone3':[0],'pretone0+3':[0]}# window to sum spknum for different trial conditions
        #                                     if len(psth_win_df_all.pretone.unique())==1:
        #                                         frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in timpntsdict['pretone'+str(psth_win_df_all.pretone.unique()[0])]]
        #                                     if len(psth_win_df_all.pretone.unique())==2:
        #                                         frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in timpntsdict['pretone0+3']]
        #                                 if alignkeys == 'JoystickStartMove':
        #                                     frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in [-300,-200,-100]]
        #                                 psth_win_df_all['*0.00*'] = psth_win_df_all[frcol_pick4ave].sum(axis=1)
        #                                 frcol = ['*0.00*']
        #                                 timpnts = [0,0]
                                       
        #                             psth_win_df = psth_win_df_all[list(set(frcol+list(set(psth_win_df_all.columns)^set(frcolall))))]

        #                             glmfitres_all_temp = getglmfitovertime(psth_win_df,'poisson',GLM_IV_list,cpus,Monkey,Date_temp,keys,area,fitdfnamestr) # get glmfit of each IVs overtime for this neuron                                                 
        #                             glmfitres_all = pd.concat([glmfitres_all,glmfitres_all_temp])
        #                             # ensure each glm factor have expected values in the filtered spk dataframe:spikeTimedf_temp_filtered
        #                             if cccount==0:
        #                                 print(filter_info)
        #                                 print(psth_win_df.groupby(GLM_IV_list).size().reset_index().to_string())
        #                                 # for column in GLM_IV_list:
        #                                 #     # Get the unique values of the column
        #                                 #     unique_values = sorted(psth_win_df[column].unique())
        #                                 #     print(f"Unique values in '{column}': {unique_values}")
        #                         cccount = cccount+1

        # # print(glmfitres_all.to_string())
        # end_time = time.monotonic()
        # print('total fit time  ')
        # print(timedelta(seconds=end_time - start_time))
        # pickle.dump([timpnts,glmfitres_all],open(ResPathway+'glmfitCoefDF_align2'+alignkeys+fileextrastr+'.pkl','wb'))  
                            
        model_comp_param = pd.DataFrame()  
        timpnts,glmfitres_all = pickle.load(open(ResPathway+'glmfitCoefDF_align2'+alignkeys+fileextrastr+'.pkl','rb'))

        # plot units modified by different IVs separately at eventTime
        if len(timwin) == 1:
            # for iv_temp in df_avMod_all_sig.iv.unique():
            #     df_avMod_all_sig_iv_temp = df_avMod_all_sig[df_avMod_all_sig['iv']==iv_temp]
            #     clsmodifybyivgrp = list(df_avMod_all_sig_iv_temp[df_avMod_all_sig_iv_temp['iv']==iv_temp].session_cls.unique())
            #     # label neurons according to selectivity in ttest
            #     glmfitres_all['modifiedby'+iv_temp] = np.where(glmfitres_all['session_cls'].isin(clsmodifybyivgrp),1,0)
            #     namestrgroup =  '_'+iv_temp

                # namestrgroup = '_Amodified_inh-exi'
                # cmpcol = 'inhibition/excitation'
                # clsmodifybyivgrp = list(df_avMod_all_sig_iv[df_avMod_all_sig_iv[cmpcol]==1].session_cls_Region.unique())
                # # label neurons according to selectivity in ttest
                # glmfitres_all[cmpcol] = np.where(glmfitres_all['session_cls_Region'].isin(clsmodifybyivgrp),1,-1)
                # df_avMod_all_sig_iv_temp = df_avMod_all_sig_iv.copy()               

                # namestrgroup = '_all_'+cmpCond+'modFlag'               
                # clsmodifybyivgrp = list(df_avMod_all_sig_iv[df_avMod_all_sig_iv[cmpCond+'modFlag']==1].session_cls_Region.unique())
                # # label neurons according to selectivity in ttest
                # cmpcol = cmpCond+'modFlag'
                # glmfitres_all[cmpCond+'modFlag'] = np.where(glmfitres_all['session_cls_Region'].isin(clsmodifybyivgrp),1,0)
                # df_avMod_all_sig_iv_temp = df_avMod_all_sig_iv.copy()  

                namestrgroup = ''
                # loop through different fitting factors or data filters 
                for filter_info in fitfilter_all:
                    filter_dict = filter_info['filter_dict']
                    GLM_IV_list = filter_info['GLM_IV']
                    fitdfnamestr = filter_info['fitdfnamestr']  

                    ########    compare AC vs PFC  
                    glmfitres_all_sub = glmfitres_all[glmfitres_all['fitdfnamestr']==fitdfnamestr].reset_index(drop=True)
                    # filter neurons based on ttest       
                    df_avMod_all_sig = df_avMod_all[(df_avMod_all['pval']<0.001)&(df_avMod_all['iv'].isin(['LED','A','RWD','CHOICE']))]
                    glmfitres_filtered = glmfitres_all_sub[glmfitres_all_sub['session_cls_Region'].isin(list(df_avMod_all_sig['session_cls_Region'].unique()))]
                    glmfitres_filtered_sig = glmfitres_all_sub.copy()
                    
                    ycolumn = 'slope_sig' #'slope_zscore'  'slope' 'slope_minmax' 'slope_zscore_sig'
                    glmfitres_filtered_sig['slope_sig'] = np.where(glmfitres_filtered_sig['pval'] < 0.05, np.abs(glmfitres_filtered_sig['slope']),np.nan)
                    # glmfitres_filtered_sig['slope_sig'] = np.abs(glmfitres_filtered_sig['slope'])
                    
                    fig, axess = plt.subplots(2,1,figsize=(10,15),sharex='col') # 
                    # print(glmfitres_filtered_sig.to_string())                
                    plotsigSlp(axess[0],'MrCassius',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrCassius'].reset_index(drop=True),ycolumn,'Region')
                    plotsigSlp(axess[1],'MrMiyagi',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrMiyagi'].reset_index(drop=True),ycolumn,'Region')
                    axess[1].set_xlabel('IVs',fontsize=fontsizeNo)

                    fig.tight_layout()
                    fig.savefig(figsavPathway+'glmfitSlopeovertime_align2'+x0str+'_ACvsPFC_'+fitdfnamestr+namestrgroup+fileextrastr+'.png',dpi=500)
                    plt.close(fig) 

                    # for area in ['AC','PFC']:
                    #     print('plotting '+area+fitdfnamestr+'.....................................................')
                    #     glmfitres_all_sub = glmfitres_all[(glmfitres_all['Region']==area)&(glmfitres_all['fitdfnamestr']==fitdfnamestr)].reset_index(drop=True)
                    #     print('GLMmod    '+str(glmfitres_all_sub['GLMmod'].unique()))

                    #     # filter out neurons have nans at any time point
                    #     nanslopeCls = list(glmfitres_all_sub[glmfitres_all_sub['slope'].isna()]['session_cls'].unique())
                    #     # print('there are '+str(len(nanslopeCls))+' clusters out of '+str(len(glmfitres_all_sub['session_cls'].unique()))+' in '+area+' have nan glmfit slope!')
                    #     glmfitres_filtered = glmfitres_all_sub[~glmfitres_all_sub['session_cls'].isin(nanslopeCls)]                        
                    #     print('Total num of useful neurons in '+area+':'+str(len(glmfitres_filtered['session_cls'].unique())))
                        
                    #     # glmfitres_filtered_sig = glmfitres_filtered[glmfitres_filtered['pval']<0.05].copy() 
                    #     glmfitres_filtered_sig = glmfitres_filtered.copy()
                        
                    #     ycolumn = 'slope_zscore_sig' #'slope_zscore'  'slope' 'slope_minmax' 'slope_zscore_sig'
                    #     glmfitres_filtered_sig['slope_zscore'] = glmfitres_filtered_sig.groupby('iv')['slope'].transform(lambda x: zscore(x, ddof=1))
                    #     glmfitres_filtered_sig['slope_minmax'] = glmfitres_filtered_sig.groupby('iv')['slope'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
                    #     glmfitres_filtered_sig['slope_zscore_sig'] = np.where(glmfitres_filtered_sig['pval'] < 0.05, glmfitres_filtered_sig['slope_zscore'],0)
                    #     # print(glmfitres_filtered_sig[glmfitres_filtered_sig['pval']<0.0000000001].sort_values(by=['pval','time','session_cls','abs(slope)'],ascending=[True,True,True,False])[['Monkey','session_cls','Region','iv','time',ycolumn,'modifiedby'+iv_temp,'pval','fitdfnamestr']].reset_index(drop=True).to_string())                           
                        
                    #     # fig, axess = plt.subplots(2,1,figsize=(10,15),sharex='col') #                 
                    #     # plotsigSlp(axess[0],'MrCassius',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrCassius'].reset_index(drop=True),ycolumn,'session_cls')
                    #     # plotsigSlp(axess[1],'MrMiyagi',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrMiyagi'].reset_index(drop=True),ycolumn,'session_cls')
                    #     # axess[1].set_xlabel('IVs',fontsize=fontsizeNo)

                    #     # fig.tight_layout()
                    #     # fig.savefig(figsavPathway+'glmfitSlopeovertime_align2'+x0str+'_'+area+fitdfnamestr+namestrgroup+fileextrastr+'.png',dpi=500)
                    #     # plt.close(fig)   
   
    end_time = time.monotonic()
    print('total fit time:  ')
    print(timedelta(seconds=end_time - start_time))
    print('done')


