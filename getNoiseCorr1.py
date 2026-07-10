import time
from datetime import timedelta

import numpy as np
import random
import seaborn as sns
import os
import pickle
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.ticker import FuncFormatter
from spikeUtilities import SortfilterDF, getPSTHTbyT,getglmfitovertime,addIVcol,addIVcol2,extract_between_tts,estNoiseCorr,sampBalanceGLM
from sharedparam import getMonkeyDate_all,neuronfilterDF
from scipy.stats import zscore
from scipy import stats
cpus = 20

def plotfitcorr(NCdf,axess,catstr, xvar, yvar, colrs, sortlist,titlestr):
    catorder = NCdf[catstr].unique()
    for ii, (cat,colr) in enumerate(zip(catorder,colrs)):
        NCdf_temp = NCdf[NCdf[catstr]==cat]
        axess.scatter(NCdf_temp.sort_values(sortlist,kind='mergesort')[xvar].values,
                    NCdf_temp.sort_values(sortlist,kind='mergesort')[yvar].values,
                        alpha=0.3,c=colr,label=str(cat),marker='o',edgecolors='none') 
        statsRes = stats.wilcoxon(NCdf_temp.sort_values(sortlist,kind='mergesort')[xvar].values,
                                    NCdf_temp.sort_values(sortlist,kind='mergesort')[yvar].values,nan_policy='omit')
        meandiff = np.nanmean(NCdf_temp.sort_values(sortlist,kind='mergesort')[xvar].values)-np.nanmean(NCdf_temp.sort_values(sortlist,kind='mergesort')[yvar].values)
        diffstr = [' Xy' if meandiff>0 else ' xY' if meandiff<0 else ' xy'][0]
        axess.text(0.7,-0.55-ii*0.09,str(cat)+' p='+str(np.round(statsRes[1],decimals=5))+diffstr,horizontalalignment='right',verticalalignment='center',fontsize=10)
    axess.set_title(titlestr)
    axess.set_xlim([-1,1])
    axess.set_ylim([-1,1])         
    axess.legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
    axess.plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'--',color='gray',linewidth=1.5)  
    axess.set_xlabel(xvar,fontsize=fontsizeNo)
    axess.set_ylabel(yvar,fontsize=fontsizeNo) 

def plotNCbar(axrow,monkey,glmfitres_monk,xcolumn,huecol):   
    if huecol == []:
        # sns.boxplot(glmfitres_monk,x=xcolumn,y='corrcoef',ax=axrow,palette='Pastel2')
        # sns.swarmplot(glmfitres_monk,x=xcolumn,y='corrcoef',ax=axrow,palette='Set2',size=1)
        sns.lineplot(glmfitres_monk,x=xcolumn,y='corrcoef',hue='NeuPairs',hue_order=sorted(glmfitres_monk['NeuPairs'].unique()),ax=axrow,estimator='mean',errorbar=('ci',95),palette='Set2',linewidth=1)

        # sns.lineplot(glmfitres_monk.groupby(['NeuPairs',xcolumn])['corrcoef'].mean().reset_index(),x=xcolumn,y='corrcoef',hue='NeuPairs',hue_order=sorted(glmfitres_monk['NeuPairs'].unique()),ax=axrow,estimator='mean',errorbar=('ci',95),palette='Set2')
    else:     
        hue_order=sorted(glmfitres_monk[huecol].unique())         
        sns.boxplot(glmfitres_monk,x=xcolumn,y='corrcoef',hue=huecol,hue_order=hue_order,ax=axrow,dodge=True,palette='Pastel2')
        sns.swarmplot(glmfitres_monk,x=xcolumn,y='corrcoef',hue=huecol,hue_order=hue_order,ax=axrow,dodge=True,palette='Set2',size=1)
        # ttest 
        ttestres = ''
        for name,group in glmfitres_monk.groupby(xcolumn):
            stats_grp,pval_grp = stats.mannwhitneyu(group[group[huecol]==hue_order[0]]['corrcoef'].values,group[group[huecol]==hue_order[1]]['corrcoef'].values,alternative='two-sided',nan_policy='omit')
            ttestres = ttestres+' '+str(name)+':'+str(np.round(pval_grp,decimals=3))
        print(monkey+'_pval    '+ttestres)
    axrow.legend([],frameon=False, framealpha=0,fontsize=fontsizeNo-4,loc='upper left',bbox_to_anchor=(1, 1), borderaxespad=0., )
    axrow.set_title('monkey '+monkey[2])
    axrow.set_ylabel('NC',fontsize=fontsizeNo)
    yt = axrow.get_yticks() 
    axrow.set_yticks([np.round(yt.min(),decimals=1),0,np.round(yt.max(),decimals=1)])


# MonkeyDate_all = {'MrCassius':['MrCassius-190418']} #,'MrMiyagi':['MrMiyagi-190417']
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/FigureOutput/noisecorrelation/'
# Pathway= '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/PSTH/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/noisecorrelation/'
# AVResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/AVmodIndex/'
# excelpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/pythonAnalysis/'

MonkeyDate_all = getMonkeyDate_all()
figSavePath = '/home/huaizhen/Documents/LalittaProj/data/FigureOutput/noisecorrelation/'
Pathway= '/home/huaizhen/Documents/LalittaProj/data/Results/PSTH/'
ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/noisecorrelation/'
AVResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/AVmodIndex/' 
excelpathway = '/home/huaizhen/Documents/LalittaProj/pythonAnalysis/'


fontsizeNo = 15

# fileextrastr = '_slidewin'
# timwin_all = [[-800,700],[-1000,500]]
# aligncond = [['targOn','TargOn'],['JoystickStartMove','JSOn']]

fileextrastr = '_winATevent'
# timwin_all = [[0],[0]]
# aligncond =[['targOn','TargOn'],['JoystickStartMove','JSOn']]# [['targOn','TargOn']]# ,[['JoystickStartMove','JSOn']]#
timwin_all = [[0]]
aligncond =[['targOn','TargOn']]# [['targOn','TargOn']]# ,[['JoystickStartMove','JSOn']]#

fitfilter_all = [{'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['High','Low'],'correct':[1],'toneLevel':[75,90,100,110]},
               'cmpIVs':['prior-pretone-Targ','toneLevel'],
               'nameStr':'_crect-soundOnTri-withpretone'}, 
               {'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['None'],'correct':[1],'toneLevel':[75,90,100,110]},
               'cmpIVs':['prior-Targ','toneLevel'],
               'nameStr':'_crect-soundOnTri-nopretone'},                                        
               ]

if __name__ == '__main__':
    # filter out drifted neurons
    df_avMod_all_filtered,_ = neuronfilterDF(excelpathway,AVResPathway,'ttest')

    # loop through different alignment conditions
    for aligncond_temp,timwin in zip(aligncond,timwin_all):
        alignkeys = aligncond_temp[0]
        x0str = aligncond_temp[1]

        # loop through differet filter conditions
        for fitfilter_temp in fitfilter_all:
            filter_dict = fitfilter_temp['filter_dict']
            cmpIVs = fitfilter_temp['cmpIVs']
            nameStr = fitfilter_temp['nameStr']
            
            start_time = time.monotonic()

            # prepare 
            AllSUspk_df_4decoding = pd.DataFrame()  
            #start go through session-cluster 
            for Monkey,Date in MonkeyDate_all.items():    
                for Date_temp in Date:
                    AllSUspk_df = pickle.load(open(Pathway+Date_temp+'_ACnPFC_allcls_align2'+x0str+'_overlaptimwin_binlen100ms_binstep50ms_PSTH_df.pkl','rb'))        
                    cccount = 0
                    for area in ['AC','PFC']:                    
                        AllSUspk_df_regiontemp = AllSUspk_df[AllSUspk_df['Region']==area]
                        for keys in list(AllSUspk_df_regiontemp.cls.unique()):
                            if Date_temp[-6:]+'_'+keys+'_'+area in list(df_avMod_all_filtered.session_cls_Region.values): # only non-drifted units and units have sig modulation                             
                                if any(substr in keys for substr in ['good','mua']):                            
                                    print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
                                    AllSUspk_df_region_cls_temp = AllSUspk_df_regiontemp[AllSUspk_df_regiontemp['cls']==keys]  
                                    psth_win_df_all,_ = SortfilterDF(AllSUspk_df_region_cls_temp,filterlable = filter_dict)  
                                    # add interaction IV columns and rank each factor as meaningful numbers for the following fit, noprior/nochoice trials are nans
                                    psth_win_df_all = addIVcol(psth_win_df_all) 
                                    # psth_win_df_all = addIVcol2(psth_win_df_all,df_avMod_prefFreq_sub[df_avMod_prefFreq_sub['session_cls']==Date_temp[-6:]+'_'+keys]['prefHi/Lo'].unique()[0]) 

                                    # filter out timepnts want to use
                                    frcolall = [s for s in list(psth_win_df_all.columns) if '*' in s]
                                    if len(timwin) == 2:
                                        frcol= [element for element in frcolall if (float(extract_between_tts(element)) >= timwin[0]) and (float(extract_between_tts(element)) <= timwin[1])]
                                        timpnts = sorted([float(extract_between_tts(element)) for element in frcol])
                                    if len(timwin) == 1:
                                        if alignkeys == 'targOn':
                                            # timpntsdict = {'pretone0':[0,100,200,300,400],'pretone3':[0],'pretone0+3':[0]}# window to sum spknum for different trial conditions
                                            timpntsdict = {'pretone0':[0],'pretone3':[0],'pretone0+3':[0]}# window to sum spknum for different trial conditions

                                            if len(psth_win_df_all.pretone.unique())==1:
                                                frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in timpntsdict['pretone'+str(psth_win_df_all.pretone.unique()[0])]]
                                            if len(psth_win_df_all.pretone.unique())==2:
                                                frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in timpntsdict['pretone0+3']]
                                        if alignkeys == 'JoystickStartMove':
                                            # frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in [-300,-200,-100]]
                                            frcol_pick4ave =  [element for element in frcolall if (float(extract_between_tts(element))) in [0,100,200,300]]

                                        psth_win_df_all['*0.00*'] = psth_win_df_all[frcol_pick4ave].sum(axis=1)
                                        frcol = ['*0.00*']
                                        timpnts = [0,0]
                                        
                                        psth_win_df = psth_win_df_all[list(set(frcol+list(set(psth_win_df_all.columns)^set(frcolall))))]

                                    AllSUspk_df_4decoding = pd.concat([AllSUspk_df_4decoding,psth_win_df],axis=0)
                                    
            end_time = time.monotonic()
            print('total save time  ')
            print(timedelta(seconds=end_time - start_time))
            pickle.dump([AllSUspk_df_4decoding,frcol],open(ResPathway+'AllSUspk_df_align2'+alignkeys+nameStr+'.pkl','wb')) 

            AllSUspk_df_4decoding,frcol = pickle.load(open(ResPathway+'AllSUspk_df_align2'+alignkeys+nameStr+'.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes
            
            NCdf = pd.DataFrame()
            # ##########################################calculate NC within regions  
            for sess_temp in AllSUspk_df_4decoding.sess.unique(): 
                randomseed = random.randint(1, 10000) # within a session using the same trials to est NC in all regions
                for area in AllSUspk_df_4decoding.Region.unique():
                    SUspk_df_sess_region = AllSUspk_df_4decoding[(AllSUspk_df_4decoding['sess']==sess_temp)&(AllSUspk_df_4decoding['Region']==area)]
                    SUspk_df_sess_region_cp = SUspk_df_sess_region.copy()
                    # group snrs into 2 cat 75,90,100,110 when est NC
                    SUspk_df_sess_region['toneLevel']=SUspk_df_sess_region['toneLevel'].replace({75:'difficult',90:'difficult',100:'easy',110:'easy'})                    
                    cls_temp1 = list(SUspk_df_sess_region.session_cls_Region.unique())
                    cls_temp = [cc for cc in cls_temp1 if 'good' in cc]
                    # cls_temp = cls_temp1

                    if len(cls_temp)>1 and SUspk_df_sess_region.shape[0]>0: # only calculate NC in sessions with more than 2 single units
                        Monkey  = SUspk_df_sess_region.Monkey.unique()[0]
                        print('...............'+'estimating noisecorr measurement in '+Monkey+'-'+sess_temp+'-'+area+'...............')
                        # select trials with balanced number across categories for this session
                        SUspk_df_sess_region_temp = SUspk_df_sess_region_cp[SUspk_df_sess_region_cp['session_cls_Region']==cls_temp[0]].sort_values('trialID').reset_index(drop=True)
                        _,selected_rows = sampBalanceGLM(SUspk_df_sess_region_temp,['prior','pretone','isHigh','toneLevel'],seeds=randomseed,method='downsample')# balancing trials accoridng to original condition labels
                        for cc1, cls1 in enumerate(cls_temp[:-1]):
                            AllSUspk_df_4decoding_sess_cls1 = SUspk_df_sess_region[SUspk_df_sess_region['session_cls_Region']==cls1].sort_values('trialID').reset_index(drop=True)
                            SPK = AllSUspk_df_4decoding_sess_cls1.loc[selected_rows].sort_values(['trialID'],kind='mergesort')                    
                            for cc2, cls2 in enumerate(cls_temp[cc1+1:]):
                                print([cls1,cls2])
                                AllSUspk_df_4decoding_sess_cls2 = SUspk_df_sess_region[SUspk_df_sess_region['session_cls_Region']==cls2].sort_values('trialID').reset_index(drop=True)
                                SPK2 = AllSUspk_df_4decoding_sess_cls2.loc[selected_rows].sort_values(['trialID'],kind='mergesort')   
                                #estimate noise correlation
                                NCdf_temp = estNoiseCorr(SPK,SPK2,[cls1,cls2],frcol,cond=cmpIVs,cpus=cpus)
                                NCdf_temp['sess'] = sess_temp
                                NCdf_temp['Region'] = area
                                NCdf_temp['Monkey'] = Monkey
                                NCdf = pd.concat((NCdf,NCdf_temp))                        
            pickle.dump(NCdf,open(ResPathway+'NC_all_df_align2'+alignkeys+nameStr+'.pkl','wb'))                       
            NCdf = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+nameStr+'.pkl','rb'))
            

#             # ##########################################calculate NC across regions  
#             # for sess_temp in AllSUspk_df_4decoding.sess.unique(): 
#             #     randomseed = random.randint(1, 10000) # within a session using the same trials to est NC in all regions
#             #     SUspk_df_sess_AC = AllSUspk_df_4decoding[(AllSUspk_df_4decoding['sess']==sess_temp)&(AllSUspk_df_4decoding['Region']=='AC')]
#             #     SUspk_df_sess_PFC = AllSUspk_df_4decoding[(AllSUspk_df_4decoding['sess']==sess_temp)&(AllSUspk_df_4decoding['Region']=='PFC')]
#             #     SUspk_df_sess_cp = AllSUspk_df_4decoding.copy()
#             #     # group snrs into 2 cat 75,90,100,110 when est NC
#             #     SUspk_df_sess_AC['toneLevel']=SUspk_df_sess_AC['toneLevel'].replace({75:'difficult',90:'difficult',100:'easy',110:'easy'}) 
#             #     SUspk_df_sess_PFC['toneLevel']=SUspk_df_sess_PFC['toneLevel'].replace({75:'difficult',90:'difficult',100:'easy',110:'easy'}) 
#             #     # get clustIDs for each brain region
#             #     cls_temp_AC = [cc for cc in list(SUspk_df_sess_AC.session_cls_Region.unique()) if 'good' in cc]
#             #     cls_temp_PFC = [cc for cc in list(SUspk_df_sess_PFC.session_cls_Region.unique()) if 'good' in cc]

#             #     if len(cls_temp_AC)>1 and len(cls_temp_PFC)>1: # only calculate NC in sessions with more than 1 single units
#             #         Monkey  = SUspk_df_sess_AC.Monkey.unique()[0]
#             #         print('...............'+'estimating noisecorr measurement in '+Monkey+'-'+sess_temp+'...............')
#             #         # select trials with balanced number across categories for this session
#             #         SUspk_df_sess_temp = SUspk_df_sess_cp[SUspk_df_sess_cp['session_cls_Region']==cls_temp_AC[0]].sort_values('trialID').reset_index(drop=True)
#             #         _,selected_rows = sampBalanceGLM(SUspk_df_sess_temp,['prior','pretone','isHigh','toneLevel'],seeds=randomseed,method='downsample')# balancing trials accoridng to original condition labels
#             #         for cc1, cls1 in enumerate(cls_temp_AC):
#             #             AllSUspk_df_4decoding_sess_cls1 = SUspk_df_sess_AC[SUspk_df_sess_AC['session_cls_Region']==cls1].sort_values('trialID').reset_index(drop=True)
#             #             SPK = AllSUspk_df_4decoding_sess_cls1.loc[selected_rows].sort_values(['trialID'],kind='mergesort')                    
#             #             for cc2, cls2 in enumerate(cls_temp_PFC):
#             #                 print([cls1,cls2])
#             #                 AllSUspk_df_4decoding_sess_cls2 = SUspk_df_sess_PFC[SUspk_df_sess_PFC['session_cls_Region']==cls2].sort_values('trialID').reset_index(drop=True)
#             #                 SPK2 = AllSUspk_df_4decoding_sess_cls2.loc[selected_rows].sort_values(['trialID'],kind='mergesort')   
#             #                 #estimate noise correlation
#             #                 NCdf_temp = estNoiseCorr(SPK,SPK2,[cls1,cls2],frcol,cond=cmpIVs,cpus=cpus)
#             #                 NCdf_temp['sess'] = sess_temp
#             #                 NCdf_temp['Monkey'] = Monkey
#             #                 NCdf = pd.concat((NCdf,NCdf_temp))                        
#             # pickle.dump(NCdf,open(ResPathway+'NC_all_df_align2'+alignkeys+nameStr+'_Xregions.pkl','wb'))                       
#             # NCdf = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+nameStr+'_Xregions.pkl','rb'))

#             ######################## plot NC ################### 
#             # NCdf_filtered = NCdf.dropna(subset=['sig'],inplace=False)
#             print(NCdf.iloc[:10].to_string())
#             # NCdf_filtered = NCdf[NCdf['sig']!=-1]
#             NCdf_filtered = NCdf.copy()            
       
# #####################################plot within region NC
#             # compare NC over congruency
#             for snr in ['easy','difficult']:
#                 fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col') #
#                 plotNCbar(axess[0],'MrCassius',NCdf_filtered[(NCdf_filtered['Monkey']=='MrCassius')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True),cmpIVs[0],'Region')
#                 plotNCbar(axess[1],'MrMiyagi',NCdf_filtered[(NCdf_filtered['Monkey']=='MrMiyagi')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True),cmpIVs[0],'Region')
#                 fig.tight_layout()
#                 fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_'+snr+'-toneLevel_align2'+alignkeys+nameStr+'.png')
#                 plt.close(fig)   
#             #  compare snr conditions, color different prior-pretone congruency in different colorsf
#             fig, axess = plt.subplots(2,2,figsize=(8,8),sharex='col') #
#             cat = sorted(NCdf_filtered[cmpIVs[0]].unique())
#             # Load the tab10 colormap
#             tab10 = plt.get_cmap('tab10')
#             # Get the first 10 colors
#             colrs = [tab10(i) for i in range(len(cat))]
#             # colrs = ['lightpink','lightcoral','orangered','indianred','maroon'] 
#             for ii,ppt in enumerate(cat):
#                 NCdf_filtered_Monk1 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrCassius')&(NCdf_filtered[cmpIVs[0]]==ppt)].reset_index(drop=True)
#                 NCdf_filtered_Monk2 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrMiyagi')&(NCdf_filtered[cmpIVs[0]]==ppt)].reset_index(drop=True)

#                 axess[0,0].scatter(NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='easy')&(NCdf_filtered_Monk1['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                     NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='difficult')&(NCdf_filtered_Monk1['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                 alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#                 statsRes = stats.wilcoxon(NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='easy')&(NCdf_filtered_Monk1['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                             NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='difficult')&(NCdf_filtered_Monk1['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#                 axess[0,0].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#                 axess[0,0].set_title('AC')
#                 axess[0,0].set_xlim([-1,1])
#                 axess[0,0].set_ylim([-1,1])         
#                 axess[0,0].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#                 axess[0,0].plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'--',color='gray',linewidth=1.5)  
#                 axess[0,0].set_xlabel('easy',fontsize=fontsizeNo)
#                 axess[0,0].set_ylabel('difficult',fontsize=fontsizeNo) 

#                 axess[0,1].scatter(NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='easy')&(NCdf_filtered_Monk1['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                     NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='difficult')&(NCdf_filtered_Monk1['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                 alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#                 statsRes = stats.wilcoxon(NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='easy')&(NCdf_filtered_Monk1['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                             NCdf_filtered_Monk1[(NCdf_filtered_Monk1['toneLevel']=='difficult')&(NCdf_filtered_Monk1['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#                 axess[0,1].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#                 axess[0,1].set_title('PFC')
#                 axess[0,1].set_xlim([-1,1])
#                 axess[0,1].set_ylim([-1,1])         
#                 axess[0,1].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#                 axess[0,1].plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'--',color='gray',linewidth=1.5) 
#                 axess[0,1].set_xlabel('easy',fontsize=fontsizeNo)
#                 axess[0,1].set_ylabel('difficult',fontsize=fontsizeNo)  

#                 axess[1,0].scatter(NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='easy')&(NCdf_filtered_Monk2['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                     NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='difficult')&(NCdf_filtered_Monk2['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                 alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#                 statsRes = stats.wilcoxon(NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='easy')&(NCdf_filtered_Monk2['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                             NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='difficult')&(NCdf_filtered_Monk2['Region']=='AC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#                 axess[1,0].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#                 axess[1,0].set_title('AC')
#                 axess[1,0].set_xlim([-1,1])
#                 axess[1,0].set_ylim([-1,1])         
#                 axess[1,0].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#                 axess[1,0].plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'--',color='gray',linewidth=1.5)  
#                 axess[1,0].set_xlabel('easy',fontsize=fontsizeNo)
#                 axess[1,0].set_ylabel('difficult',fontsize=fontsizeNo) 

#                 axess[1,1].scatter(NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='easy')&(NCdf_filtered_Monk2['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                     NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='difficult')&(NCdf_filtered_Monk2['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                 alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#                 statsRes = stats.wilcoxon(NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='easy')&(NCdf_filtered_Monk2['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#                                             NCdf_filtered_Monk2[(NCdf_filtered_Monk2['toneLevel']=='difficult')&(NCdf_filtered_Monk2['Region']=='PFC')].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#                 axess[1,1].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#                 axess[1,1].set_title('PFC')
#                 axess[1,1].set_xlim([-1,1])
#                 axess[1,1].set_ylim([-1,1])         
#                 axess[1,1].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#                 axess[1,1].plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'--',color='gray',linewidth=1.5)  
#                 axess[1,1].set_xlabel('easy',fontsize=fontsizeNo)
#                 axess[1,1].set_ylabel('difficult',fontsize=fontsizeNo)                       
#             fig.tight_layout()
#             fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+nameStr+'_CMPtoneLevel.png')
#             plt.close(fig) 



# #####################################plot cross region NC
#             # for snr in ['easy','difficult']:
#             #     fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col') #
#             #     plotNCbar(axess[0],'MrCassius',NCdf_filtered[(NCdf_filtered['Monkey']=='MrCassius')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True),cmpIVs[0],[])
#             #     plotNCbar(axess[1],'MrMiyagi',NCdf_filtered[(NCdf_filtered['Monkey']=='MrMiyagi')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True),cmpIVs[0],[])
#             #     fig.tight_layout()
#             #     fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+nameStr+'_toneLevel-'+snr+'_Xregions.png')
#             #     plt.close(fig)  

#             # # plot to compare ppt conditions 
#             # for snr in ['easy','difficult']:
#             #     fig, axess = plt.subplots(2,1,figsize=(4,8),sharex='col') #
#             #     cat = sorted(NCdf_filtered[cmpIVs[0]].unique())
#             #     # Load the tab10 colormap
#             #     tab10 = plt.get_cmap('tab10')
#             #     # Get the first 10 colors
#             #     colrs = [tab10(i) for i in range(len(cat))]            
#             #     # colrs = ['lightpink','lightcoral','orangered','indianred','maroon'] 

#             #     NCdf_filtered_Monk1 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrCassius')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True)
#             #     NCdf_filtered_Monk2 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrMiyagi')&(NCdf_filtered['toneLevel']==snr)].reset_index(drop=True)

#             #     for ii,ppt in enumerate(cat[1:]):
#             #         axess[0].scatter(NCdf_filtered_Monk1[NCdf_filtered_Monk1[cmpIVs[0]]==cat[0]].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                          NCdf_filtered_Monk1[NCdf_filtered_Monk1[cmpIVs[0]]==ppt].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                         alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#             #         statsRes = stats.wilcoxon(NCdf_filtered_Monk1[NCdf_filtered_Monk1[cmpIVs[0]]==cat[0]].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                                   NCdf_filtered_Monk1[NCdf_filtered_Monk1[cmpIVs[0]]==ppt].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#             #         # axess[0].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=10)
#             #         axess[0].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)

#             #         axess[1].scatter(NCdf_filtered_Monk2[NCdf_filtered_Monk2[cmpIVs[0]]==cat[0]].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                          NCdf_filtered_Monk2[NCdf_filtered_Monk2[cmpIVs[0]]==ppt].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                         alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#             #         statsRes = stats.wilcoxon(NCdf_filtered_Monk2[NCdf_filtered_Monk2[cmpIVs[0]]==cat[0]].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                                   NCdf_filtered_Monk2[NCdf_filtered_Monk2[cmpIVs[0]]==ppt].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#             #         axess[1].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#             #     axisLoBound = min([axess[0].get_xlim()[0],axess[0].get_ylim()[0],axess[1].get_xlim()[0],axess[1].get_ylim()[0]])
#             #     axisHiBound = max([axess[0].get_xlim()[1],axess[0].get_ylim()[1],axess[1].get_xlim()[1],axess[1].get_ylim()[1]])               
#             #     axess[0].set_xlim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             #     axess[0].set_ylim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             #     axess[1].set_xlim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             #     axess[1].set_ylim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])   
#             #     axess[0].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6) 
#             #     axess[1].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#             #     axess[0].plot(np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),
#             #                   np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),'--',color='gray',linewidth=1.5)
#             #     axess[1].plot(np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),
#             #                   np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),'--',color='gray',linewidth=1.5)
             
#             #     fig.tight_layout()
#             #     fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+nameStr+'_toneLevel-'+snr+'_Xregions.png')
#             #     plt.close(fig) 

#             # #  compare snr conditions, color different prior-pretone congruency in different colorsf
#             # fig, axess = plt.subplots(2,1,figsize=(4,8),sharex='col') #
#             # cat = sorted(NCdf_filtered[cmpIVs[0]].unique())
#             # # Load the tab10 colormap
#             # tab10 = plt.get_cmap('tab10')
#             # # Get the first 10 colors
#             # colrs = [tab10(i) for i in range(len(cat))]
#             # # colrs = ['lightpink','lightcoral','orangered','indianred','maroon'] 
#             # for ii,ppt in enumerate(cat):
#             #     NCdf_filtered_Monk1 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrCassius')&(NCdf_filtered[cmpIVs[0]]==ppt)].reset_index(drop=True)
#             #     NCdf_filtered_Monk2 = NCdf_filtered[(NCdf_filtered['Monkey']=='MrMiyagi')&(NCdf_filtered[cmpIVs[0]]==ppt)].reset_index(drop=True)

#             #     axess[0].scatter(NCdf_filtered_Monk1[NCdf_filtered_Monk1['toneLevel']=='easy'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                         NCdf_filtered_Monk1[NCdf_filtered_Monk1['toneLevel']=='difficult'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                     alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#             #     statsRes = stats.wilcoxon(NCdf_filtered_Monk1[NCdf_filtered_Monk1['toneLevel']=='easy'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                                 NCdf_filtered_Monk1[NCdf_filtered_Monk1['toneLevel']=='difficult'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#             #     axess[0].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)

#             #     axess[1].scatter(NCdf_filtered_Monk2[NCdf_filtered_Monk2['toneLevel']=='easy'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                         NCdf_filtered_Monk2[NCdf_filtered_Monk2['toneLevel']=='difficult'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                     alpha=0.3,c=colrs[ii],label=str(ppt),marker='o',edgecolors='none') 
#             #     statsRes = stats.wilcoxon(NCdf_filtered_Monk2[NCdf_filtered_Monk2['toneLevel']=='easy'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,
#             #                                 NCdf_filtered_Monk2[NCdf_filtered_Monk2['toneLevel']=='difficult'].sort_values('NeuPairs',kind='mergesort')['corrcoef'].values,nan_policy='omit')
#             #     axess[1].text(0.7,-0.55-ii*0.09,str(ppt)+' p='+str(np.round(statsRes[1],decimals=4)),horizontalalignment='right',verticalalignment='center',fontsize=10)
#             # axisLoBound = min([axess[0].get_xlim()[0],axess[0].get_ylim()[0],axess[1].get_xlim()[0],axess[1].get_ylim()[0]])
#             # axisHiBound = max([axess[0].get_xlim()[1],axess[0].get_ylim()[1],axess[1].get_xlim()[1],axess[1].get_ylim()[1]])               
#             # axess[0].set_xlim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             # axess[0].set_ylim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             # axess[1].set_xlim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])
#             # axess[1].set_ylim([-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)])])   
#             # axess[0].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6) 
#             # axess[1].legend(frameon=False,loc='upper left',fontsize=fontsizeNo-6)   
#             # axess[0].plot(np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),
#             #                 np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),'--',color='gray',linewidth=1.5)
#             # axess[1].plot(np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),
#             #                 np.linspace(-max([np.abs(axisLoBound),np.abs(axisHiBound)]),max([np.abs(axisLoBound),np.abs(axisHiBound)]),100),'--',color='gray',linewidth=1.5)
#             # axess[1].set_xlabel('easy',fontsize=fontsizeNo)
#             # axess[1].set_ylabel('difficult',fontsize=fontsizeNo)            
#             # fig.tight_layout()
#             # fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+nameStr+'_CMPtoneLevel_Xregions.png')
#             # plt.close(fig) 


#             times2 = time.monotonic()
#             print('total time spend for 1 time:')
#             # print(timedelta(seconds= times2- start_time)) 
    
    
    
###################################plot with pretone vs nopretone conditions
        NCdfpretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-withpretone.pkl','rb'))
        NCdfnopretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-nopretone.pkl','rb'))

        # NCdfpretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-withpretone_afterJS.pkl','rb'))
        # NCdfnopretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-nopretone_afterJS.pkl','rb'))

        # NCdfpretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-withpretone_Xregions.pkl','rb'))
        # NCdfnopretone = pickle.load(open(ResPathway+'NC_all_df_align2'+alignkeys+'_crect-soundOnTri-nopretone_Xregions.pkl','rb'))

        # plot sync vs opposite
        NCdfpretone_part = NCdfpretone[NCdfpretone['prior-pretone-Targ'].isin([-2,3])]
        NCdfnopretone_part = NCdfnopretone[NCdfnopretone['prior-Targ'].isin([-1,1])]
        NCdfpretone_part['sync'] = NCdfpretone_part['prior-pretone-Targ'].replace({-2:'oppo',3:'cong'})
        NCdfnopretone_part['sync'] = NCdfnopretone_part['prior-Targ'].replace({-1:'oppo',1:'cong'})
        NCdfpretone_part_sorted = NCdfpretone_part.sort_values(by=['NeuPairs','toneLevel','sync']).copy()
        NCdfnopretone_part_sorted = NCdfnopretone_part.sort_values(by=['NeuPairs','toneLevel','sync']).copy()

        if (NCdfpretone_part_sorted.NeuPairs.values == NCdfnopretone_part_sorted.NeuPairs.values).all() and\
        (NCdfpretone_part_sorted.toneLevel.values == NCdfnopretone_part_sorted.toneLevel.values).all():
            NCdf_comb = NCdfpretone_part_sorted[['NeuPairs','toneLevel','sync','corrcoef','sess','Region','Monkey']]
            NCdf_comb['nopretone'] = NCdfnopretone_part_sorted.corrcoef.values
            NCdf_comb.rename(columns={'corrcoef':'withpretone'},inplace=True)
        ####################within region NC
        for reg in ['AC','PFC']:      
            NCdf_comb_temp = NCdf_comb[NCdf_comb['Region']==reg]
            #  compare snr conditions, color different prior-pretone congruency in different colorsf
            fig, axess = plt.subplots(2,2,figsize=(8,8),sharex='col') #
            cat = NCdf_comb_temp.sync.unique()
            # Load the tab10 colormap
            tab10 = plt.get_cmap('tab10')
            # Get the first 10 colors
            colrs = [tab10(i) for i in range(len(cat))]

            plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrCassius')&(NCdf_comb_temp['toneLevel']=='easy')],axess[0,0],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrC-easy-'+reg)
            plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrCassius')&(NCdf_comb_temp['toneLevel']=='difficult')],axess[0,1],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrC-diff-'+reg)
            plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrMiyagi')&(NCdf_comb_temp['toneLevel']=='easy')],axess[1,0],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrM-easy-'+reg)
            plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrMiyagi')&(NCdf_comb_temp['toneLevel']=='difficult')],axess[1,1],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrM-diff-'+reg)
            
            fig.tight_layout()
            fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+'_CMPwithOno-pretone_'+reg+'.png')
            plt.close(fig) 

        # ####################cross region NC             
        # NCdf_comb_temp = NCdf_comb.copy()
        # #  compare snr conditions, color different prior-pretone congruency in different colorsf
        # fig, axess = plt.subplots(2,2,figsize=(8,8),sharex='col') #
        # cat = NCdf_comb_temp.sync.unique()
        # # Load the tab10 colormap
        # tab10 = plt.get_cmap('tab10')
        # # Get the first 10 colors
        # colrs = [tab10(i) for i in range(len(cat))]

        # plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrCassius')&(NCdf_comb_temp['toneLevel']=='easy')],axess[0,0],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrC-easy-Xregions')
        # plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrCassius')&(NCdf_comb_temp['toneLevel']=='difficult')],axess[0,1],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrC-diff-Xregions')
        # plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrMiyagi')&(NCdf_comb_temp['toneLevel']=='easy')],axess[1,0],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrM-easy-Xregions')
        # plotfitcorr(NCdf_comb_temp[(NCdf_comb_temp['Monkey']=='MrMiyagi')&(NCdf_comb_temp['toneLevel']=='difficult')],axess[1,1],'sync', 'nopretone', 'withpretone', colrs, ['NeuPairs','sync'],'MrM-diff-Xregions')
        
        # fig.tight_layout()
        # fig.savefig(figSavePath+'NC_byPriorPretoneTargSync_align2'+alignkeys+'_CMPwithOno-pretone_Xregions.png')
        # plt.close(fig) 
       
  
    print('done')

