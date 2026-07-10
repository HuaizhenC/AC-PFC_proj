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
from spikeUtilities import SortfilterDF, getPSTHTbyT,getglmfitovertime,addIVcol,extract_between_tts
from sharedparam import getMonkeyDate_all,neuronfilterDF
from scipy.stats import zscore
cpus = 20

## plot slopes
def add0Str2xtick(xt,x0str):            
    xt = [vv for vv in xt if vv!=0]
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl

def plotsigSlp(axrow,monkey,glmfitres_monk,ycolumn):                 
    sns.lineplot(glmfitres_monk,x='time',y=ycolumn,hue='iv',hue_order=sorted(glmfitres_monk['iv'].unique()),ax=axrow,estimator='mean',errorbar=('ci',95),palette='Set2')
    axrow.legend(frameon=False, framealpha=0,fontsize=fontsizeNo,loc='upper left',bbox_to_anchor=(1, 1), borderaxespad=0., )
    axrow.set_title('monkey '+monkey[2])
    axrow.set_ylabel('Predictor Variable Weights',fontsize=fontsizeNo)
    yt = axrow.get_yticks() 
    axrow.set_yticks([np.round(yt.min(),decimals=1),0,np.round(yt.max(),decimals=1)])


# MonkeyDate_all = {'MrCassius':['MrCassius-190414'],'MrMiyagi':['MrMiyagi-190417']} #
# figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/FigureOutput/glmfit/'
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/PSTH/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/glmfit/'
# excelpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/pythonAnalysis/'
# AVResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/Results/AVmodIndex/'
MonkeyDate_all = getMonkeyDate_all()
figsavPathway = '/home/huaizhen/Documents/LalittaProj/data/FigureOutput/glmfit/'
Pathway='/home/huaizhen/Documents/LalittaProj/data/Results/PSTH/'
ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/glmfit/' 
AVResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/AVmodIndex/' 
excelpathway = '/Users/caihuaizhen1991gmail.com/Documents/LalittaProj/data'
figformat = 'png'

fontsizeNo = 12


timwin_all = [[-800,700],[-1000,500]]
aligncond = [['targOn','TargOn'],['JoystickStartMove','JSOn']]# 
fitfilter_all = [
               {'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['None'],'correct':[1]},
               'GLM_IV':['prior','isHigh','prior-Targ','toneLevel'],
               'fitdfnamestr':'_crect-soundOnTri-nopretone-3IV1interaction'},  
               {'filter_dict':{'soundOnFlag':[1],'prior':['Low','High','Neutral'],'pretone':['High','Low'],'correct':[1]},
               'GLM_IV':['prior','pretone','isHigh','prior-Targ','pretone-Targ','prior-pretone-Targ','toneLevel'],
               'fitdfnamestr':'_crect-soundOnTri-withpretone-4IV3interactions'},                        
               ]

if __name__ == '__main__':
    # filter out drifted neurons
    dfinspect = pd.read_excel(excelpathway+'AllClusters4inspectionSheet.xlsx')
    nodriftedUnites = dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)'].isin([0])]

    # loop through different alignment conditions
    for aligncond_temp,timwin in zip(aligncond,timwin_all):
        alignkeys = aligncond_temp[0]
        x0str = aligncond_temp[1]

        # prepare for glmfit
        glmfitres_all = pd.DataFrame()  
        #start go through session-cluster 
        for Monkey,Date in MonkeyDate_all.items():    
            for Date_temp in Date:
                AllSUspk_df = pickle.load(open(Pathway+Date_temp+'_ACnPFC_allcls_align2'+x0str+'_overlaptimwin_binlen100ms_binstep50ms_PSTH_df.pkl','rb'))
                
                for area in ['AC','PFC']:
                    cccount = 0
                    AllSUspk_df_regiontemp = AllSUspk_df[AllSUspk_df['Region']==area]
                    nodriftedUnites_sub = nodriftedUnites[nodriftedUnites['Region']==area]
                    for keys in list(AllSUspk_df_regiontemp.cls.unique()):
                        if Date_temp[-6:]+'_'+keys in list(nodriftedUnites_sub.session_cls.values): # only non-drifted units
                            if any(substr in keys for substr in ['good','mua']):                            
                                print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
                                AllSUspk_df_region_cls_temp = AllSUspk_df_regiontemp[AllSUspk_df_regiontemp['cls']==keys] 

                                # loop through different fitting factors or data filters 
                                for filter_info in fitfilter_all:                                   
                                    filter_dict = filter_info['filter_dict']
                                    GLM_IV_list = filter_info['GLM_IV']
                                    fitdfnamestr = filter_info['fitdfnamestr']                                   
                                    
                                    psth_win_df_all,_ = SortfilterDF(AllSUspk_df_region_cls_temp,filterlable = filter_dict)  
                                    # add interaction IV columns and rank each factor as meaningful numbers for the following fit, noprior/nochoice trials are nans
                                    psth_win_df_all = addIVcol(psth_win_df_all) 
                                    # filter out timepnts want to use
                                    frcolall = [s for s in list(psth_win_df_all.columns) if '*' in s]
                                    frcol= [element for element in frcolall if (float(extract_between_tts(element)) >= timwin[0]) and (float(extract_between_tts(element)) <= timwin[1])]
                                    timpnts = sorted([float(extract_between_tts(element)) for element in frcol])
                                    psth_win_df = psth_win_df_all[frcol+list(set(psth_win_df_all.columns)^set(frcolall))]
                                    # print(frcol)

                                    glmfitres_all_temp = getglmfitovertime(psth_win_df,'poisson',GLM_IV_list,cpus,Monkey,Date_temp,keys,area,fitdfnamestr) # get glmfit of each IVs overtime for this neuron                                                 
                                    glmfitres_all = pd.concat([glmfitres_all,glmfitres_all_temp])
                                    # ensure each glm factor have expected values in the filtered spk dataframe:spikeTimedf_temp_filtered
                                    if cccount==0:
                                        print(filter_info)
                                        print(psth_win_df.groupby(GLM_IV_list).size().reset_index().to_string())
                                        # for column in GLM_IV_list:
                                        #     # Get the unique values of the column
                                        #     unique_values = sorted(psth_win_df[column].unique())
                                        #     print(f"Unique values in '{column}': {unique_values}")
                                cccount = cccount+1

        # print(glmfitres_all.to_string())
        end_time = time.monotonic()
        print('total fit time  ')
        print(timedelta(seconds=end_time - start_time))
        pickle.dump([timpnts,glmfitres_all],open(ResPathway+'glmfitCoefDF_align2'+alignkeys+'.pkl','wb'))  
                            
        model_comp_param = pd.DataFrame()  
        timpnts,glmfitres_all = pickle.load(open(ResPathway+'glmfitCoefDF_align2'+alignkeys+'.pkl','rb'))
        # print(glmfitres_all.to_string())
        # filter neurons based on ttest
        df_avMod_all = pickle.load(open(AVResPathway+'AVmodTTestDF.pkl','rb')).reset_index()
        df_avMod_all_sig = df_avMod_all[(df_avMod_all['pval']<0.0001)&(df_avMod_all['iv'].isin(['LED','A','RWD','CHOICE']))]
        cmpCond = 'CHOICE' #['LED','A','RWD','CHOICE','MOVE']
        df_avMod_all_sig[cmpCond+'modFlag'] = np.where(df_avMod_all_sig['iv'].isin([cmpCond]),0,1)
        # df_avMod_all_sig =  df_avMod_all.copy()

        # df_avMod_all_sig_iv = df_avMod_all_sig[df_avMod_all_sig['iv'].isin(['LED','A','RWD','CHOICE'])]#['LED','A','RWD','CHOICE','MOVE']

        # plot units modified by different IVs separatly
        # for iv_temp in df_avMod_all_sig.iv.unique():
        #     df_avMod_all_sig_iv = df_avMod_all_sig[df_avMod_all_sig['iv']==iv_temp]

        # for inex_temp in df_avMod_all_sig_iv['inhibition/excitation'].unique():
        #     df_avMod_all_sig_iv_temp = df_avMod_all_sig_iv[df_avMod_all_sig_iv['inhibition/excitation']==inex_temp]
        #     # namestrgroup = '_'+iv_temp+['_inhibition' if inex_temp<0 else '_excitation'][0]
        #     namestrgroup = '_all_'+['_inhibition' if inex_temp<0 else '_excitation'][0]

        for inex_temp in df_avMod_all_sig[cmpCond+'modFlag'].unique():
            df_avMod_all_sig_iv_temp = df_avMod_all_sig[df_avMod_all_sig[cmpCond+'modFlag']==inex_temp]
            # namestrgroup = '_'+iv_temp+['_inhibition' if inex_temp<0 else '_excitation'][0]
            namestrgroup = '_'+cmpCond+'modFlag'+str(inex_temp)

            print('...........................................'+namestrgroup)

            # loop through different fitting factors or data filters 
            for filter_info in fitfilter_all:
                filter_dict = filter_info['filter_dict']
                GLM_IV_list = filter_info['GLM_IV']
                fitdfnamestr = filter_info['fitdfnamestr']    
                for area in ['AC','PFC']:
                    print('plotting '+area+fitdfnamestr+'.....................................................')
                    glmfitres_all_sub = glmfitres_all[(glmfitres_all['Region']==area)&(glmfitres_all['fitdfnamestr']==fitdfnamestr)].reset_index(drop=True)
                    df_avMod_all_sub = df_avMod_all_sig_iv_temp[(df_avMod_all_sig_iv_temp['Region']==area)]
                    print('GLMmod    '+str(glmfitres_all_sub['GLMmod'].unique()))

                    # filter out neurons have nans at any time point
                    nanslopeCls = list(glmfitres_all_sub[glmfitres_all_sub['slope'].isna()]['session_cls'].unique())
                    print('there are '+str(len(nanslopeCls))+' clusters out of '+str(len(glmfitres_all_sub['session_cls'].unique()))+' in '+area+' have nan glmfit slope!')
                    glmfitres_filtered = glmfitres_all_sub[~glmfitres_all_sub['session_cls'].isin(nanslopeCls)]
                    # filter neurons based on selectivity in ttest 
                    glmfitres_filtered = glmfitres_filtered[glmfitres_filtered['session_cls'].isin(list(df_avMod_all_sub['session_cls'].values))]
                    print('Total num of useful neurons in '+area+':'+str(len(glmfitres_filtered['session_cls'].unique())))
                    
                    glmfitres_filtered_sig = glmfitres_filtered[glmfitres_filtered['pval']<0.05].copy() 
                    # glmfitres_sig = glmfitres_filtered.copy()
                    
                    ycolumn = 'slope_zscore' #'slope_zscore'  'slope' 'slope_minmax'
                    glmfitres_filtered_sig['slope_zscore'] = glmfitres_filtered_sig.groupby('iv')['slope'].transform(lambda x: zscore(x, ddof=1))
                    glmfitres_filtered_sig['slope_minmax'] = glmfitres_filtered_sig.groupby('iv')['slope'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
                    glmfitres_filtered_sig['abs(slope)'] = glmfitres_filtered_sig[ycolumn].abs()
                    # print(glmfitres_filtered_sig[glmfitres_filtered_sig['pval']<0.0000000001].sort_values(by=['pval','time','session_cls','abs(slope)'],ascending=[True,True,True,False])[['Monkey','session_cls','Region','iv','time',ycolumn,'pval','fitdfnamestr']].reset_index(drop=True).to_string())                           
                    
                    fig, axess = plt.subplots(2,1,figsize=(10,10),sharex='col') #                 
                    # print('df2print')
                    if glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrCassius'].shape[0]>0:
                        plotsigSlp(axess[0],'MrCassius',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrCassius'].reset_index(drop=True),ycolumn)
                    if glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrMiyagi'].shape[0]>0:
                        plotsigSlp(axess[1],'MrMiyagi',glmfitres_filtered_sig[glmfitres_filtered_sig['Monkey']=='MrMiyagi'].reset_index(drop=True),ycolumn)
                    xt = list(np.arange(timpnts[0],timpnts[-1],200)) 
                    xt,xtl = add0Str2xtick(xt,x0str)
                    axess[1].set_xticks(xt)
                    axess[1].set_xticklabels(xtl)            
                    axess[1].set_xlabel('Time (ms)',fontsize=fontsizeNo)

                    fig.tight_layout()
                    fig.savefig(figsavPathway+'glmfitSlopeovertime_align2'+x0str+'_'+area+fitdfnamestr+namestrgroup+'.png')
                    plt.close(fig)

            # model_comp_param_temp = glmfitres.groupby(['Monkey','session_cls','time'])[['aic','bic']].mean().reset_index()
            # model_comp_param_temp['model'] = str(GLM_IV_list)+str(mm)
            # model_comp_param = pd.concat((model_comp_param,model_comp_param_temp))
            # # model comparison measurements
            # fig, axess = plt.subplots(2,2,figsize=(10,5),sharex='col') # 
            # sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='aic',hue='time',ax=axess[0,0],dodge=True,size=1.5)
            # sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='bic',hue='time',ax=axess[0,1],dodge=True,size=1.5)
            # axess[0,0].legend(frameon=False, framealpha=0,fontsize=5)
            # axess[0,1].legend(frameon=False, framealpha=0,fontsize=5)

            # sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='aic',hue='time',ax=axess[1,0],dodge=True,size=1.5)
            # sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='bic',hue='time',ax=axess[1,1],dodge=True,size=1.5)
            # axess[1,0].legend(frameon=False, framealpha=0,fontsize=5)
            # axess[1,0].tick_params(axis='x',rotation=45)
            # axess[1,1].legend(frameon=False, framealpha=0,fontsize=5)
            # axess[1,1].tick_params(axis='x',rotation=45)

            # fig.tight_layout()
            # fig.savefig(figSavePath+'glmfitModComp_'+alignkeys+fitdfnamestr+'.png')
            # plt.close(fig)

    end_time = time.monotonic()
    print('total fit time:  ')
    print(timedelta(seconds=end_time - start_time))
    print('done')


