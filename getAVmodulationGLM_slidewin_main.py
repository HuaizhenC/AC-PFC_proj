import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import os
import pickle
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.ticker import FuncFormatter
from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,loadPreprocessMat,glmfit,SortfilterDF,addIVcol,addIVcol2,sampBalanceGLM

# MonkeyDate_all = {'Elay':['230420']} #

# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/glmfit/'

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
                        '230602','230606','230608','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810','230814','230818','230822','230829',
                        '230906','230908','230915','230919','230922','230927',
                         '231003','231004','231010'], 
                  'Wu':['230809','230815','230821','230830',
                        '230905','230911','230913','230918','230925',
                          '231002','231006','231009']}

Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/Vprobe+EEG/'
ResPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'
figSavePath = '/data/by-user/Huaizhen/Figures/glmfit/'

winlen = 0.3
aligncond = [['cooOnsetIndwithVirtual','CooOnset',np.arange(-0.9,0.9,winlen)]]
fontsizeNo = 12

modsettings = [[{'trialMod':['a','av'],'respLabel':[0,1]},
                ['ali2co_2mod2lab_2IVs_hitmiss','ali2co_2mod2lab_3IVs_hitmiss'],
                [['snr-shift','V'],['snr-shift','V','AV-snr']],
                '_2mod2lab']]

# aligncond = [['JSOnsetIndwithVirtual','jsOn',np.arange(-1.5,0.6,winlen)]]
# modsettings = [[{'trialMod':['a','av'],'respLabel':[0]},
#                 ['ali2co_2mod1lab_2IVs_hit','ali2co_2mod1lab_3IVs_hit'],
#                 [['snr-shift','V'],['snr-shift','V','AV-snr']],
#                 '_2mod1lab']]

for aligncond_temp in aligncond:
    alignkeys = aligncond_temp[0]
    x0str = aligncond_temp[1]
    timwinStart = aligncond_temp[2] # 
    for modsetting_temp in modsettings:
        filter_dict = modsetting_temp[0]
        extrastr_all = modsetting_temp[1]
        GLM_IV_all = modsetting_temp[2]
        fitdfnamestr = modsetting_temp[3]

        glmfitres_all = pd.DataFrame()  
        for Monkey,Date in MonkeyDate_all.items():    
            for Date_temp in Date: 
                start_time = time.monotonic()
                print('...............'+'glm fit for '+Monkey+'-'+Date_temp+'...............')
                spikeTimeDict,labelDictFilter, \
                    timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)

                # save sessions have clusters
                for cls in list(spikeTimeDict.keys()):
                    if any(substr in cls for substr in ['good','mua']):   
                        print('cluster '+cls+' in progress............')
                        # # get trial by trial raster in each cluster
                        # labelDict_sub = {}
                        # ntrials = 200
                        # for key,value in labelDictFilter.items():
                        #     labelDict_sub[key] = value[-ntrials:]
                        # labelDictFilter = labelDict_sub.copy()
                        # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                            
                        spikeTime_temp = spikeTimeDict[cls]

                        spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                            labelDictFilter['chorusOnsetInd'],\
                                                            labelDictFilter[alignkeys] ,\
                                                            labelDictFilter['JSOnsetIndwithVirtual'],\
                                                            [timwinStart[0],timwinStart[-1]+winlen],\
                                                            labelDictFilter)  
                        spikeTimedf_temp_baseline = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                            labelDictFilter['chorusOnsetInd'],\
                                                            labelDictFilter['chorusOnsetInd'] ,\
                                                            labelDictFilter['JSOnsetIndwithVirtual'],\
                                                            [-winlen-0.05,-0.05],\
                                                            labelDictFilter)              
                        # filter trials
                        spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = filter_dict)
                        spikeTimedf_temp_baseline_filtered,_ = SortfilterDF(spikeTimedf_temp_baseline,filterlable = filter_dict)
                        # add IV columns is sigdf
                        spikeTimedf_temp_filtered =  addIVcol(spikeTimedf_temp_filtered)# A=snr-shift V=[-1,1],AV=A*V
                        # estimate baseline spknum
                        spikeNumdf_temp_baseline_raw=countTrialSPKs(spikeTimedf_temp_baseline_filtered,estwin='subwinoff',fs = behavefs,winTim=[-winlen-0.05,-0.05])                                                

                        for mm,(extrastr,GLM_IV_list) in enumerate(zip(extrastr_all,GLM_IV_all)):                    
                            IVnum = len(GLM_IV_list)
                            for tt, tStart in enumerate(timwinStart):
                                spikeNumdf_temp_raw=countTrialSPKs(spikeTimedf_temp_filtered,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])                                                
                                # baseline correction spknum, mitigate drift effect, 
                                # spknum can be negative, hence use gaussian distribution instead of poisson distributio to fit corrected spknum
                                spikeNumdf_temp_raw['spknum'] = spikeNumdf_temp_raw['spknum']-spikeNumdf_temp_baseline_raw['spknum']
                                
                                # balance samples within groups, main effect group need to be strictly balanced
                                spikeNumdf_temp = sampBalanceGLM(spikeNumdf_temp_raw,GLM_IV_list)
                                try :
                                    coeff_temp,pval_temp,evalparam = glmfit(spikeNumdf_temp,['spknum'],GLM_IV_list,'gaussian') #familystr=='poisson' or 'gaussian'
                                except RuntimeWarning:
                                    print('fail to fit glm model in'+Monkey+Date_temp+' unit:'+cls)
                                    coeff_temp = pd.DataFrame([[np.nan]*IVnum],columns=['coef_'+iv for iv in GLM_IV_list])
                                    pval_temp = pd.DataFrame([[np.nan]*IVnum],columns=['pval_'+iv for iv in GLM_IV_list])
                                    evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])
                                glmfitres_temp = pd.DataFrame.from_dict({'Monkey':[Monkey]*IVnum,'session_cls':[Date_temp+'_'+cls]*IVnum,\
                                                                        'time':[np.around(tStart,decimals=2)]*IVnum,'iv':GLM_IV_list,\
                                                                        'slope':[coeff_temp['coef_'+iv].values[0] for iv in GLM_IV_list],\
                                                                        'pval':[pval_temp['pval_'+iv].values[0] for iv in GLM_IV_list],\
                                                                        'aic':[evalparam.aic.values[0]]*IVnum,'bic':[evalparam.bic.values[0]]*IVnum,\
                                                                        'GLMmod':[str(GLM_IV_list)]*IVnum})                        
                                glmfitres_all = pd.concat([glmfitres_all,glmfitres_temp])
                        # print('done with cluster '+cls)
                end_time = time.monotonic()
                print('fit time for session '+Date_temp)
                print(timedelta(seconds=end_time - start_time))
        pickle.dump(glmfitres_all,open(ResPathway+'glmfitCoefDF_'+alignkeys+fitdfnamestr+'.pkl','wb'))  
        print(glmfitres_all.to_string())                     
                        
        model_comp_param = pd.DataFrame()  
        glmfitres_all = pickle.load(open(ResPathway+'glmfitCoefDF_'+alignkeys+fitdfnamestr+'.pkl','rb')).reset_index()


        # filter out drifted neurons
        dfinspect = pd.read_excel('AllClusters4inspectionSheet.xlsx')
        driftedUnites = list(dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)']==1]['session_cls'].values)
        glmfitres_all = glmfitres_all[~glmfitres_all['session_cls'].isin(driftedUnites)]
        
        for mm,(extrastr,GLM_IV_list) in enumerate(zip(extrastr_all,GLM_IV_all)): 
            glmfitres = glmfitres_all[glmfitres_all['GLMmod']== str(GLM_IV_list)]
            model_comp_param_temp = glmfitres.groupby(['Monkey','session_cls','time'])[['aic','bic']].mean().reset_index()
            model_comp_param_temp['model'] = str(GLM_IV_list)+str(mm)
            model_comp_param = pd.concat((model_comp_param,model_comp_param_temp))
            
            glmfitres_sig = glmfitres[glmfitres['pval']<0.05].copy()
            glmfitres_sig['iv'] = glmfitres_sig['iv'].replace({'snr-shift':'SNR'})

            ## plot sig slopes
            def plotsigSlp(axrow,monkey,x0str,axess):
                # Define custom formatter function
                def scale_formatter(x, pos):
                    return f'{100*x/totalNumCls:.0f}'
                def add0Str2xtick(xt,x0str):            
                    xt = np.delete(xt,np.where(xt==0)[0])
                    xt = np.append(xt,0)
                    xtl = xt.tolist()
                    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
                    xtl[-1] = x0str
                    return xt,xtl
                totalNumCls = glmfitres[glmfitres['Monkey']==monkey].groupby(['session_cls']).size().shape[0]

                sns.stripplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',y='slope',hue='iv',hue_order=sorted(glmfitres_sig['iv'].unique()),ax=axess[axrow,0],dodge=True,size=1.5)
                xt = axess[1,0].get_xticks() 
                axess[1,0].set_xticks(xt)
                xticklabels = axess[1,0].get_xticklabels()
                formatted_xticklabels = [f'{float(tick_val._text):.1f}' for tick_val in xticklabels]
                axess[1,0].set_xticklabels(formatted_xticklabels)
                legend = axess[axrow,0].legend(frameon=False, framealpha=0,fontsize=fontsizeNo)
                legend.set_title('monkey '+monkey[0])
                legend.get_title().set_fontsize(fontsizeNo)
                axess[axrow,0].set_ylabel('Predictor Variable Weights',fontsize=fontsizeNo)

                sns.lineplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',y='slope',hue='iv',hue_order=sorted(glmfitres_sig['iv'].unique()),style='iv',ax=axess[axrow,1],estimator='mean',markers=True,dashes=False,errorbar=('ci',95))
                xt = list(np.arange(-0.9,0.9,0.3)) 
                # xt,xtl = add0Str2xtick(xt,x0str)
                axess[axrow,1].set_xticks(xt)
                # axess[axrow,1].set_xticklabels(xtl)
                legend=axess[axrow,1].legend(frameon=False, framealpha=0,fontsize=fontsizeNo,loc='upper left')
                legend.set_title('monkey '+monkey[0])
                legend.get_title().set_fontsize(fontsizeNo)                
                axess[axrow,1].set_ylabel('Predictor Variable Weights',fontsize=fontsizeNo)
                axess[axrow,1].set_xlabel('Time (s)',fontsize=fontsizeNo)
                yt = axess[axrow,1].get_yticks() 
                axess[axrow,1].set_yticks([np.round(yt.min(),decimals=1),0,np.round(yt.max(),decimals=1)])

                sns.countplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',hue='iv',hue_order=sorted(glmfitres_sig['iv'].unique()),ax=axess[axrow,2])
                axess[axrow,2].yaxis.set_major_formatter(FuncFormatter(scale_formatter))
                axess[axrow,2].set_ylabel('Percent of Units',fontsize=fontsizeNo)
                # axess[axrow,2].set_xticks(xt)
                legend = axess[axrow,2].legend(frameon=False, framealpha=0,fontsize=fontsizeNo,loc='upper left')
                legend.set_title('monkey '+monkey[0])
                legend.get_title().set_fontsize(fontsizeNo)                
                axess[axrow,2].set_ylabel('Percent of Units',fontsize=fontsizeNo)
                axess[axrow,2].set_xlabel('Time (s)',fontsize=fontsizeNo)
                yt = axess[axrow,2].get_yticks() 
                axess[axrow,2].set_yticks([0,np.round(yt.max(),decimals=0)])

            fig, axess = plt.subplots(2,3,figsize=(12,5),sharex='col') # 
            plotsigSlp(0,'Elay',x0str,axess)
            plotsigSlp(1,'Wu',x0str,axess)

            fig.tight_layout()
            fig.savefig(figSavePath+'glmfitSlopeovertime_'+extrastr+'_'+alignkeys+''+'.png')
            plt.close(fig)

        # model comparison measurements
        fig, axess = plt.subplots(2,2,figsize=(10,5),sharex='col') # 
        sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='aic',hue='time',ax=axess[0,0],dodge=True,size=1.5)
        sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='bic',hue='time',ax=axess[0,1],dodge=True,size=1.5)
        axess[0,0].legend(frameon=False, framealpha=0,fontsize=5)
        axess[0,1].legend(frameon=False, framealpha=0,fontsize=5)

        sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='aic',hue='time',ax=axess[1,0],dodge=True,size=1.5)
        sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='bic',hue='time',ax=axess[1,1],dodge=True,size=1.5)
        axess[1,0].legend(frameon=False, framealpha=0,fontsize=5)
        axess[1,0].tick_params(axis='x',rotation=45)
        axess[1,1].legend(frameon=False, framealpha=0,fontsize=5)
        axess[1,1].tick_params(axis='x',rotation=45)

        fig.tight_layout()
        fig.savefig(figSavePath+'glmfitModComp_'+alignkeys+fitdfnamestr+'.png')
        plt.close(fig)

print('done')


