import numpy as np
import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt 
import matplotlib as mpl
from spikeUtilities import getClsRasterMultiprocess,SortfilterDF,loadPreprocessMat,loadBehavMat, getPSTH
from sharedparam import getMonkeyDate_all,neuronfilterDF
from datetime import datetime
cpus =10
# plot trial by trial raster in each condition of interest
def generate_color_gradient(start_color, end_color, num_colors):
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", [start_color, end_color])
    gradient = [mpl.colors.rgb2hex(cmap(i / (num_colors - 1))) for i in range(num_colors)]
    return gradient #generate_color_gradient('lightpink', 'maroon', len(subgroup_Tartone_df_tep['toneLevel'].unique()))

def plotraster2(spikeTimedf_temp,timwin,bintimeLen,axes,fig,extraStr):
    # plot raster
    plt.text(0.3, 0.99, 'pretone', ha='left', va='top', transform=fig.transFigure,weight='bold',fontsize=18)
    plt.text(0.05, 0.5, 'prior', ha='left', va='center', rotation=90, transform=fig.transFigure,weight='bold',fontsize=18)
    yaxmax = []
    allax2 = []
    for cc,labprior in enumerate(['High', 'Low', 'Neutral']):#enumerate(sorted(spikeTimedf_temp.prior.unique())): 
        group_prior_df = spikeTimedf_temp[spikeTimedf_temp['prior']==labprior]
        for gg,labpretone in enumerate(['High', 'Low', 'None']):#enumerate(sorted(spikeTimedf_temp.pretone.unique())):
            subgroup_pretone_df = group_prior_df[group_prior_df['pretone']==labpretone]
            trialcum = 0
            subgroup_pretone_df_sorted = pd.DataFrame()
            ax2 = axes[cc,gg].twinx()
            allax2.append(ax2)
 
            # if labpretone!='None': # plot no pretone condition differently 
            #     # Generate a color map
            #     categories =['preSame','preDiff']
            #     colors = ['red','blue']   
            #     color_dict = dict(zip(categories, colors))
            #     for ii,labTartone in enumerate(sorted(subgroup_pretone_df.isHigh.unique())):
            #         print([labprior,labpretone,labTartone])
            #         subgroup_Tartone_df_tep = subgroup_pretone_df[subgroup_pretone_df['isHigh']==labTartone]
            #         subgroup_Tartone_df_tep = subgroup_Tartone_df_tep.sort_values(by=['toneLevel','trialID'],ascending=True)           
            #         subgroup_Tartone_df_tep['trialNumGlob'] =pd.factorize(subgroup_Tartone_df_tep['trialID'])[0]+trialcum               
            #         PTcat =['preSame' if labpretone==labTartone else 'preDiff'][0]
            #         subgroup_Tartone_df_tep['preToneCat'] = PTcat
            #         trialcum = len(subgroup_Tartone_df_tep.trialID.unique())+ trialcum
            #         timpnts = np.arange(timwin[0],timwin[1],1)
            #         psth,_ = getPSTH(list(subgroup_Tartone_df_tep.spktim.values),timpnts,bintimeLen,len(subgroup_Tartone_df_tep['trialID'].unique()),len(subgroup_pretone_df['trialID'].unique()))                    
            #         ax2.plot(timpnts,psth,color=color_dict[PTcat],linewidth=1,label=PTcat)
            #         yaxmax.append(np.max(psth))
            #         subgroup_pretone_df_sorted = pd.concat([subgroup_pretone_df_sorted,subgroup_Tartone_df_tep],axis=0)
            #     sns.scatterplot(subgroup_pretone_df_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes[cc,gg],
            #                     hue='toneLevel',hue_order=tone_levels,size='preToneCat',sizes={'preSame':5,'preDiff':0},palette=palette_R)    
            #     sns.scatterplot(subgroup_pretone_df_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes[cc,gg],
            #                     hue='toneLevel',hue_order=tone_levels,size='preToneCat',sizes={'preSame':0,'preDiff':5},palette=palette_B)             
            #     legend_elements = [plt.Line2D([0], [0], color=vv, lw=2,linestyle=':', label=str(kk)) for ii,(kk,vv) in enumerate(color_dict.items())] 
            #     legend=axes[cc,gg].legend(handles=legend_elements, frameon=False, framealpha=0,loc='upper left',bbox_to_anchor=(0, 1), borderaxespad=0., fontsize=8) 
            # if labpretone=='None':
            categories =['High','Low'] #PSTH color
            colors = ['red','blue']   
            color_dict = dict(zip(categories, colors))
            for ii,labTartone in enumerate(categories):
                print([labprior,labpretone,labTartone])
                subgroup_Tartone_df_tep = subgroup_pretone_df[subgroup_pretone_df['isHigh']==labTartone]
                subgroup_Tartone_df_tep = subgroup_Tartone_df_tep.sort_values(by=['toneLevel','trialID'],ascending=True)           
                subgroup_Tartone_df_tep['trialNumGlob'] =pd.factorize(subgroup_Tartone_df_tep['trialID'])[0]+trialcum                                  
                trialcum = len(subgroup_Tartone_df_tep.trialID.unique())+ trialcum
                timpnts = np.arange(timwin[0],timwin[1],1)
                if len(subgroup_Tartone_df_tep['trialID'].unique())>=1:
                    psth,_ = getPSTH(list(subgroup_Tartone_df_tep.spktim.values),timpnts,bintimeLen,len(subgroup_Tartone_df_tep['trialID'].unique()),len(subgroup_pretone_df['trialID'].unique()))
                    ax2.plot(timpnts,psth,color=color_dict[labTartone],linewidth=2,label=labTartone)
                    yaxmax.append(np.max(psth))
                subgroup_pretone_df_sorted = pd.concat([subgroup_pretone_df_sorted,subgroup_Tartone_df_tep],axis=0)
            sns.scatterplot(subgroup_pretone_df_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes[cc,gg],
                            hue='toneLevel',hue_order=tone_levels,size='isHigh',sizes={'High':4,'Low':0},palette=palette_R)    
            sns.scatterplot(subgroup_pretone_df_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes[cc,gg],
                            hue='toneLevel',hue_order=tone_levels,size='isHigh',sizes={'High':0,'Low':4},palette=palette_B)             
            axes[cc,gg].yaxis.set_ticks([0,subgroup_pretone_df_sorted['trialNumGlob'].values.max()])           
            axes[cc,gg].set_xlim([timwin[0],timwin[1]]) 
            axes[cc,gg].tick_params(axis='both', which='major', labelsize=8)
            # set legend   
            if cc==0 and gg==0:
                legend_elements = [plt.Line2D([0], [0], color=vv, lw=2,linestyle=':', label=str(kk)) for ii,(kk,vv) in enumerate(color_dict.items())] 
                legend=axes[cc,gg].legend(handles=legend_elements, frameon=False, framealpha=0,loc='upper left',bbox_to_anchor=(-0.5, 1), borderaxespad=0., fontsize=8,prop={'weight': 'bold'}) 
            else:
                axes[cc,gg].legend([], frameon=False, framealpha=0) 
            #set labels
            if cc==0:
                axes[cc,gg].set_title(labpretone,fontsize=14,weight='bold')
            if gg==0:
                axes[cc,gg].set_ylabel(labprior,fontsize=14,weight='bold')
            else:
                axes[cc,gg].set_ylabel('',fontsize=14,weight='bold')  
            if cc<2 and gg<3:
                axes[cc,gg].set_xticklabels([])
            if cc<2:
                axes[cc,gg].set_xlabel('')
    # set all ax2 ylim the same range
    for allax2_temp in allax2:
        ymaxval = np.round(np.max(yaxmax),decimals=1)
        allax2_temp.set_ylim([0,ymaxval+0.5])
        allax2_temp.yaxis.set_ticks([0,np.round(ymaxval/2,decimals=1),ymaxval+0.5]) 
        allax2_temp.tick_params(axis='both', which='major', labelsize=8)

def plotraster(spikeTimedf_temp,compcol,timwin,bintimeLen,axes,colors,categories,legendshiftStr=[0,0.5]):
    ax2 = axes.twinx()
    yaxmax = []
    # Generate a color map
    color_dict = dict(zip(categories, colors))    
    trialcum = 0
    spikeTimedf_temp_sorted = pd.DataFrame()
    for cc,lab in enumerate(categories): 
        subgroup_df = spikeTimedf_temp[spikeTimedf_temp[compcol]==lab]
        subgroup_df = subgroup_df.sort_values(by=['trialID'],ascending=True)
        subgroup_df['trialNumGlob'] =pd.factorize(subgroup_df['trialID'])[0]+trialcum               
        timpnts = np.arange(timwin[0],timwin[1],1)
        if len(subgroup_df['trialID'].unique())>=1:
            psth,_ = getPSTH(list(subgroup_df.spktim.values),timpnts,bintimeLen,len(subgroup_df['trialID'].unique()),len(spikeTimedf_temp['trialID'].unique()))        
            ax2.plot(timpnts,psth,color=color_dict[lab],linewidth=1,label=[lab if isinstance(lab,str) else str(lab)])
            yaxmax.append(np.max(psth))
        trialcum = len(subgroup_df.trialID.unique())+ trialcum 
        spikeTimedf_temp_sorted = pd.concat([spikeTimedf_temp_sorted,subgroup_df],axis=0)
    if len(categories)==2:
        sns.scatterplot(spikeTimedf_temp_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes,s=legendshiftStr[1],hue=compcol,hue_order=categories,palette=color_dict)
    else:
        sns.scatterplot(spikeTimedf_temp_sorted.reset_index(drop=True),x='spktim',y='trialNumGlob',ax=axes,s=legendshiftStr[1],hue=compcol,hue_order=categories,palette=palette_G)

    legend_elements = [plt.Line2D([0], [0], color=vv, lw=3,linestyle=':', label=str(kk)) for ii,(kk,vv) in enumerate(color_dict.items())] 
    if legendshiftStr[0]=='off':
        axes.legend([], frameon=False, framealpha=0) 
    else:
        legend=axes.legend(handles=legend_elements, frameon=False, framealpha=0,loc='upper left',bbox_to_anchor=(legendshiftStr[0], 1), borderaxespad=0., fontsize=8) 
        # legend.set_title(compcol)

    axes.set_title(compcol,fontsize=10) 
    axes.set_xlim([timwin[0],timwin[1]])  
    axes.yaxis.set_ticks([0,spikeTimedf_temp_sorted['trialNumGlob'].values.max()])
    axes.set_ylabel('',fontsize=14,weight='bold')  
    axes.tick_params(axis='both', which='major', labelsize=8)
    ymaxVal = np.round(np.max(yaxmax),decimals=1)
    ax2.set_ylim([0,ymaxVal+0.5])
    ax2.yaxis.set_ticks([0,ymaxVal+0.5])  
    ax2.tick_params(axis='both', which='major', labelsize=8)
    return  spikeTimedf_temp_sorted,[ax2,np.max(yaxmax)]

    

# MonkeyDate_all = {'MrCassius':['MrCassius-190727']} #
# figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/FigureOutput/Raster/'
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/LalittaProj/data/preprocNeuralMatfiles/'
# ResPathway = os.getcwd()

MonkeyDate_all = getMonkeyDate_all()
figsavPathway = '/home/huaizhen/Documents/LalittaProj/data/FigureOutput/Raster_KS3'
Pathway='/home/huaizhen/Documents/LalittaProj/data/preprocNeuralMatfiles_KS3/'
ResPathway = '/home/huaizhen/Documents/LalittaProj/data/Results/Raster/' 

figformat = 'png'

# # concatenate along the manual inspected excelsheet if it is in the current directory
# if os.path.exists(os.getcwd()+'/AllClusters4inspectionSheet.xlsx'):
#     manualInspectDF = pd.read_excel(os.getcwd()+'/AllClusters4inspectionSheet.xlsx',engine='openpyxl')
# else:
#     manualInspectDF = pd.DataFrame()
#     print('Did not find AllClusters4inspectionSheet.xlsx file in the current directory!!')

# generate gradient color palette for tonelevel
tone_levels = [0,75,90,100,110]
color_map_B = plt.cm.Blues(np.linspace(0.4,0.8, len(tone_levels))) # light to Dark blue
color_map_R = plt.cm.Reds(np.linspace(0.4,0.8, len(tone_levels)))   # light to Dark red
color_map_G = plt.cm.gist_yarg(np.linspace(0.2,0.6, len(tone_levels)))   # light to Dark gray

# color_map_B = plt.cm.winter(np.linspace(0.05,1, len(tone_levels))) # light to Dark blue
# color_map_R = plt.cm.autumn(np.linspace(0.05,0.8, len(tone_levels)))   # light to Dark red
# Creating dictionary mapping toneLevel to colors for each palette
palette_B = dict(zip(tone_levels, color_map_B)) #low
palette_R = dict(zip(tone_levels, color_map_R)) #high
palette_G = dict(zip(tone_levels, color_map_G)) #high

manualInspectDF = pd.DataFrame()
if __name__ == '__main__':
    #start go through session-cluster 
    for Monkey,Date in MonkeyDate_all.items():    
        for Date_temp in Date:
            outputPathway = os.path.join(figsavPathway,Date_temp)
            try:
                os.mkdir(outputPathway)
            except FileExistsError:
                pass

            # load behavioral data
            labeltimcombDF = loadBehavMat(Date_temp,Pathway)

            for area in ['AC','PFC']:
                # load spike times 
                spikeTimeDict,timeSamp2stimOn,spikefs= loadPreprocessMat(Date_temp,Pathway,area)           
                # get trial by trial raster in each cluster, align2stimOn
                cccount = 0
                for keys in list(spikeTimeDict.keys()):
                    if 'cls' in keys:
                        # print(spikeTimeDict[keys])
                        spktotal_temp = [item for sublist in spikeTimeDict[keys] for item in (sublist[0] if isinstance(sublist[0],np.ndarray) and sublist[0].ndim > 0 else [sublist[0]])]
                        fr_ave = np.count_nonzero(~np.isnan(spktotal_temp))/(len(spikeTimeDict[keys])*20) # estimate average firing rate of this cluster
                        if any(substr in keys for substr in ['good','mua']) and fr_ave>1:
                            print('..................'+Date_temp+' '+area+' '+keys+' in progress............')
                           
                            # # # # get trial by trial raster in each cluster
                            # # sttrials = 600
                            # # ntrials = 1500               
                            # # labeltimcombDF = labeltimcombDF.iloc[sttrials:(sttrials+ntrials),:]
                            # # spikeTime_temp = spikeTimeDict[keys][sttrials:(sttrials+ntrials)]
                                                
                            # spikeTime_temp = spikeTimeDict[keys]

                            # fig, axess = plt.subplots(3,5,figsize=(24,10),gridspec_kw={'width_ratios': [0.23,0.23,0.23,0.13,0.13]})
                            # # extrastr = 'targOn'  
                            # # timwin = [-1000,900] #ms   
                            # extrastr = 'soundOn'  
                            # timwin = [-400,1400] #ms                       
                            # spikeTimedf_temp = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                            #                                     'soundOn',extrastr,
                            #                                     timwin,labeltimcombDF,cpus)                            
                            # spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = {'soundOnFlag':[1],'correct':[1]})
                            # plotraster2(spikeTimedf_temp_filtered,timwin,50,axess,fig,extrastr) #binlength in ms
                            # _,ax2Hi = plotraster(spikeTimedf_temp_filtered[(spikeTimedf_temp_filtered['isHigh']=='High')&(spikeTimedf_temp_filtered['pretone'].isin(['High','Low']))],
                            #                    'toneLevel',timwin,50,axess[2,3],color_map_R,tone_levels,[-6.8,2]) #binlength in ms
                            # _,ax2Lo = plotraster(spikeTimedf_temp_filtered[(spikeTimedf_temp_filtered['isHigh']=='Low')&(spikeTimedf_temp_filtered['pretone'].isin(['High','Low']))],
                            #                    'toneLevel',timwin,50,axess[2,4],color_map_B,tone_levels,[-8.5,2]) #binlength in ms
                            # ymax = np.max([ax2Hi[1],ax2Lo[1]])
                            # ax2Hi[0].set_ylim([0,np.round(ymax,decimals=1)+0.5])
                            # ax2Hi[0].yaxis.set_ticks([0,np.round(ymax,decimals=1)+0.5]) 
                            # ax2Hi[0].tick_params(axis='both', which='major', labelsize=8)
                            # ax2Lo[0].set_ylim([0,np.round(ymax,decimals=1)+0.5])
                            # ax2Lo[0].yaxis.set_ticks([0,np.round(ymax,decimals=1)+0.5]) 
                            # ax2Lo[0].tick_params(axis='both', which='major', labelsize=8)

                            # extrastr = 'LED_On'
                            # timwin = [-200,1000] #ms
                            # labeltimcombDF_temp = labeltimcombDF.copy()                 
                            # spikeTimedf_temp = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                            #                                     'soundOn',extrastr,
                            #                                     timwin,labeltimcombDF_temp,cpus)                            
                            # spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = {'soundOnFlag':[1],'LEDFlag':[1]})
                            # spikeTimedf_temp_sorted,_ = plotraster(spikeTimedf_temp_filtered,'LEDFlag',timwin,50,axess[0,3],['darkgreen','darkorange'],[0,1],[0,0.5]) #binlength in ms
                            # sorted_trialSPKs = spikeTimedf_temp_sorted.groupby(['trialNumGlob','pretoneLength'])['spktim'].size().reset_index()
                            # axess[0,3].set_xlabel('')
                            # # add fr over session for drift check
                            # nopretonetrials = np.where(sorted_trialSPKs.pretoneLength.values==0)[0][0]
                            # axess[0,4].plot(1050+250*sorted_trialSPKs.spktim.values/sorted_trialSPKs.spktim.values.max(),sorted_trialSPKs.trialNumGlob.values,color='r',linewidth=0.5)# add fr over trials to check drifting
                            # axess[0,4].arrow(1300,nopretonetrials,-10,0,head_width=30, head_length=40, fc='black', ec='black')
                            # axess[0,4].set_xlim([1000,2000])
                            # axess[0,4].set_xlabel('')
                            # axess[0,4].set_yticks([])
                            # axess[0,4].set_xticks([])

                            # extrastr = 'JS_On'
                            # timwin = [-300,300] #ms
                            # labeltimcombDF_tempJS = labeltimcombDF.copy()
                            # spikeTimedf_temp = getClsRasterMultiprocess(spikeTime_temp,spikefs,\
                            #                                     'soundOn','JoystickStartMove',
                            #                                     timwin,labeltimcombDF_tempJS,cpus)                            
                            # spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = {'soundOnFlag':[1],'choiceName':['Low','High']})
                            # _ ,_= plotraster(spikeTimedf_temp_filtered,'choiceName',timwin,50,axess[1,3],['darkgreen','darkorange'],['High','Low'],[0,0.5]) #binlength in ms
                            # axess[1,3].set_xlabel('')

                            # extrastr = 'RWD_On'                                                            
                            # _,_ = plotraster(spikeTimedf_temp_filtered,'rewardTTL',timwin,50,axess[1,4],['darkgreen','darkorange'],[0,1],[0,0.5]) #binlength in ms 
                            # axess[1,4].set_xlabel('')

                            # fig.suptitle(str(cccount)+'_'+Date_temp+'_'+keys)
                            # # fig.tight_layout()
                            # fig.savefig(outputPathway+os.path.sep+area+'_'+str(cccount)+'_'+keys+'_raster'+'.'+figformat,format = figformat)
                            # plt.close()   

                            manualInspectDF_temp = pd.DataFrame({'Monkey':[Monkey],'Region':[area],'session_cls':[Date_temp[-5:]+'_'+keys],'clsNum':[cccount],'driftYES/NO/MAYBE(1,0,2)':[np.nan]})
                            manualInspectDF = pd.concat((manualInspectDF,manualInspectDF_temp))
                            cccount = cccount+1
                            print('..................'+Date_temp+' '+area+' '+keys+' done............')

    manualInspectDF.to_excel(ResPathway+'AllClusters4inspectionSheet_'+datetime.today().strftime('%Y-%m-%d')+'.xlsx',index=False)

            
              
                    




