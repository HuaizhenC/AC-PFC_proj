import pickle
import pandas as pd
def getMonkeyDate_all():
    # MonkeyDate_all = {'MrCassius':
    #                   ['MrCassius-190414','MrCassius-190418','MrCassius-190419','MrCassius-190429',
    #                     'MrCassius-190515','MrCassius-190517','MrCassius-190531','MrCassius-190603',
    #                     'MrCassius-190605','MrCassius-190711','MrCassius-190713','MrCassius-190718',
    #                     'MrCassius-190720','MrCassius-190723','MrCassius-190725',

    #                     'MrCassius-190324','MrCassius-190328',
    #                     'MrCassius-190413','MrCassius-190520',
    #                     'MrCassius-190522','MrCassius-190526','MrCassius-190705','MrCassius-190715',
    #                     'MrCassius-190727','MrCassius-190728'],
    #                     'MrMiyagi':
    #                     ['MrMiyagi-190417','MrMiyagi-190425','MrMiyagi-190427','MrMiyagi-190502',
    #                     'MrMiyagi-190514','MrMiyagi-190516','MrMiyagi-190525','MrMiyagi-190527',
    #                     'MrMiyagi-190530','MrMiyagi-190601','MrMiyagi-190604','MrMiyagi-190704',
    #                     'MrMiyagi-190709','MrMiyagi-190717','MrMiyagi-190719','MrMiyagi-190722',

    #                     'MrMiyagi-190422','MrMiyagi-190518']}
    MonkeyDate_all = {'MrMiyagi':
                        ['MrMiyagi-190427','MrMiyagi-190502',
                        'MrMiyagi-190514','MrMiyagi-190516','MrMiyagi-190525','MrMiyagi-190527',
                        'MrMiyagi-190530','MrMiyagi-190601','MrMiyagi-190604','MrMiyagi-190704',
                        'MrMiyagi-190709','MrMiyagi-190717','MrMiyagi-190719','MrMiyagi-190722',

                        'MrMiyagi-190422','MrMiyagi-190518']}    
    # MonkeyDate_all = {'MrCassius':
    #                   ['MrCassius-190414']}  
    return MonkeyDate_all

def neuronfilterDF(excelpathway,AVResPathway,method):
    if method=='ttest':
        # filter out neurons based on ttest
        df_avMod_all = pickle.load(open(AVResPathway+'AVmodTTestDF.pkl','rb')).reset_index()
        df_avMod_all['session_cls_Region'] = df_avMod_all['session_cls']+'_'+df_avMod_all['Region'] 
        df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.05] 
        clscolstr = 'session_cls_Region'
    if method=='all':
        # keep all neurons
        df_avMod_all = pickle.load(open(AVResPathway+'AVmodTTestDF.pkl','rb')).reset_index()
        df_avMod_all['session_cls_Region'] = df_avMod_all['session_cls']+'_'+df_avMod_all['Region'] 
        df_avMod_all_sig = df_avMod_all.copy()
        clscolstr = 'session_cls_Region'  
    # filter out drifted neurons
    dfinspect = pd.read_excel(excelpathway+'AllClusters4inspectionSheet.xlsx')
    dfinspect['session_cls_Region'] = dfinspect['session_cls']+'_'+dfinspect['Region']
    nodriftedUnites = list(dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)'].isin([0])]['session_cls_Region'].values)
    df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['session_cls_Region'].isin(nodriftedUnites)]
    print('after remove drifting (only keep 0 drift) : '+str(len(df_avMod_all_sig['session_cls_Region'].unique())))  
        
    # check usable neurons in each session
    df_usableBYsess=df_avMod_all_sig.drop_duplicates(subset='session_cls_Region',keep='first')
    df_usableBYsess.loc[:,'sess'] = df_usableBYsess.session_cls.str[:6]
    print('total usable neurons in each session')
    print(df_usableBYsess.groupby(['sess','Region']).size().reset_index())

    return df_avMod_all_sig,clscolstr



