import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import re
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# Total number of functions!!!: 9!

# 1 function
def term_paper_main_func(model0, model1, df, n_splits, show_BLP, extended):    
    '''
    The whole pipeline of the Chernozhukov paper, which will be used in my paper.
    All aspects, by which the quality of heterogeneity ML estimation is observed: 
    from MAE, MSE to specific metrics (BLP, GATES, CLAN).
    Note that column names of the df should be strongly specific (check first .ipynb notebooks!!!)
    '''
    # setting ggplot style for the graphs
    plt.style.use('ggplot')

    ######## CALCULATION OF CRITICAL VALUE FOR THE SCI!!!!
        # SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

    # 500000 samples of Max (absolute value) out of 5 standard normals FOR CRITICAL VALUE!!!
    k = np.max(np.abs(np.random.multivariate_normal(np.zeros(5), np.identity(5), 500000)), axis = 1)

    # crit value for SCI!!!!!
    if n_splits == 1: 
        # critical value for two-sided 95% interval!!!! (taking not 97.5%, since used np.abs above!!!!)
        crit_val_simult = np.percentile(k, 95)
    else:
        # critical value for two-sided 97.5% interval!!!! (taking not 98.75%, since used np.abs above!!!!)
        # DOING 97.5% SINCE AFTERWARDS I WILL DO MEDIANS AND INTERVAL WILL BECOME 95%!!!!
        crit_val_simult = np.percentile(k, 97.5)

    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+')
        if regexp.search(i):
            x_cols.append(i)

    ######## Initializing arrays for tables and metrices of each split for 
                # further median SCI or histrograms and further descriptive statistics!!!!!!!!!!!!!!!!!!!!!
    BLP_metrics = np.zeros(n_splits)
    GATES_metrics = np.zeros(n_splits)
    GATES_tables = np.zeros((n_splits, 5, 6))
    CLAN_group1_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    CLAN_group5_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    CLAN_group5_minus_group1_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    MAEs = np.zeros(n_splits)
    MSEs = np.zeros(n_splits)
    #UPD: new vectors!!!
    beta_2s = np.zeros(n_splits)
    real_beta_2s = np.zeros(n_splits)
    real_BLP_metrics = np.zeros(n_splits)

    # iterating for different data splits!!!!
    for split in tqdm(range(n_splits)):

        ### Split the data onto 2 parts
        obs = df.shape[0]
        ind_A = random.sample(range(int(obs)), int(obs / 2))
        ind_B = list(set(range(int(obs))).difference(set(ind_A)))
        df_A = df.iloc[ind_A, :].copy()
        df_B = df.iloc[ind_B, :].copy()

        ###### ML Modelling
        # B(X) (D = 0)
        X_train0 = df_A[df_A['D'] == 0][x_cols]
        y_train0 = df_A[df_A['D'] == 0]['y']
        X_pred0 = df_B[x_cols]
        # MinMaxSCALER!!!!! (as mentioned in paper)
        scaler0 = MinMaxScaler()
        scaler0.fit(X_train0)
        model0.fit(scaler0.transform(X_train0), y_train0)
        df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

        # D = 1
        X_train1 = df_A[df_A['D'] == 1][x_cols]
        y_train1 = df_A[df_A['D'] == 1]['y']
        X_pred1 = df_B[x_cols]
        # MinMaxSCALER!!!!! (as mentioned in paper)
        scaler1 = MinMaxScaler()
        scaler1.fit(X_train1)
        model1.fit(scaler1.transform(X_train1), y_train1)
        df_B['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred1))

        # ML estimation of heterogeneity S(X) 
        df_B['S(X)'] = df_B['E(Y|D=1)'] - df_B['B(X)']


        ######## I) BLP: page 11 of the paper (1st strategy)
        
        # creating missing variables for BLP estimation
        df_B['const'] = 1
        df_B['D_minus_p'] = df_B['D'] - df_B['p']
        df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))
        df_B['S(X)_minus_ES'] = df_B['S(X)'] - np.mean(df_B['S(X)'])

        vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ### Weighting variable for the WLS
        df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

        ### Variables for the BLP WLS
        Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                     # if heterogeneity exists!!!!
        model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')

        # displaying BLP if specified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if show_BLP == True:
            print('BLP regression summary, iteration: {}'.format(split))
            display(model_BLP.summary())

        ################### FIRST METRIC OF MODEL QUALITY FROM THE PAPER (FIRST TARGET IN HYPERPARAMETER TUNING)
        Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_B['S(X)'])
        BLP_metrics[split] = Lambda_hat
        beta_2 = model_BLP.params[-1]
        beta_2s[split] = beta_2
        
        ##### UPD REAL BLP!!!!!! AND REAL LAMBDA HAT BLP METRIC!!!
        model_real_BLP = sm.OLS(df_B['s_0(X)'], df_B[['const', 'S(X)_minus_ES']]).fit()
        real_beta_2 = model_real_BLP.params[-1]
        real_beta_2s[split] = real_beta_2
        real_lambda_hat = real_beta_2 ** 2 * np.var(df_B['S(X)'])
        real_BLP_metrics[split] = real_lambda_hat
                                                       

        ## Creating group dummy variables: Split onto 5 equal-sized groups, based on S(X) 
                 #(1st group with lowest S(X), 5th group with highest S(X))
        df_B = df_B.sort_values('S(X)')

        for i in range(5):
            name = 'Group_{}'.format(i + 1)
            zero_vector = np.zeros(df_B.shape[0])
            # needed quantile gets 1!!
            start_index = int(round(i * df_B.shape[0] / 5))
            end_index = int(round((i + 1) * df_B.shape[0] / 5))
            zero_vector[start_index: end_index] = 1
            df_B[name] = zero_vector

        df_B = df_B.sort_index()

        ## Creating variables for the GATES 1st strategy WLS
        # non-weighted new vars
        for i in range(5):
            name = 'D_minus_p_Group_{}'.format(i + 1)
            group_name = 'Group_{}'.format(i + 1)
            df_B[name] = df_B['D_minus_p'] * df_B[group_name]

        Xs_for_GATES = ['const', 'B(X)', 'D_minus_p_Group_1',
               'D_minus_p_Group_2', 'D_minus_p_Group_3',
               'D_minus_p_Group_4', 'D_minus_p_Group_5']

        ##### Actual GATES WLS
        model_GATES = sm.WLS(df_B['y'], df_B[Xs_for_GATES], weights = df_B['omega(X)']).fit(cov_type='HC3')

        ################### SECOND METRIC OF MODEL QUALITY FROM THE PAPER (SECOND CANDIDATE OF TARGET IN HYPERPARAMETER TUNING)
        # assuming that there will always be a partition onto 5 groups, 
            # SO TAKING LAST 5 PARAMETERS HERE IS ALWAYS CORRECT!!!!!!
        Lambda_bar_hat = 0.2 * np.sum(model_GATES.params[-5:] ** 2)
        GATES_metrics[split] = Lambda_bar_hat

        ######### GATES CALCULATION
        group_cols = ['Group_1', 'Group_2', 'Group_3', 'Group_4', 'Group_5']

        Estimated_GATES = model_GATES.params[-5:]
        True_GATES = [df_B[df_B[group_cols[i]] == 1]['s_0(X)'].mean() for i in range(5)]

        #################### III) CLAN
        #### Page 17 of the paper, just do GATES difference and other covariates difference

        df_B_group1 = df_B[df_B['Group_1'] == 1]
        df_B_group5 = df_B[df_B['Group_5'] == 1]
        # leave only needed columns for CLAN, based on tables 4, 5, 6 from the paper!!!!
        cols_for_CLAN = ['x1', 'x2', 'x3', 'x4', 'x5', 'D', 'y', 'p', 's_0(X)', 'B(X)',
               'E(Y|D=1)', 'S(X)']

        ########### a) Analysis of the GATES, SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

        ##### Proceeding with simultaneous confidence intervals!!!
        
        # SE of coefs, and coefs themselves
        EST_GATES = model_GATES.params[-5:]
        SE_EST_GATES = model_GATES.bse[-5:]

        # simult CI for the GATES in estimated groups!!!
        simult_CI_L_EST_GATES = EST_GATES - crit_val_simult * SE_EST_GATES
        simult_CI_U_EST_GATES = EST_GATES + crit_val_simult * SE_EST_GATES

        #### Grouping by REAL treatment effect!!!: Split onto 5 equal-sized groups, 
            # based on s_0(X) (1st group with lowest s_0(X), 5th group with highest s_0(X))
        df_B = df_B.sort_values('s_0(X)')

        for i in range(5):
            name = 'Group_{}_REAL'.format(i + 1)
            zero_vector = np.zeros(df_B.shape[0])
            # needed quantile gets 1!!
            start_index = int(round(i * df_B.shape[0] / 5))
            end_index = int(round((i + 1) * df_B.shape[0] / 5))
            zero_vector[start_index: end_index] = 1
            df_B[name] = zero_vector

        df_B = df_B.sort_index()

        ##### REAL GATES
        REAL_GATES = np.zeros(5)
        SE_REAL_GATES = np.zeros(5)
        for num, i in enumerate(['Group_1_REAL', 'Group_2_REAL', 'Group_3_REAL', 'Group_4_REAL', 'Group_5_REAL']):
            REAL_GATES[num] = df_B[df_B[i] == 1]['s_0(X)'].mean()
            SE_REAL_GATES[num] = df_B[df_B[i] == 1]['s_0(X)'].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])


        #### simult CI for the GATES in REAL groups!!!
        simult_CI_L_REAL_GATES = REAL_GATES - crit_val_simult * SE_REAL_GATES
        simult_CI_U_REAL_GATES = REAL_GATES + crit_val_simult * SE_REAL_GATES

        ####### 97.5% SCI, GATES
        GATES_table = pd.DataFrame({'Group': range(1, 6), 'GATES_estimated': EST_GATES, 'SCI_L_estimated': simult_CI_L_EST_GATES, 
                      'SCI_U_estimated': simult_CI_U_EST_GATES, 'GATES_real': REAL_GATES,
                      'SCI_L_real': simult_CI_L_REAL_GATES, 'SCI_U_real': simult_CI_U_REAL_GATES}).set_index('Group')
        GATES_tables[split] = GATES_table.to_numpy()

        ####### 97.5% SCI everywhere below (AFTER MEDIAN IT WILL BE 95%!!!!

        ####### CLAN Group 1

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G1_est = np.zeros(num_rows_CLAN)
        SCI_L_G1_est = np.zeros(num_rows_CLAN)
        SCI_U_G1_est = np.zeros(num_rows_CLAN)
        G1_real = np.zeros(num_rows_CLAN)
        SCI_L_G1_real = np.zeros(num_rows_CLAN)
        SCI_U_G1_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G1 = np.zeros((num_rows_CLAN, 6))
        # GATES
        CLAN_data_G1[0] = GATES_table.iloc[0]
        # Other vars (X), filling by column
        for num, i in enumerate(['Group_1', 'Group_1_REAL']):
            # point estimate
            CLAN_data_G1[1:, num*3] = df_B[df_B[i] == 1][x_cols].mean()
            # Lower SCI
            CLAN_data_G1[1:, num*3 + 1] = df_B[df_B[i] == 1][x_cols].mean() -\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])
            # Upper SCI
            CLAN_data_G1[1:, num*3 + 2] = df_B[df_B[i] == 1][x_cols].mean() +\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])

        CLAN_group1_tables[split] = CLAN_data_G1

        ####### CLAN Group 5

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G5_est = np.zeros(num_rows_CLAN)
        SCI_L_G5_est = np.zeros(num_rows_CLAN)
        SCI_U_G5_est = np.zeros(num_rows_CLAN)
        G5_real = np.zeros(num_rows_CLAN)
        SCI_L_G5_real = np.zeros(num_rows_CLAN)
        SCI_U_G5_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G5 = np.zeros((num_rows_CLAN, 6))
        # GATES
        CLAN_data_G5[0] = GATES_table.iloc[4]
        # Other vars (X), filling by column
        for num, i in enumerate(['Group_5', 'Group_5_REAL']):
            # point estimate
            CLAN_data_G5[1:, num*3] = df_B[df_B[i] == 1][x_cols].mean()
            # Lower SCI
            CLAN_data_G5[1:, num*3 + 1] = df_B[df_B[i] == 1][x_cols].mean() -\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])
            # Upper SCI
            CLAN_data_G5[1:, num*3 + 2] = df_B[df_B[i] == 1][x_cols].mean() +\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])

        CLAN_group5_tables[split] = CLAN_data_G5

        ######### Group 5 - Group 1 
        ## IN SCI DIVIDING BY SQRT OF NUMBER OF GROUP 5 OBSERVATIONS, ASSUMING THAT THE NUMBER OF 
            # OBSERVATIONS IN FIRST AND FIFTH GROUP IS THE SAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G5_minus_G1_est = np.zeros(num_rows_CLAN)
        SCI_L_G5_minus_G1_est = np.zeros(num_rows_CLAN)
        SCI_U_G5_minus_G1_est = np.zeros(num_rows_CLAN)
        G5_minus_G1_real = np.zeros(num_rows_CLAN)
        SCI_L_G5_minus_G1_real = np.zeros(num_rows_CLAN)
        SCI_U_G5_minus_G1_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G5_minus_G1 = np.zeros((num_rows_CLAN, 6))
        ## GATES (real and estimated simultaneously)
        # point estimates
        CLAN_data_G5_minus_G1[0, [0, 3]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]])
        SCI_radius_G5_minus_G1_GATES = np.sqrt((np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[4, [1, 4]])) ** 2 +\
                                               (np.array(GATES_table.iloc[0, [0, 3]]) - np.array(GATES_table.iloc[0, [1, 4]])) ** 2)
        # SCI L
        CLAN_data_G5_minus_G1[0, [1, 4]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]]) -\
                                                SCI_radius_G5_minus_G1_GATES
        # SCI U
        CLAN_data_G5_minus_G1[0, [2, 5]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]]) +\
                                                SCI_radius_G5_minus_G1_GATES

        ## Other vars (X), filling by column
        for num, i in enumerate([['Group_5', 'Group_1'], ['Group_5_REAL', 'Group_1_REAL']]):
            G5_var = i[0]
            G1_var = i[1]
            # point estimate
            CLAN_data_G5_minus_G1[1:, num*3] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean()
            # Lower SCI: DIVIDING BY SQRT OF NUMBER OF GROUP 5 OBSERVATIONS, ASSUMING THAT THE NUMBER OF OBSERVATIONS IN FIRST AND 
                # FIFTH GROUP IS THE SAME!!!!
            CLAN_data_G5_minus_G1[1:, num*3 + 1] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean() -\
                                    (df_B[df_B[G5_var] == 1][x_cols].var() + df_B[df_B[G1_var] == 1][x_cols].var()) ** 0.5 /\
                                    np.sqrt(df_B[df_B[G5_var] == 1].shape[0]) * crit_val_simult
            # Upper SCI
            CLAN_data_G5_minus_G1[1:, num*3 + 2] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean() +\
                                    (df_B[df_B[G5_var] == 1][x_cols].var() + df_B[df_B[G1_var] == 1][x_cols].var()) ** 0.5 /\
                                    np.sqrt(df_B[df_B[G5_var] == 1].shape[0]) * crit_val_simult


        CLAN_group5_minus_group1_tables[split] = CLAN_data_G5_minus_G1

        ############## MSE and MAE of ML prediction of the treatment effect!!!
        MSE = np.sum((df_B['s_0(X)'] - df_B['S(X)']) ** 2) / df_B.shape[0]
        MAE = np.sum(np.abs(df_B['s_0(X)'] - df_B['S(X)'])) / df_B.shape[0]

        MSEs[split] = MSE
        MAEs[split] = MAE
    
    # Output depends on the number of splits (1 or more than 1)
    if n_splits == 1:
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics, 3), 'GATES_metrics': np.round(GATES_metrics, 3), 
                                  'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3), 
                                  'real_BLP_metrics': np.round(real_BLP_metrics, 3), 'beta_2s': np.round(beta_2s, 3), 
                                  'real_beta_2s': np.round(real_beta_2s, 3)})
        print('Metrics table')
        display(df_metrics)
        df_GATES = pd.DataFrame(np.round(GATES_tables[0], 3))
        df_GATES.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
        df_GATES.index = ['G1', 'G2', 'G3', 'G4', 'G5']
        print('GATES table')
        display(df_GATES)
        CLAN_dfs = []
        for i in [CLAN_group1_tables, CLAN_group5_tables, CLAN_group5_minus_group1_tables]:
            df0 = pd.DataFrame(np.round(i[0], 3))
            df0.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
            df0.index = ['GATES'] + x_cols
            CLAN_dfs.append(df0)

        df_CLAN_group1 = CLAN_dfs[0]
        print('CLAN table group 1')
        display(df_CLAN_group1)
        df_CLAN_group5 = CLAN_dfs[1]
        print('CLAN table group 5')
        display(df_CLAN_group5)
        df_CLAN_group5_minus_group1 = CLAN_dfs[2]
        print('CLAN table group 5 minus group 1')
        display(df_CLAN_group5_minus_group1)
        
        if extended == True:
            print('EXTENDED OPTION IS AVAILABLE ONLY IF N_SPLITS IS BIGGER THAN 1')

    else: 
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics, 3), 'GATES_metrics': np.round(GATES_metrics, 3), 
                                  'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3), 
                                  'real_BLP_metrics': np.round(real_BLP_metrics, 3), 'beta_2s': np.round(beta_2s, 3), 
                                  'real_beta_2s': np.round(real_beta_2s, 3)})
        print('Metrics table')
        display(df_metrics.describe())
        df_GATES = pd.DataFrame(np.round(np.median(GATES_tables, axis = 0), 3))
        df_GATES.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
        df_GATES.index = ['G1', 'G2', 'G3', 'G4', 'G5']
        print('GATES table')
        display(df_GATES)
        CLAN_dfs = []
        for i in [CLAN_group1_tables, CLAN_group5_tables, CLAN_group5_minus_group1_tables]:
            df0 = pd.DataFrame(np.round(np.median(i, axis = 0), 3))
            df0.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
            df0.index = ['GATES'] + x_cols
            CLAN_dfs.append(df0)

        df_CLAN_group1 = CLAN_dfs[0]
        print('CLAN table group 1')
        display(df_CLAN_group1)
        df_CLAN_group5 = CLAN_dfs[1]
        print('CLAN table group 5')
        display(df_CLAN_group5)
        df_CLAN_group5_minus_group1 = CLAN_dfs[2]
        print('CLAN table group 5 minus group 1')
        display(df_CLAN_group5_minus_group1)
        
        if extended == True: 
            Metrics_strings = ['BLP_metrics', 'real_BLP_metrics', 'GATES_metrics', 'MAEs', 'MSEs', 
                               'beta_2s', 'real_beta_2s', 'BLP_metrics_diff_real', 'beta_2s_diff_real']
            for num, metric in enumerate([BLP_metrics, real_BLP_metrics, GATES_metrics, MAEs, MSEs, 
                                          beta_2s, real_beta_2s, BLP_metrics - real_BLP_metrics, 
                                                                beta_2s - real_beta_2s]): 
                bins_num = int(np.sqrt(n_splits))
                plt.hist(metric, bins = bins_num)
                plt.title(Metrics_strings[num], fontsize = 25)
                plt.show()


    return df_metrics, df_GATES, df_CLAN_group1, df_CLAN_group5, df_CLAN_group5_minus_group1



#2 function
def beta_2(df, model0, model1):
    '''
    Coefficient which shows how accurate the ML estimation of heterogeneity is (if heterogeneity exists)
    '''
    
    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+')
        if regexp.search(i):
            x_cols.append(i)

    ### Split the data onto 2 parts
    obs = df.shape[0]
    ind_A = random.sample(range(int(obs)), int(obs / 2))
    ind_B = list(set(range(int(obs))).difference(set(ind_A)))
    df_A = df.iloc[ind_A, :].copy()
    df_B = df.iloc[ind_B, :].copy()

    ###### ML Modelling
    # B(X) (D = 0)
    X_train0 = df_A[df_A['D'] == 0][x_cols]
    y_train0 = df_A[df_A['D'] == 0]['y']
    X_pred0 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler0 = MinMaxScaler()
    scaler0.fit(X_train0)
    model0.fit(scaler0.transform(X_train0), y_train0)
    df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

    # D = 1
    X_train1 = df_A[df_A['D'] == 1][x_cols]
    y_train1 = df_A[df_A['D'] == 1]['y']
    X_pred1 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler1 = MinMaxScaler()
    scaler1.fit(X_train1)
    model1.fit(scaler1.transform(X_train1), y_train1)
    df_B['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred1))

    # ML estimation of heterogeneity S(X) 
    df_B['S(X)'] = df_B['E(Y|D=1)'] - df_B['B(X)']


    ######## I) BLP: page 11 of the paper (1st strategy)

    # creating missing variables for BLP estimation
    df_B['const'] = 1
    df_B['D_minus_p'] = df_B['D'] - df_B['p']
    df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))

    vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

    ### Weighting variable for the WLS
    df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

    ### Variables for the BLP WLS
    Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

    ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                 # if heterogeneity exists!!!!
    model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')
    Beta_2 = model_BLP.params[-1] 
    
    # Returning beta 2 parameter!!!
    return Beta_2
 
    

# 3 function
def lambda_hat(df, model0, model1):
    '''
    BLP metric from the paper
    '''
    
    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+')
        if regexp.search(i):
            x_cols.append(i)

    ### Split the data onto 2 parts
    obs = df.shape[0]
    ind_A = random.sample(range(int(obs)), int(obs / 2))
    ind_B = list(set(range(int(obs))).difference(set(ind_A)))
    df_A = df.iloc[ind_A, :].copy()
    df_B = df.iloc[ind_B, :].copy()

    ###### ML Modelling
    # B(X) (D = 0)
    X_train0 = df_A[df_A['D'] == 0][x_cols]
    y_train0 = df_A[df_A['D'] == 0]['y']
    X_pred0 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler0 = MinMaxScaler()
    scaler0.fit(X_train0)
    model0.fit(scaler0.transform(X_train0), y_train0)
    df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

    # D = 1
    X_train1 = df_A[df_A['D'] == 1][x_cols]
    y_train1 = df_A[df_A['D'] == 1]['y']
    X_pred1 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler1 = MinMaxScaler()
    scaler1.fit(X_train1)
    model1.fit(scaler1.transform(X_train1), y_train1)
    df_B['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred1))

    # ML estimation of heterogeneity S(X) 
    df_B['S(X)'] = df_B['E(Y|D=1)'] - df_B['B(X)']


    ######## I) BLP: page 11 of the paper (1st strategy)

    # creating missing variables for BLP estimation
    df_B['const'] = 1
    df_B['D_minus_p'] = df_B['D'] - df_B['p']
    df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))

    vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

    ### Weighting variable for the WLS
    df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

    ### Variables for the BLP WLS
    Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

    ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                 # if heterogeneity exists!!!!
    model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')
    Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_B['S(X)'])
    
    # returning lambda_hat
    return Lambda_hat



# 4 function
def lambda_bar_hat(df, model0, model1):
    '''
    GATES metric from the paper
    '''
    
    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+')
        if regexp.search(i):
            x_cols.append(i)

    ### Split the data onto 2 parts
    obs = df.shape[0]
    ind_A = random.sample(range(int(obs)), int(obs / 2))
    ind_B = list(set(range(int(obs))).difference(set(ind_A)))
    df_A = df.iloc[ind_A, :].copy()
    df_B = df.iloc[ind_B, :].copy()

    ###### ML Modelling
    # B(X) (D = 0)
    X_train0 = df_A[df_A['D'] == 0][x_cols]
    y_train0 = df_A[df_A['D'] == 0]['y']
    X_pred0 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler0 = MinMaxScaler()
    scaler0.fit(X_train0)
    model0.fit(scaler0.transform(X_train0), y_train0)
    df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

    # D = 1
    X_train1 = df_A[df_A['D'] == 1][x_cols]
    y_train1 = df_A[df_A['D'] == 1]['y']
    X_pred1 = df_B[x_cols]
    # MinMaxSCALER!!!!! (as mentioned in paper)
    scaler1 = MinMaxScaler()
    scaler1.fit(X_train1)
    model1.fit(scaler1.transform(X_train1), y_train1)
    df_B['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred1))

    # ML estimation of heterogeneity S(X) 
    df_B['S(X)'] = df_B['E(Y|D=1)'] - df_B['B(X)']
    
    # creating missing variables for GATES estimation
    df_B['const'] = 1
    df_B['D_minus_p'] = df_B['D'] - df_B['p']
    
    ### Weighting variable for the WLS
    df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

    ## Creating group dummy variables: Split onto 5 equal-sized groups, based on S(X) 
                 #(1st group with lowest S(X), 5th group with highest S(X))
    df_B = df_B.sort_values('S(X)')

    for i in range(5):
        name = 'Group_{}'.format(i + 1)
        zero_vector = np.zeros(df_B.shape[0])
        # needed quantile gets 1!!
        start_index = int(round(i * df_B.shape[0] / 5))
        end_index = int(round((i + 1) * df_B.shape[0] / 5))
        zero_vector[start_index: end_index] = 1
        df_B[name] = zero_vector

    df_B = df_B.sort_index()

    ## Creating variables for the GATES 1st strategy WLS
    # non-weighted new vars
    for i in range(5):
        name = 'D_minus_p_Group_{}'.format(i + 1)
        group_name = 'Group_{}'.format(i + 1)
        df_B[name] = df_B['D_minus_p'] * df_B[group_name]

    Xs_for_GATES = ['const', 'B(X)', 'D_minus_p_Group_1',
           'D_minus_p_Group_2', 'D_minus_p_Group_3',
           'D_minus_p_Group_4', 'D_minus_p_Group_5']

    ##### Actual GATES WLS
    model_GATES = sm.WLS(df_B['y'], df_B[Xs_for_GATES], weights = df_B['omega(X)']).fit(cov_type='HC3')

    ################### SECOND METRIC OF MODEL QUALITY FROM THE PAPER (SECOND CANDIDATE OF TARGET IN HYPERPARAMETER TUNING)
    # assuming that there will always be a partition onto 5 groups, 
        # SO TAKING LAST 5 PARAMETERS HERE IS ALWAYS CORRECT!!!!!!
    Lambda_bar_hat = 0.2 * np.sum(model_GATES.params[-5:] ** 2)
    
    # returning lambda_hat
    return Lambda_bar_hat


# function 5!!!!!!
def lambda_hat_tuning_CV(model1, df1, x_cols, list_models0, OOF0_samples, scalers0):
    Lambda_hats = []
    i = 0
    for train_index, test_index in KFold(n_splits = 3).split(df1):
        tr_index = train_index
        te_index = test_index
        
        # samples, D = 1
        df_train1 = df1.iloc[tr_index].copy()
        df_test1 = df1.iloc[te_index].copy()
        X_train1 = df_train1[x_cols]
        y_train1 = df_train1['y']
        
        # samples, D = 0
        df_test0 = OOF0_samples[i]
        
        # creating MAIN OOF sample, which is imitation of B sample!!!!
        df_test = pd.concat([df_test0, df_test1])
        X_pred = df_test[x_cols]
        
        # predicting B(X) on main OOF (replica of B sample)
        model0 = list_models0[i]
        scaler0 = scalers0[i]
        df_test['B(X)'] = model0.predict(scaler0.transform(X_pred))
        
        # predicting E(Y|D=1) on main OOF (replica of B sample), WITH FIT OF MODEL WITH CERTAIN HYPERPARAMS!!!!
        # MinMaxSCALER!!!!! (as mentioned in paper)
        scaler1 = MinMaxScaler()
        scaler1.fit(X_train1)
        model1.fit(scaler1.transform(X_train1), y_train1)
        df_test['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred))
        
        # Main OOF (replica of B sample) prediction of heterogeneity S(X) 
        df_test['S(X)'] = df_test['E(Y|D=1)'] - df_test['B(X)']

        ######## I) BLP: page 11 of the paper (1st strategy)

        # creating missing variables for BLP estimation
        df_test['const'] = 1
        df_test['D_minus_p'] = df_test['D'] - df_test['p']
        df_test['D_minus_p_times_S(X)_minus_ES'] = df_test['D_minus_p'] * (df_test['S(X)'] - np.mean(df_test['S(X)']))

        vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ### Weighting variable for the WLS
        df_test['omega(X)'] = 1 / df_test['p'] / (1 - df_test['p']) 

        ### Variables for the BLP WLS
        Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                     # if heterogeneity exists!!!!
        model_BLP = sm.WLS(df_test['y'], df_test[Xs_BLP], weights = df_test['omega(X)'] ** 2).fit(cov_type='HC3')
        Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_test['S(X)'])
        Lambda_hats.append(Lambda_hat)
        i += 1

    return np.array(Lambda_hats).mean()


# 6 function
def term_paper_main_func_compare_two_models(model0_A, model0_B, model1_A, model1_B, df, n_splits, show_BLP, extended):    
    '''
    The whole pipeline of the Chernozhukov paper, which will be used in my paper.
    All aspects, by which the quality of heterogeneity ML estimation is observed: 
    from MAE, MSE to specific metrics (BLP, GATES, CLAN).
    Note that column names of the df should be strongly specific (check first .ipynb notebooks!!!)
    '''
    # setting ggplot style for the graphs
    plt.style.use('ggplot')

    ######## CALCULATION OF CRITICAL VALUE FOR THE SCI!!!!
        # SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

    # 500000 samples of Max (absolute value) out of 5 standard normals FOR CRITICAL VALUE!!!
    k = np.max(np.abs(np.random.multivariate_normal(np.zeros(5), np.identity(5), 500000)), axis = 1)

    # crit value for SCI!!!!!
    if n_splits == 1: 
        # critical value for two-sided 95% interval!!!! (taking not 97.5%, since used np.abs above!!!!)
        crit_val_simult = np.percentile(k, 95)
    else:
        # critical value for two-sided 97.5% interval!!!! (taking not 98.75%, since used np.abs above!!!!)
        # DOING 97.5% SINCE AFTERWARDS I WILL DO MEDIANS AND INTERVAL WILL BECOME 95%!!!!
        crit_val_simult = np.percentile(k, 97.5)

    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+$')
        if regexp.search(i):
            x_cols.append(i)

    ######## Initializing arrays for tables and metrices of each split for 
                # further median SCI or histrograms and further descriptive statistics!!!!!!!!!!!!!!!!!!!!!
    BLP_metrics_diffs = np.zeros(n_splits)
    GATES_metrics_diffs = np.zeros(n_splits)
    MAEs_diffs = np.zeros(n_splits)
    MSEs_diffs = np.zeros(n_splits)
    
    ### models A and B initializing
    models0 = [model0_A, model0_B]
    models1 = [model1_A, model1_B]

    # iterating for different data splits!!!!
    for split in tqdm(range(n_splits)):

        ### Split the data onto 2 parts
        obs = df.shape[0]
        ind_A = random.sample(range(int(obs)), int(obs / 2))
        ind_B = list(set(range(int(obs))).difference(set(ind_A)))
        df_A = df.iloc[ind_A, :].copy()
        df_B = df.iloc[ind_B, :].copy()
        
        # initializing metrics store for both models, 1 split
        BLP_metrics_split = []
        GATES_metrics_split = []
        MAEs_split = []
        MSEs_split = []
        
        # estimation of metric for both set of models, given same split
        for type_model in range(2):
            
            # 1st: A models, 2nd: B models
            model0 = models0[type_model]
            model1 = models1[type_model]

            ###### ML Modelling
            # B(X) (D = 0)
            X_train0 = df_A[df_A['D'] == 0][x_cols]
            y_train0 = df_A[df_A['D'] == 0]['y']
            X_pred0 = df_B[x_cols]
            # MinMaxSCALER!!!!! (as mentioned in paper)
            scaler0 = MinMaxScaler()
            scaler0.fit(X_train0)
            model0.fit(scaler0.transform(X_train0), y_train0)
            df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

            # D = 1
            X_train1 = df_A[df_A['D'] == 1][x_cols]
            y_train1 = df_A[df_A['D'] == 1]['y']
            X_pred1 = df_B[x_cols]
            # MinMaxSCALER!!!!! (as mentioned in paper)
            scaler1 = MinMaxScaler()
            scaler1.fit(X_train1)
            model1.fit(scaler1.transform(X_train1), y_train1)
            df_B['E(Y|D=1)'] = model1.predict(scaler1.transform(X_pred1))

            # ML estimation of heterogeneity S(X) 
            df_B['S(X)'] = df_B['E(Y|D=1)'] - df_B['B(X)']


            ######## I) BLP: page 11 of the paper (1st strategy)

            # creating missing variables for BLP estimation
            df_B['const'] = 1
            df_B['D_minus_p'] = df_B['D'] - df_B['p']
            df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))

            vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

            ### Weighting variable for the WLS
            df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

            ### Variables for the BLP WLS
            Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

            ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                         # if heterogeneity exists!!!!
            model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')

            # displaying BLP if specified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if show_BLP == True:
                print('BLP regression summary, iteration: {}'.format(split))
                display(model_BLP.summary())

            ################### FIRST METRIC OF MODEL QUALITY FROM THE PAPER (FIRST TARGET IN HYPERPARAMETER TUNING)
            Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_B['S(X)'])
            BLP_metrics_split.append(Lambda_hat)


            ## Creating group dummy variables: Split onto 5 equal-sized groups, based on S(X) 
                     #(1st group with lowest S(X), 5th group with highest S(X))
            df_B = df_B.sort_values('S(X)')

            for i in range(5):
                name = 'Group_{}'.format(i + 1)
                zero_vector = np.zeros(df_B.shape[0])
                # needed quantile gets 1!!
                start_index = int(round(i * df_B.shape[0] / 5))
                end_index = int(round((i + 1) * df_B.shape[0] / 5))
                zero_vector[start_index: end_index] = 1
                df_B[name] = zero_vector

            df_B = df_B.sort_index()

            ## Creating variables for the GATES 1st strategy WLS
            # non-weighted new vars
            for i in range(5):
                name = 'D_minus_p_Group_{}'.format(i + 1)
                group_name = 'Group_{}'.format(i + 1)
                df_B[name] = df_B['D_minus_p'] * df_B[group_name]

            Xs_for_GATES = ['const', 'B(X)', 'D_minus_p_Group_1',
                   'D_minus_p_Group_2', 'D_minus_p_Group_3',
                   'D_minus_p_Group_4', 'D_minus_p_Group_5']

            ##### Actual GATES WLS
            model_GATES = sm.WLS(df_B['y'], df_B[Xs_for_GATES], weights = df_B['omega(X)']).fit(cov_type='HC3')

            ################### SECOND METRIC OF MODEL QUALITY FROM THE PAPER (SECOND CANDIDATE OF TARGET IN HYPERPARAMETER TUNING)
            # assuming that there will always be a partition onto 5 groups, 
                # SO TAKING LAST 5 PARAMETERS HERE IS ALWAYS CORRECT!!!!!!
            Lambda_bar_hat = 0.2 * np.sum(model_GATES.params[-5:] ** 2)
            GATES_metrics_split.append(Lambda_bar_hat)

            ############## MSE and MAE of ML prediction of the treatment effect!!!
            MSE = np.sum((df_B['s_0(X)'] - df_B['S(X)']) ** 2) / df_B.shape[0]
            MAE = np.sum(np.abs(df_B['s_0(X)'] - df_B['S(X)'])) / df_B.shape[0]

            MSEs_split.append(MSE)
            MAEs_split.append(MAE)
            
        # adding differences for each split:
        BLP_metrics_diffs[split] = BLP_metrics_split[0] - BLP_metrics_split[1]
        GATES_metrics_diffs[split] = GATES_metrics_split[0] - GATES_metrics_split[1]
        MAEs_diffs[split] = MAEs_split[0] - MAEs_split[1]
        MSEs_diffs[split] = MSEs_split[0] - MSEs_split[1]
    
    # Output depends on the number of splits (1 or more than 1)
    if n_splits == 1:
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics_diffs, 3), 
                                   'GATES_metrics': np.round(GATES_metrics_diffs, 3), 
                                  'MAEs': np.round(MAEs_diffs, 3), 'MSEs': np.round(MSEs_diffs, 3)})
        print('Metrics table, A model metrics minus B model metrics')
        display(df_metrics)
        
        if extended == True:
            print('EXTENDED OPTION IS AVAILABLE ONLY IF N_SPLITS IS BIGGER THAN 1')

    else: 
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics_diffs, 3), 
                                   'GATES_metrics': np.round(GATES_metrics_diffs, 3), 
                                  'MAEs': np.round(MAEs_diffs, 3), 'MSEs': np.round(MSEs_diffs, 3)})
        print('Metrics table, A model metrics minus B model metrics')
        display(df_metrics.describe())
        
        if extended == True: 
            Metrics_strings = ['BLP_metrics', 'GATES_metrics', 'MAEs', 'MSEs']
            for num, metric in enumerate([BLP_metrics_diffs, GATES_metrics_diffs, MAEs_diffs, MSEs_diffs]): 
                bins_num = int(np.sqrt(n_splits))
                plt.hist(metric, bins = bins_num)
                plt.title(Metrics_strings[num], fontsize = 25)
                plt.show()


    return df_metrics


# 7 function
def term_paper_main_func_easy_baseline(df, n_splits, extended):    
    '''
    How coef to simple linear regression performs, compared to all models above
    '''
    # setting ggplot style for the graphs
    plt.style.use('ggplot')

    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+$')
        if regexp.search(i):
            x_cols.append(i)

    ######## Initializing arrays for tables and metrices of each split for 
                # further median SCI or histrograms and further descriptive statistics!!!!!!!!!!!!!!!!!!!!!
    MAEs = np.zeros(n_splits)
    MSEs = np.zeros(n_splits)

    # iterating for different data splits!!!!
    for split in tqdm(range(n_splits)):

        ### Split the data onto 2 parts
        obs = df.shape[0]
        ind_A = random.sample(range(int(obs)), int(obs / 2))
        ind_B = list(set(range(int(obs))).difference(set(ind_A)))
        df_A = df.iloc[ind_A, :].copy()
        df_B = df.iloc[ind_B, :].copy()

        ###### Simple linear regression
        df_A['const'] = 1
        vars_reg = x_cols + ['const', 'D']
        X_train = df_A[vars_reg]
        y_train = df_A['y']
        model = sm.OLS(y_train, X_train).fit()
        
        # Baseline average treatment effect
        df_B['S(X)'] = model.params[-1]

        ############## MSE and MAE of ML prediction of the baseline average treatment effect!!!
        MSE = np.sum((df_B['s_0(X)'] - df_B['S(X)']) ** 2) / df_B.shape[0]
        MAE = np.sum(np.abs(df_B['s_0(X)'] - df_B['S(X)'])) / df_B.shape[0]

        MSEs[split] = MSE
        MAEs[split] = MAE
    
    # Output depends on the number of splits (1 or more than 1)
    if n_splits == 1:
        df_metrics = pd.DataFrame({'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3)})
        print('Metrics table')
        display(df_metrics)
        
        if extended == True:
            print('EXTENDED OPTION IS AVAILABLE ONLY IF N_SPLITS IS BIGGER THAN 1')

    else: 
        df_metrics = pd.DataFrame({'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3)})
        print('Metrics table')
        display(df_metrics.describe())
        
        if extended == True: 
            Metrics_strings = ['MAEs', 'MSEs']
            for num, metric in enumerate([MAEs, MSEs]): 
                bins_num = int(np.sqrt(n_splits))
                plt.hist(metric, bins = bins_num)
                plt.title(Metrics_strings[num], fontsize = 25)
                plt.show()


    return df_metrics


# 8th function
def term_paper_main_func_PAPER_ALGO(model0, model1, df, n_splits, show_BLP, extended):    
    '''
    The whole pipeline of the Chernozhukov paper, which will be used in my paper.
    All aspects, by which the quality of heterogeneity ML estimation is observed: 
    from MAE, MSE to specific metrics (BLP, GATES, CLAN).
    Note that column names of the df should be strongly specific (check first .ipynb notebooks!!!)
    '''
    # setting ggplot style for the graphs
    plt.style.use('ggplot')

    ######## CALCULATION OF CRITICAL VALUE FOR THE SCI!!!!
        # SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

    # 500000 samples of Max (absolute value) out of 5 standard normals FOR CRITICAL VALUE!!!
    k = np.max(np.abs(np.random.multivariate_normal(np.zeros(5), np.identity(5), 500000)), axis = 1)

    # crit value for SCI!!!!!
    if n_splits == 1: 
        # critical value for two-sided 95% interval!!!! (taking not 97.5%, since used np.abs above!!!!)
        crit_val_simult = np.percentile(k, 95)
    else:
        # critical value for two-sided 97.5% interval!!!! (taking not 98.75%, since used np.abs above!!!!)
        # DOING 97.5% SINCE AFTERWARDS I WILL DO MEDIANS AND INTERVAL WILL BECOME 95%!!!!
        crit_val_simult = np.percentile(k, 97.5)

    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+')
        if regexp.search(i):
            x_cols.append(i)

    ######## Initializing arrays for tables and metrices of each split for 
                # further median SCI or histrograms and further descriptive statistics!!!!!!!!!!!!!!!!!!!!!
    BLP_metrics = np.zeros(n_splits)
    GATES_metrics = np.zeros(n_splits)
    GATES_tables = np.zeros((n_splits, 5, 6))
    CLAN_group1_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    CLAN_group5_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    CLAN_group5_minus_group1_tables = np.zeros((n_splits, len(x_cols) + 1, 6))
    MAEs = np.zeros(n_splits)
    MSEs = np.zeros(n_splits)
    #UPD: new vectors!!!
    beta_2s = np.zeros(n_splits)
    real_beta_2s = np.zeros(n_splits)
    real_BLP_metrics = np.zeros(n_splits)

    # iterating for different data splits!!!!
    for split in tqdm(range(n_splits)):

        ### Split the data onto 2 parts
        obs = df.shape[0]
        ind_A = random.sample(range(int(obs)), int(obs / 2))
        ind_B = list(set(range(int(obs))).difference(set(ind_A)))
        df_A = df.iloc[ind_A, :].copy()
        df_B = df.iloc[ind_B, :].copy()

        ###### ML Modelling
        # B(X) (D = 0)
        X_train0 = df_A[df_A['D'] == 0][x_cols]
        y_train0 = df_A[df_A['D'] == 0]['y']
        X_pred0 = df_B[x_cols]
        # MinMaxSCALER!!!!! (as mentioned in paper)
        scaler0 = MinMaxScaler()
        scaler0.fit(X_train0)
        model0.fit(scaler0.transform(X_train0), y_train0)
        df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

        # UPD: LIKE IN PAPER ESTIMATING S(X) STRAIGHT AWAY!!!!!
        X_train1 = df_A[df_A['D'] == 1][x_cols]
        # MAKING IT CLOSE TO s_0 by subtracting prediction of b_0 BEFORE MODELLING!!!
        y_train1 = df_A[df_A['D'] == 1]['y'] - model0.predict(scaler0.transform(X_train1))
        X_pred1 = df_B[x_cols]
        # MinMaxSCALER!!!!! (as mentioned in paper)
        scaler1 = MinMaxScaler()
        scaler1.fit(X_train1)
        model1.fit(scaler1.transform(X_train1), y_train1)
        
        # ML estimation of heterogeneity S(X) 
        df_B['S(X)'] = model1.predict(scaler1.transform(X_pred1))


        ######## I) BLP: page 11 of the paper (1st strategy)
        
        # creating missing variables for BLP estimation
        df_B['const'] = 1
        df_B['D_minus_p'] = df_B['D'] - df_B['p']
        df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))
        df_B['S(X)_minus_ES'] = df_B['S(X)'] - np.mean(df_B['S(X)'])

        vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ### Weighting variable for the WLS
        df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

        ### Variables for the BLP WLS
        Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

        ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                     # if heterogeneity exists!!!!
        model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')

        # displaying BLP if specified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if show_BLP == True:
            print('BLP regression summary, iteration: {}'.format(split))
            display(model_BLP.summary())

        ################### FIRST METRIC OF MODEL QUALITY FROM THE PAPER (FIRST TARGET IN HYPERPARAMETER TUNING)
        Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_B['S(X)'])
        BLP_metrics[split] = Lambda_hat
        beta_2 = model_BLP.params[-1]
        beta_2s[split] = beta_2
        
        ##### UPD REAL BLP!!!!!! AND REAL LAMBDA HAT BLP METRIC!!!
        model_real_BLP = sm.OLS(df_B['s_0(X)'], df_B[['const', 'S(X)_minus_ES']]).fit()
        real_beta_2 = model_real_BLP.params[-1]
        real_beta_2s[split] = real_beta_2
        real_lambda_hat = real_beta_2 ** 2 * np.var(df_B['S(X)'])
        real_BLP_metrics[split] = real_lambda_hat


        ## Creating group dummy variables: Split onto 5 equal-sized groups, based on S(X) 
                 #(1st group with lowest S(X), 5th group with highest S(X))
        df_B = df_B.sort_values('S(X)')

        for i in range(5):
            name = 'Group_{}'.format(i + 1)
            zero_vector = np.zeros(df_B.shape[0])
            # needed quantile gets 1!!
            start_index = int(round(i * df_B.shape[0] / 5))
            end_index = int(round((i + 1) * df_B.shape[0] / 5))
            zero_vector[start_index: end_index] = 1
            df_B[name] = zero_vector

        df_B = df_B.sort_index()

        ## Creating variables for the GATES 1st strategy WLS
        # non-weighted new vars
        for i in range(5):
            name = 'D_minus_p_Group_{}'.format(i + 1)
            group_name = 'Group_{}'.format(i + 1)
            df_B[name] = df_B['D_minus_p'] * df_B[group_name]

        Xs_for_GATES = ['const', 'B(X)', 'D_minus_p_Group_1',
               'D_minus_p_Group_2', 'D_minus_p_Group_3',
               'D_minus_p_Group_4', 'D_minus_p_Group_5']

        ##### Actual GATES WLS
        model_GATES = sm.WLS(df_B['y'], df_B[Xs_for_GATES], weights = df_B['omega(X)']).fit(cov_type='HC3')

        ################### SECOND METRIC OF MODEL QUALITY FROM THE PAPER (SECOND CANDIDATE OF TARGET IN HYPERPARAMETER TUNING)
        # assuming that there will always be a partition onto 5 groups, 
            # SO TAKING LAST 5 PARAMETERS HERE IS ALWAYS CORRECT!!!!!!
        Lambda_bar_hat = 0.2 * np.sum(model_GATES.params[-5:] ** 2)
        GATES_metrics[split] = Lambda_bar_hat

        ######### GATES CALCULATION
        group_cols = ['Group_1', 'Group_2', 'Group_3', 'Group_4', 'Group_5']

        Estimated_GATES = model_GATES.params[-5:]
        True_GATES = [df_B[df_B[group_cols[i]] == 1]['s_0(X)'].mean() for i in range(5)]

        #################### III) CLAN
        #### Page 17 of the paper, just do GATES difference and other covariates difference

        df_B_group1 = df_B[df_B['Group_1'] == 1]
        df_B_group5 = df_B[df_B['Group_5'] == 1]
        # leave only needed columns for CLAN, based on tables 4, 5, 6 from the paper!!!!
        cols_for_CLAN = ['x1', 'x2', 'x3', 'x4', 'x5', 'D', 'y', 'p', 's_0(X)', 'B(X)',
               'E(Y|D=1)', 'S(X)']

        ########### a) Analysis of the GATES, SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

        ##### Proceeding with simultaneous confidence intervals!!!
        
        # SE of coefs, and coefs themselves
        EST_GATES = model_GATES.params[-5:]
        SE_EST_GATES = model_GATES.bse[-5:]

        # simult CI for the GATES in estimated groups!!!
        simult_CI_L_EST_GATES = EST_GATES - crit_val_simult * SE_EST_GATES
        simult_CI_U_EST_GATES = EST_GATES + crit_val_simult * SE_EST_GATES

        #### Grouping by REAL treatment effect!!!: Split onto 5 equal-sized groups, 
            # based on s_0(X) (1st group with lowest s_0(X), 5th group with highest s_0(X))
        df_B = df_B.sort_values('s_0(X)')

        for i in range(5):
            name = 'Group_{}_REAL'.format(i + 1)
            zero_vector = np.zeros(df_B.shape[0])
            # needed quantile gets 1!!
            start_index = int(round(i * df_B.shape[0] / 5))
            end_index = int(round((i + 1) * df_B.shape[0] / 5))
            zero_vector[start_index: end_index] = 1
            df_B[name] = zero_vector

        df_B = df_B.sort_index()

        ##### REAL GATES
        REAL_GATES = np.zeros(5)
        SE_REAL_GATES = np.zeros(5)
        for num, i in enumerate(['Group_1_REAL', 'Group_2_REAL', 'Group_3_REAL', 'Group_4_REAL', 'Group_5_REAL']):
            REAL_GATES[num] = df_B[df_B[i] == 1]['s_0(X)'].mean()
            SE_REAL_GATES[num] = df_B[df_B[i] == 1]['s_0(X)'].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])


        #### simult CI for the GATES in REAL groups!!!
        simult_CI_L_REAL_GATES = REAL_GATES - crit_val_simult * SE_REAL_GATES
        simult_CI_U_REAL_GATES = REAL_GATES + crit_val_simult * SE_REAL_GATES

        ####### 97.5% SCI, GATES
        GATES_table = pd.DataFrame({'Group': range(1, 6), 'GATES_estimated': EST_GATES, 'SCI_L_estimated': simult_CI_L_EST_GATES, 
                      'SCI_U_estimated': simult_CI_U_EST_GATES, 'GATES_real': REAL_GATES,
                      'SCI_L_real': simult_CI_L_REAL_GATES, 'SCI_U_real': simult_CI_U_REAL_GATES}).set_index('Group')
        GATES_tables[split] = GATES_table.to_numpy()

        ####### 97.5% SCI everywhere below (AFTER MEDIAN IT WILL BE 95%!!!!

        ####### CLAN Group 1

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G1_est = np.zeros(num_rows_CLAN)
        SCI_L_G1_est = np.zeros(num_rows_CLAN)
        SCI_U_G1_est = np.zeros(num_rows_CLAN)
        G1_real = np.zeros(num_rows_CLAN)
        SCI_L_G1_real = np.zeros(num_rows_CLAN)
        SCI_U_G1_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G1 = np.zeros((num_rows_CLAN, 6))
        # GATES
        CLAN_data_G1[0] = GATES_table.iloc[0]
        # Other vars (X), filling by column
        for num, i in enumerate(['Group_1', 'Group_1_REAL']):
            # point estimate
            CLAN_data_G1[1:, num*3] = df_B[df_B[i] == 1][x_cols].mean()
            # Lower SCI
            CLAN_data_G1[1:, num*3 + 1] = df_B[df_B[i] == 1][x_cols].mean() -\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])
            # Upper SCI
            CLAN_data_G1[1:, num*3 + 2] = df_B[df_B[i] == 1][x_cols].mean() +\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])

        CLAN_group1_tables[split] = CLAN_data_G1

        ####### CLAN Group 5

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G5_est = np.zeros(num_rows_CLAN)
        SCI_L_G5_est = np.zeros(num_rows_CLAN)
        SCI_U_G5_est = np.zeros(num_rows_CLAN)
        G5_real = np.zeros(num_rows_CLAN)
        SCI_L_G5_real = np.zeros(num_rows_CLAN)
        SCI_U_G5_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G5 = np.zeros((num_rows_CLAN, 6))
        # GATES
        CLAN_data_G5[0] = GATES_table.iloc[4]
        # Other vars (X), filling by column
        for num, i in enumerate(['Group_5', 'Group_5_REAL']):
            # point estimate
            CLAN_data_G5[1:, num*3] = df_B[df_B[i] == 1][x_cols].mean()
            # Lower SCI
            CLAN_data_G5[1:, num*3 + 1] = df_B[df_B[i] == 1][x_cols].mean() -\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])
            # Upper SCI
            CLAN_data_G5[1:, num*3 + 2] = df_B[df_B[i] == 1][x_cols].mean() +\
                                    crit_val_simult * df_B[df_B[i] == 1][x_cols].std() / np.sqrt(df_B[df_B[i] == 1].shape[0])

        CLAN_group5_tables[split] = CLAN_data_G5

        ######### Group 5 - Group 1 
        ## IN SCI DIVIDING BY SQRT OF NUMBER OF GROUP 5 OBSERVATIONS, ASSUMING THAT THE NUMBER OF 
            # OBSERVATIONS IN FIRST AND FIFTH GROUP IS THE SAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # to Add GATES params
        CLAN_cols = ['GATES'] + x_cols
        num_rows_CLAN = len(CLAN_cols) 

        G5_minus_G1_est = np.zeros(num_rows_CLAN)
        SCI_L_G5_minus_G1_est = np.zeros(num_rows_CLAN)
        SCI_U_G5_minus_G1_est = np.zeros(num_rows_CLAN)
        G5_minus_G1_real = np.zeros(num_rows_CLAN)
        SCI_L_G5_minus_G1_real = np.zeros(num_rows_CLAN)
        SCI_U_G5_minus_G1_real = np.zeros(num_rows_CLAN)
        # matrix with CLAN data
        CLAN_data_G5_minus_G1 = np.zeros((num_rows_CLAN, 6))
        ## GATES (real and estimated simultaneously)
        # point estimates
        CLAN_data_G5_minus_G1[0, [0, 3]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]])
        SCI_radius_G5_minus_G1_GATES = np.sqrt((np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[4, [1, 4]])) ** 2 +\
                                               (np.array(GATES_table.iloc[0, [0, 3]]) - np.array(GATES_table.iloc[0, [1, 4]])) ** 2)
        # SCI L
        CLAN_data_G5_minus_G1[0, [1, 4]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]]) -\
                                                SCI_radius_G5_minus_G1_GATES
        # SCI U
        CLAN_data_G5_minus_G1[0, [2, 5]] = np.array(GATES_table.iloc[4, [0, 3]]) - np.array(GATES_table.iloc[0, [0, 3]]) +\
                                                SCI_radius_G5_minus_G1_GATES

        ## Other vars (X), filling by column
        for num, i in enumerate([['Group_5', 'Group_1'], ['Group_5_REAL', 'Group_1_REAL']]):
            G5_var = i[0]
            G1_var = i[1]
            # point estimate
            CLAN_data_G5_minus_G1[1:, num*3] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean()
            # Lower SCI: DIVIDING BY SQRT OF NUMBER OF GROUP 5 OBSERVATIONS, ASSUMING THAT THE NUMBER OF OBSERVATIONS IN FIRST AND 
                # FIFTH GROUP IS THE SAME!!!!
            CLAN_data_G5_minus_G1[1:, num*3 + 1] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean() -\
                                    (df_B[df_B[G5_var] == 1][x_cols].var() + df_B[df_B[G1_var] == 1][x_cols].var()) ** 0.5 /\
                                    np.sqrt(df_B[df_B[G5_var] == 1].shape[0]) * crit_val_simult
            # Upper SCI
            CLAN_data_G5_minus_G1[1:, num*3 + 2] = df_B[df_B[G5_var] == 1][x_cols].mean() - df_B[df_B[G1_var] == 1][x_cols].mean() +\
                                    (df_B[df_B[G5_var] == 1][x_cols].var() + df_B[df_B[G1_var] == 1][x_cols].var()) ** 0.5 /\
                                    np.sqrt(df_B[df_B[G5_var] == 1].shape[0]) * crit_val_simult


        CLAN_group5_minus_group1_tables[split] = CLAN_data_G5_minus_G1

        ############## MSE and MAE of ML prediction of the treatment effect!!!
        MSE = np.sum((df_B['s_0(X)'] - df_B['S(X)']) ** 2) / df_B.shape[0]
        MAE = np.sum(np.abs(df_B['s_0(X)'] - df_B['S(X)'])) / df_B.shape[0]

        MSEs[split] = MSE
        MAEs[split] = MAE
    
    # Output depends on the number of splits (1 or more than 1)
    if n_splits == 1:
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics, 3), 'GATES_metrics': np.round(GATES_metrics, 3), 
                                  'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3), 
                                  'real_BLP_metrics': np.round(real_BLP_metrics, 3), 'beta_2s': np.round(beta_2s, 3), 
                                  'real_beta_2s': np.round(real_beta_2s, 3)})
        print('Metrics table')
        display(df_metrics)
        df_GATES = pd.DataFrame(np.round(GATES_tables[0], 3))
        df_GATES.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
        df_GATES.index = ['G1', 'G2', 'G3', 'G4', 'G5']
        print('GATES table')
        display(df_GATES)
        CLAN_dfs = []
        for i in [CLAN_group1_tables, CLAN_group5_tables, CLAN_group5_minus_group1_tables]:
            df0 = pd.DataFrame(np.round(i[0], 3))
            df0.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
            df0.index = ['GATES'] + x_cols
            CLAN_dfs.append(df0)

        df_CLAN_group1 = CLAN_dfs[0]
        print('CLAN table group 1')
        display(df_CLAN_group1)
        df_CLAN_group5 = CLAN_dfs[1]
        print('CLAN table group 5')
        display(df_CLAN_group5)
        df_CLAN_group5_minus_group1 = CLAN_dfs[2]
        print('CLAN table group 5 minus group 1')
        display(df_CLAN_group5_minus_group1)
        
        if extended == True:
            print('EXTENDED OPTION IS AVAILABLE ONLY IF N_SPLITS IS BIGGER THAN 1')

    else: 
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics, 3), 'GATES_metrics': np.round(GATES_metrics, 3), 
                                  'MAEs': np.round(MAEs, 3), 'MSEs': np.round(MSEs, 3), 
                                  'real_BLP_metrics': np.round(real_BLP_metrics, 3), 'beta_2s': np.round(beta_2s, 3), 
                                  'real_beta_2s': np.round(real_beta_2s, 3)})
        print('Metrics table')
        display(df_metrics.describe())
        df_GATES = pd.DataFrame(np.round(np.median(GATES_tables, axis = 0), 3))
        df_GATES.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
        df_GATES.index = ['G1', 'G2', 'G3', 'G4', 'G5']
        print('GATES table')
        display(df_GATES)
        CLAN_dfs = []
        for i in [CLAN_group1_tables, CLAN_group5_tables, CLAN_group5_minus_group1_tables]:
            df0 = pd.DataFrame(np.round(np.median(i, axis = 0), 3))
            df0.columns = ['est', 'est_[0.025', '0.975]_est', 'real', 'real_[0.025', '0.975]_real']
            df0.index = ['GATES'] + x_cols
            CLAN_dfs.append(df0)

        df_CLAN_group1 = CLAN_dfs[0]
        print('CLAN table group 1')
        display(df_CLAN_group1)
        df_CLAN_group5 = CLAN_dfs[1]
        print('CLAN table group 5')
        display(df_CLAN_group5)
        df_CLAN_group5_minus_group1 = CLAN_dfs[2]
        print('CLAN table group 5 minus group 1')
        display(df_CLAN_group5_minus_group1)
        
        if extended == True: 
            Metrics_strings = ['BLP_metrics', 'real_BLP_metrics', 'GATES_metrics', 'MAEs', 'MSEs', 
                               'beta_2s', 'real_beta_2s', 'BLP_metrics_diff_real', 'beta_2s_diff_real']
            for num, metric in enumerate([BLP_metrics, real_BLP_metrics, GATES_metrics, MAEs, MSEs, 
                                          beta_2s, real_beta_2s, BLP_metrics - real_BLP_metrics, 
                                                                beta_2s - real_beta_2s]): 
                bins_num = int(np.sqrt(n_splits))
                plt.hist(metric, bins = bins_num)
                plt.title(Metrics_strings[num], fontsize = 25)
                plt.show()


    return df_metrics, df_GATES, df_CLAN_group1, df_CLAN_group5, df_CLAN_group5_minus_group1


# 9th function
def term_paper_main_func_compare_two_models_PAPER_ALGO(model0_A, model0_B, model1_A, model1_B, df, n_splits, show_BLP, extended):    
    '''
    The whole pipeline of the Chernozhukov paper, which will be used in my paper.
    All aspects, by which the quality of heterogeneity ML estimation is observed: 
    from MAE, MSE to specific metrics (BLP, GATES, CLAN).
    Note that column names of the df should be strongly specific (check first .ipynb notebooks!!!)
    '''
    # setting ggplot style for the graphs
    plt.style.use('ggplot')

    ######## CALCULATION OF CRITICAL VALUE FOR THE SCI!!!!
        # SCI taken from the Chernozhukov's 1st lecture: 
        ### https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf

    # 500000 samples of Max (absolute value) out of 5 standard normals FOR CRITICAL VALUE!!!
    k = np.max(np.abs(np.random.multivariate_normal(np.zeros(5), np.identity(5), 500000)), axis = 1)

    # crit value for SCI!!!!!
    if n_splits == 1: 
        # critical value for two-sided 95% interval!!!! (taking not 97.5%, since used np.abs above!!!!)
        crit_val_simult = np.percentile(k, 95)
    else:
        # critical value for two-sided 97.5% interval!!!! (taking not 98.75%, since used np.abs above!!!!)
        # DOING 97.5% SINCE AFTERWARDS I WILL DO MEDIANS AND INTERVAL WILL BECOME 95%!!!!
        crit_val_simult = np.percentile(k, 97.5)

    ### Automatic search for the covariates columns!!!
    x_cols = []
    for i in df.columns:
        regexp = re.compile(r'^x\d+$')
        if regexp.search(i):
            x_cols.append(i)

    ######## Initializing arrays for tables and metrices of each split for 
                # further median SCI or histrograms and further descriptive statistics!!!!!!!!!!!!!!!!!!!!!
    BLP_metrics_diffs = np.zeros(n_splits)
    GATES_metrics_diffs = np.zeros(n_splits)
    MAEs_diffs = np.zeros(n_splits)
    MSEs_diffs = np.zeros(n_splits)
    
    ### models A and B initializing
    models0 = [model0_A, model0_B]
    models1 = [model1_A, model1_B]

    # iterating for different data splits!!!!
    for split in tqdm(range(n_splits)):

        ### Split the data onto 2 parts
        obs = df.shape[0]
        ind_A = random.sample(range(int(obs)), int(obs / 2))
        ind_B = list(set(range(int(obs))).difference(set(ind_A)))
        df_A = df.iloc[ind_A, :].copy()
        df_B = df.iloc[ind_B, :].copy()
        
        # initializing metrics store for both models, 1 split
        BLP_metrics_split = []
        GATES_metrics_split = []
        MAEs_split = []
        MSEs_split = []
        
        # estimation of metric for both set of models, given same split
        for type_model in range(2):
            
            # 1st: A models, 2nd: B models
            model0 = models0[type_model]
            model1 = models1[type_model]

            ###### ML Modelling
            # B(X) (D = 0)
            X_train0 = df_A[df_A['D'] == 0][x_cols]
            y_train0 = df_A[df_A['D'] == 0]['y']
            X_pred0 = df_B[x_cols]
            # MinMaxSCALER!!!!! (as mentioned in paper)
            scaler0 = MinMaxScaler()
            scaler0.fit(X_train0)
            model0.fit(scaler0.transform(X_train0), y_train0)
            df_B['B(X)'] = model0.predict(scaler0.transform(X_pred0))

            # UPD: LIKE IN PAPER ESTIMATING S(X) STRAIGHT AWAY!!!!!
            X_train1 = df_A[df_A['D'] == 1][x_cols]
            # MAKING IT CLOSE TO s_0 by subtracting prediction of b_0 BEFORE MODELLING!!!
            y_train1 = df_A[df_A['D'] == 1]['y'] - model0.predict(scaler0.transform(X_train1))
            X_pred1 = df_B[x_cols]
            # MinMaxSCALER!!!!! (as mentioned in paper)
            scaler1 = MinMaxScaler()
            scaler1.fit(X_train1)
            model1.fit(scaler1.transform(X_train1), y_train1)

            # ML estimation of heterogeneity S(X) 
            df_B['S(X)'] = model1.predict(scaler1.transform(X_pred1))



            ######## I) BLP: page 11 of the paper (1st strategy)

            # creating missing variables for BLP estimation
            df_B['const'] = 1
            df_B['D_minus_p'] = df_B['D'] - df_B['p']
            df_B['D_minus_p_times_S(X)_minus_ES'] = df_B['D_minus_p'] * (df_B['S(X)'] - np.mean(df_B['S(X)']))

            vars_to_use_in_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

            ### Weighting variable for the WLS
            df_B['omega(X)'] = 1 / df_B['p'] / (1 - df_B['p']) 

            ### Variables for the BLP WLS
            Xs_BLP = ['const', 'B(X)', 'D_minus_p', 'D_minus_p_times_S(X)_minus_ES']

            ########## BLP regression and beta_2 which is proportional to the quality of prediction of heterogeneity 
                         # if heterogeneity exists!!!!
            model_BLP = sm.WLS(df_B['y'], df_B[Xs_BLP], weights = df_B['omega(X)'] ** 2).fit(cov_type='HC3')

            # displaying BLP if specified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if show_BLP == True:
                print('BLP regression summary, iteration: {}'.format(split))
                display(model_BLP.summary())

            ################### FIRST METRIC OF MODEL QUALITY FROM THE PAPER (FIRST TARGET IN HYPERPARAMETER TUNING)
            Lambda_hat = model_BLP.params[-1] ** 2 * np.var(df_B['S(X)'])
            BLP_metrics_split.append(Lambda_hat)


            ## Creating group dummy variables: Split onto 5 equal-sized groups, based on S(X) 
                     #(1st group with lowest S(X), 5th group with highest S(X))
            df_B = df_B.sort_values('S(X)')

            for i in range(5):
                name = 'Group_{}'.format(i + 1)
                zero_vector = np.zeros(df_B.shape[0])
                # needed quantile gets 1!!
                start_index = int(round(i * df_B.shape[0] / 5))
                end_index = int(round((i + 1) * df_B.shape[0] / 5))
                zero_vector[start_index: end_index] = 1
                df_B[name] = zero_vector

            df_B = df_B.sort_index()

            ## Creating variables for the GATES 1st strategy WLS
            # non-weighted new vars
            for i in range(5):
                name = 'D_minus_p_Group_{}'.format(i + 1)
                group_name = 'Group_{}'.format(i + 1)
                df_B[name] = df_B['D_minus_p'] * df_B[group_name]

            Xs_for_GATES = ['const', 'B(X)', 'D_minus_p_Group_1',
                   'D_minus_p_Group_2', 'D_minus_p_Group_3',
                   'D_minus_p_Group_4', 'D_minus_p_Group_5']

            ##### Actual GATES WLS
            model_GATES = sm.WLS(df_B['y'], df_B[Xs_for_GATES], weights = df_B['omega(X)']).fit(cov_type='HC3')

            ################### SECOND METRIC OF MODEL QUALITY FROM THE PAPER (SECOND CANDIDATE OF TARGET IN HYPERPARAMETER TUNING)
            # assuming that there will always be a partition onto 5 groups, 
                # SO TAKING LAST 5 PARAMETERS HERE IS ALWAYS CORRECT!!!!!!
            Lambda_bar_hat = 0.2 * np.sum(model_GATES.params[-5:] ** 2)
            GATES_metrics_split.append(Lambda_bar_hat)

            ############## MSE and MAE of ML prediction of the treatment effect!!!
            MSE = np.sum((df_B['s_0(X)'] - df_B['S(X)']) ** 2) / df_B.shape[0]
            MAE = np.sum(np.abs(df_B['s_0(X)'] - df_B['S(X)'])) / df_B.shape[0]

            MSEs_split.append(MSE)
            MAEs_split.append(MAE)
            
        # adding differences for each split:
        BLP_metrics_diffs[split] = BLP_metrics_split[0] - BLP_metrics_split[1]
        GATES_metrics_diffs[split] = GATES_metrics_split[0] - GATES_metrics_split[1]
        MAEs_diffs[split] = MAEs_split[0] - MAEs_split[1]
        MSEs_diffs[split] = MSEs_split[0] - MSEs_split[1]
    
    # Output depends on the number of splits (1 or more than 1)
    if n_splits == 1:
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics_diffs, 3), 
                                   'GATES_metrics': np.round(GATES_metrics_diffs, 3), 
                                  'MAEs': np.round(MAEs_diffs, 3), 'MSEs': np.round(MSEs_diffs, 3)})
        print('Metrics table, A model metrics minus B model metrics')
        display(df_metrics)
        
        if extended == True:
            print('EXTENDED OPTION IS AVAILABLE ONLY IF N_SPLITS IS BIGGER THAN 1')

    else: 
        df_metrics = pd.DataFrame({'BLP_metrics': np.round(BLP_metrics_diffs, 3), 
                                   'GATES_metrics': np.round(GATES_metrics_diffs, 3), 
                                  'MAEs': np.round(MAEs_diffs, 3), 'MSEs': np.round(MSEs_diffs, 3)})
        print('Metrics table, A model metrics minus B model metrics')
        display(df_metrics.describe())
        
        if extended == True: 
            Metrics_strings = ['BLP_metrics', 'GATES_metrics', 'MAEs', 'MSEs']
            for num, metric in enumerate([BLP_metrics_diffs, GATES_metrics_diffs, MAEs_diffs, MSEs_diffs]): 
                bins_num = int(np.sqrt(n_splits))
                plt.hist(metric, bins = bins_num)
                plt.title(Metrics_strings[num], fontsize = 25)
                plt.show()


    return df_metrics