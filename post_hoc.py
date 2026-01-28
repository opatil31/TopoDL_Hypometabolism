import argparse
import itertools
import pandas as pd
import numpy as np
from scipy import stats

from demographics import create_merged_table, merge_tables, get_age_and_education

# Post-hoc analysis
PET_METADATA_PATH = 'adni-tables/All_Preprocessed_PET.csv'
DEMOGRAPHIC_PATH = 'adni-tables/All_Subjects_Demographic.csv'
DXSUM_PATH = 'adni-tables/All_Subjects_DXSUM.csv'
CDRSB_PATH = 'adni-tables/All_Subjects_CDR.csv'
MMSE_PATH = 'adni-tables/MMSE.csv'
MOCA_PATH = 'adni-tables/MOCA.csv'
AV45_PATH = 'adni-tables/All_Subjects_UCBERKELEY_AMY_6MM.csv'
APOE_PATH = 'adni-tables/All_Subjects_APOERES.csv'
PHC_PATH = 'adni-tables/ADSP_PHC_COGN.csv'
CSF_PATH = 'adni-tables/All_Subjects_UPENNBIOMK_ROCHE_ELECSYS.csv'

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def prettify_col(col):
    col = col.copy().astype(str)
    for var_name in col.index:
        if float(col[var_name]) < 0.001:
            col[var_name] = '<0.001**'
        elif float(col[var_name]) < 0.01:
            col[var_name] = '<0.01**'
        elif float(col[var_name]) < 0.05:
            col[var_name] = '<0.05*'
        else:
            col[var_name] = f'{float(col[var_name]):.4f}'
    return col

def months_to_conversion(cohorts_df, subject_id, mci_baseline=False):
    subject_df = cohorts_df[cohorts_df['PTID'] == subject_id]
#     if 2.0 not in subject_df['DIAGNOSIS'].values:
#         print('not found')
#         return 0
    if 3.0 not in subject_df['DIAGNOSIS'].values:
        return float('inf')
    
    bl_viscode = ('bl' if 'bl' in subject_df['VISCODE2'].values else 'sc')
    bl_date = subject_df[subject_df['VISCODE2'] == bl_viscode]
    if len(bl_date) > 1:
        raise ValueError()
    if mci_baseline and bl_date.iloc[0]['DIAGNOSIS'] != 2.0:
        return -1
    bl_date = bl_date.iloc[0]['EXAMDATE']
    first_ad_date = subject_df[subject_df['DIAGNOSIS'] == 3.0]['EXAMDATE'].min()
    
    return diff_month(first_ad_date, bl_date)

def compute_conversion_times(df):
    '''Get conversion events, but consider all patients that were diagnosed with AD at any point'''
    followup = np.zeros(250) # maximum 250 months of followup 
    n_subjects = 0
    
    for ptid in df['PTID'].unique():
        pt_data = df[df['PTID'] == ptid]
        examdates = pt_data['EXAMDATE']

        if pd.Series([3.0]).isin(pt_data['DIAGNOSIS']).all():
            diff = months_to_conversion(df, ptid)
            if diff > 0:
                followup[diff] += 1
            n_subjects += 1
    return followup, n_subjects

def compute_mci_conversion_times(df):
    '''Get conversion events, but only consider patients that started as MCI and were later diagnosed with AD'''
    followup = np.zeros(250) # maximum 250 months of followup 
    n_subjects = 0
    
    for ptid in df['PTID'].unique():
        pt_data = df[df['PTID'] == ptid]
        examdates = pt_data['EXAMDATE']

        if pd.Series([3.0]).isin(pt_data['DIAGNOSIS']).all():
            diff = months_to_conversion(df, ptid, mci_baseline=True)
            if diff > 0:
                followup[diff] += 1
                n_subjects += 1
    return followup, n_subjects

def km_curve(events, num):
    s_t = 1 # survival probability at time t=0
    vals = [s_t]
    
    for i in range(len(events)):
        if i == 60:
            break
        if i > 24:
            vals.append(s_t)
            continue
        s_t = s_t * (num - events[i]) / num
        num -= events[i]
        vals.append(s_t)
        if num == 0:
            break
    return vals
    

def get_ad_baseline_ids():
    '''Get image IDs for patients diagnosed with AD at baseline'''
    pet = pd.read_csv(PET_METADATA_PATH, parse_dates=['image_date'])
    dxsum = pd.read_csv(DXSUM_PATH, parse_dates=['EXAMDATE'])
    
    df = create_merged_table(pet, dxsum, 'subject_id', 'PTID', 'image_date', 'EXAMDATE', how='inner')
    df = df[((df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc')) & (df['DIAGNOSIS'] == 3.0)]

    return df['image_id'].tolist()


def get_amy_positive_ids():
    amy = pd.read_csv(AV45_PATH, parse_dates=['SCANDATE'])
    pet = pd.read_csv(PET_METADATA_PATH, parse_dates=['image_date'])
    dxsum = pd.read_csv(DXSUM_PATH, parse_dates=['EXAMDATE'])
    
    df = create_merged_table(pet, amy, 'subject_id', 'PTID', 'image_date', 'SCANDATE', how='inner')
    df = create_merged_table(df, dxsum, 'subject_id', 'PTID', 'image_date', 'EXAMDATE', how='inner')
    
    df = df[(df['AMYLOID_STATUS'] == 1.) & (df['DIAGNOSIS'] == 3.0)]
    return df['image_id'].tolist()


def get_posthoc_vars(img_ids, cluster_assignments):
    '''
    Get post-hoc data for the selected variables for patients at the time the given images were taken (img_ids).
    '''
    dfs = [
        get_age_and_education(img_ids, DEMOGRAPHIC_PATH, PET_METADATA_PATH),
        pd.read_csv(CDRSB_PATH, parse_dates=['VISDATE']),
        pd.read_csv(MMSE_PATH, parse_dates=['VISDATE']),
        pd.read_csv(MOCA_PATH, parse_dates=['VISDATE']),
        pd.read_csv(AV45_PATH, parse_dates=['SCANDATE']),
        pd.read_csv(PHC_PATH, parse_dates=['EXAMDATE']),
        pd.read_csv(CSF_PATH, parse_dates=['EXAMDATE'])
    ]
    apoe = pd.read_csv(APOE_PATH)
    
    id_cols = ['subject_id', 'PTID', 'PTID', 'PTID', 'PTID', 'PTID', 'PTID']
    date_cols = ['image_date', 'VISDATE', 'VISDATE', 'VISDATE', 'SCANDATE', 'EXAMDATE', 'EXAMDATE']
    keep_cols = [
        ['image_id', 'subject_id', 'image_date', 'PTEDUCAT', 'PTGENDER', 'age'],
        ['CDRSB'],
        ['MMSCORE'],
        ['MOCA'],
        ['SUMMARY_SUVR'],
        ['PHC_VSP', 'PHC_LAN', 'PHC_MEM', 'PHC_EXF'],
        ['TAU', 'PTAU', 'ABETA42']
    ]
    apoe_cols = ['PTID', 'GENOTYPE']

    df = merge_tables(dfs, id_cols, date_cols, keep_cols)
    df = pd.merge(df, apoe[apoe_cols], how='left', left_on='subject_id', right_on='PTID').drop('PTID', axis=1)
    df['apoe4'] = df['GENOTYPE'].str.contains('4').astype(float)
    df = df.set_index('image_id')
    df['cluster'] = pd.Series(index=img_ids, data=cluster_assignments)

    return df


def run_group_tests(vars_df, var_names, test_names):
    '''Run ANOVA and chi square tests among cluster groups'''
    f_vals, p_vals = [], []
    means = []
    stds = []
    cluster_groups = vars_df.groupby('cluster')
    for var_name, test_name in zip(var_names, test_names):
        groups = [group[var_name].tolist() for clust_id, group in cluster_groups]
        if test_name == 'anova':
            result = stats.f_oneway(*groups, 
                                    nan_policy='omit', 
#                                     equal_var=False
                                   )
            f_vals.append(result.statistic)
            p_vals.append(result.pvalue)
            means.append([np.nanmean(group_data) for group_data in groups])
            stds.append([np.nanstd(group_data) for group_data in groups])
#         elif test_name == 'chisq':
#             pass
        else:
            raise ValueError()

    return f_vals, p_vals, means, stds

def run_pairwise_tests(vars_df, var_names, n_clust=4):
    '''Run pairwise Welch t-test for each pair of subtypes'''
    data = {}
    cluster_groups = vars_df.groupby('cluster')
    for clust_a, clust_b in itertools.combinations(list(range(n_clust)), r=2):
        data[f'{clust_a} vs {clust_b}'] = []
        for var_name in var_names:
            var_a = cluster_groups.get_group(clust_a)[var_name].tolist()
            var_b = cluster_groups.get_group(clust_b)[var_name].tolist()
            
            result = stats.ttest_ind(var_a, var_b, equal_var=False, nan_policy='omit')
            data[f'{clust_a} vs {clust_b}'].append(result.pvalue)
    
    return pd.DataFrame(index=var_names, data=data)


def run_group_conversion_test(vars_df, mci_baseline=False):
#     all_cohorts = pd.read_csv(all_cohorts_path, parse_dates=['EXAMDATE'])
    dxsum = pd.read_csv(DXSUM_PATH, parse_dates=['EXAMDATE'])
    times = []
    means, stds = [], []

    for c in [0,1,2,3]:
        vars_clust = vars_df[vars_df['cluster'] == c]
        df = dxsum[dxsum['PTID'].isin(vars_clust['subject_id'])]
        followup, n_subjects = compute_mci_conversion_times(df) if mci_baseline else compute_conversion_times(df)
#         m_time = np.sum(followup * np.arange(len(followup))) / n_subjects
        vals = km_curve(followup, n_subjects)

        times.append([[i] * int(followup[i]) for i in range(len(followup))])

    times = [list(itertools.chain(*subtype_times)) for subtype_times in times]
    means = [np.mean(x) for x in times]
    stds = [np.std(x) for x in times]
    
    res = stats.f_oneway(*times, nan_policy='omit', equal_var=False)
    
    return res.statistic, res.pvalue, means, stds


def run_pairwise_conversion_tests(vars_df):
    dxsum = pd.read_csv(DXSUM_PATH, parse_dates=['EXAMDATE'])
    data = {}
    
    for clust_a, clust_b in itertools.combinations([0,1,2,3], r=2):
        vars_clust_a = vars_df[vars_df['cluster'] == clust_a]
        vars_clust_b = vars_df[vars_df['cluster'] == clust_b]
        df_a = dxsum[dxsum['PTID'].isin(vars_clust_a['subject_id'])]
        df_b = dxsum[dxsum['PTID'].isin(vars_clust_b['subject_id'])]
        followup_a, n_subjects_a = compute_conversion_times(df_a)
        followup_b, n_subjects_b = compute_conversion_times(df_b)
        
        times_a = list(itertools.chain(*[[i] * int(followup_a[i]) for i in range(len(followup_a))]))
        times_b = list(itertools.chain(*[[i] * int(followup_b[i]) for i in range(len(followup_b))]))
        
        res = stats.ttest_ind(times_a, times_b, equal_var=False, nan_policy='omit')
        data[f'{clust_a} vs {clust_b}'] = [res.pvalue]
        
    return data
#         data[f'{clust_a} vs {clust_b}'] = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subset', type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    var_names = ['PTEDUCAT', 'PTGENDER', 'age', 'CDRSB', 'MMSCORE', 'MOCA', 'SUMMARY_SUVR', 'PHC_VSP', 'PHC_LAN', 'PHC_MEM', 'PHC_EXF', 'TAU', 'PTAU', 'ABETA42']
    test_names = ['anova'] * len(var_names)
    
    if args.subset == 'ad':
        df = pd.read_csv('ad_vars.csv', index_col='image_id')
    elif args.subset == 'ad_baseline':
        df = pd.read_csv('ad_baseline_vars.csv', index_col='image_id')
    elif args.subset == 'ad_amy':
        df = pd.read_csv('ad_amy_vars.csv', index_col='image_id')
    elif args.subset == 'mci':
        df = pd.read_csv('mci_vars.csv', index_col='image_id')
    else:
        raise ValueError()
        
    f_vals, p_vals, means, stds = run_group_tests(df, var_names, test_names)
    data = {'pval': p_vals, '0': [], '1': [], '2': [], '3': []}
    for var_means, var_stds in zip(means, stds):
        for i, (clust_group_mean, clust_group_std) in enumerate(zip(var_means, var_stds)):
            data[str(i)].append(f'{clust_group_mean:.4f} ({clust_group_std:.3f})')
            
    res_df = pd.DataFrame(index=var_names, data=data)
    res_df['pval'] = prettify_col(res_df['pval'])
    
        
    if args.subset != 'mci':
        conv_f, conv_p, means, stds = run_group_conversion_test(df, mci_baseline=True)
        df_conv_group = pd.DataFrame(index=['time (months)'], data={i: f'{means[i]:.3f} ({stds[i]:.3f})'  for i in range(len(means))})
        data_pairwise = run_pairwise_conversion_tests(df)
        df_conv_pairwise = pd.DataFrame(index=['pvalue'], data=data_pairwise)
        df_conv_pairwise['global'] = [conv_p]
#         print(conv_f, conv_p)
        print(df_conv_group)
        print(df_conv_pairwise)
        
        if args.save and args.subset == 'ad':
            df_conv_group.to_csv('ad_conv_group.csv', index=True)
            df_conv_pairwise.to_csv('ad_conv_pairwise.csv', index=True)
            print('Saved conversion times to ad_conv_group.csv and ad_conv_pairwise.csv')
        
    if args.save:
        res_df.to_csv(f'{args.subset}_group_tests.csv', index=True)
        print(f'Saved to {args.subset}_group_tests.csv')