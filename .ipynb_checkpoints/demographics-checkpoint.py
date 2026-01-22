import pandas as pd


# PET_METADATA_PATH = 'All_Preprocessed_PET.csv'
# DEMOGRAPHIC_PATH = 'All_Subjects_Demographic.csv'
# DIAGNOSIS_PATH = 'All_Subjects_DXSUM.csv'

def get_age_and_education(img_ids, demog_path, pet_path):
    '''
    Get the education levels and age of patients at the time the given images were taken
    '''
    edu_and_dob = pd.read_csv(demog_path)[['PTID', 'PTEDUCAT', 'PTGENDER', 'PTDOB']]
    edu_and_dob['PTDOB'] =  pd.to_datetime(edu_and_dob['PTDOB'], format='%m/%Y')
    pet_meta = pd.read_csv(pet_path, parse_dates=['image_date'])
    pet_meta = pet_meta[pet_meta['image_id'].isin(img_ids)]
    
    df = pd.merge(pet_meta, edu_and_dob, left_on='subject_id', right_on='PTID', how='left').drop_duplicates(subset=['image_id'])
    df['age'] = (df['image_date'] - df['PTDOB']).dt.days / 365.25 
    
    return df[['image_id', 'image_date', 'subject_id', 'PTEDUCAT', 'PTGENDER', 'age']]



def create_merged_table(df1, df2, df1_id_col, df2_id_col, df1_date_col, df2_date_col, how='left', suffixes=('_x', '_y')):
    '''
    Merge two DataFrames based on subject ID and closest visit date.
    '''
    def get_closest_date(row):
        # Get the closest date in df2 given a row in df1
        deltas = (df2[df2[df2_id_col] == row[df1_id_col]][df2_date_col] - row[df1_date_col]).abs()
        return deltas.idxmin() if not deltas.isna().all() else pd.NA
    
    deltas = df1.apply(get_closest_date, axis=1)
    data = df1.copy()
    data['closest_ind'] = deltas
    data = pd.merge(data, df2, left_on='closest_ind', right_index=True, how=how, suffixes=suffixes)#.dropna(subset=['closest_ind'])
    return data


def merge_tables(dfs, id_cols, date_cols, keep_cols, how='left'):
    for i in range(len(dfs)):
        if i == 0:
            df = dfs[i]
            cur_cols = keep_cols[i]
        else:
            df = create_merged_table(df, dfs[i], id_cols[0], id_cols[i], date_cols[0], date_cols[i], how=how)
            cur_cols = cur_cols + keep_cols[i]
      
        df = df[cur_cols]
        
    return df