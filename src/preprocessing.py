import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

def preprocess():

    # Data Loading
    print("Loading application_train.csv")
    df = pd.read_csv("data/application_train.csv")
    bureau = pd.read_csv("data/bureau.csv")
    print("Original Shape:",df.shape)
    print(bureau.shape)

    # Bureau Aggregation
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum']
    })

    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper()for col in bureau_agg.columns]
    bureau_agg.reset_index(inplace=True)

    # Merge
    df = df.merge(bureau_agg,on='SK_ID_CURR',how='left')

    print("\nMerged Shape:")
    print(df.shape)

    # Drop Columns
    drop_cols = [
    'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20',
    'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'FLAG_DOCUMENT_8',
    'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12',
    'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'REGION_POPULATION_RELATIVE',
    'FLOORSMAX_MEDI',
    'FLOORSMIN_MEDI',
    'LIVINGAREA_MEDI',
    'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'DAYS_LAST_PHONE_CHANGE',
    'DAYS_ID_PUBLISH',
    'CNT_CHILDREN',
    'BASEMENTAREA_MEDI',
    'YEARS_BUILD_MEDI',
    'COMMONAREA_MEDI',
    'ELEVATORS_MEDI',
    'ENTRANCES_MEDI',
    'APARTMENTS_MEDI',
    'FLAG_MOBIL',
    'YEARS_BEGINEXPLUATATION_MEDI',
    'NONLIVINGAREA_MODE',
    'FLOORSMIN_MODE',
    'ELEVATORS_MODE',
    'ENTRANCES_MODE',
    'FLOORSMAX_MODE',
    'FLAG_EMP_PHONE',
    'LIVINGAPARTMENTS_MEDI',
    'LANDAREA_MEDI',
    'NONLIVINGAPARTMENTS_MODE',
    'LIVINGAPARTMENTS_MODE',
    'LANDAREA_MODE',
    'COMMONAREA_MODE',
    'YEARS_BUILD_MODE',
    'YEARS_BEGINEXPLUATATION_MODE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLOORSMIN_AVG',
    'LANDAREA_AVG',
    'LIVINGAPARTMENTS_AVG',
    'LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG',
    'NONLIVINGAREA_AVG',
    'APARTMENTS_MODE',
    'BASEMENTAREA_MODE',
    'APARTMENTS_AVG',
    'BASEMENTAREA_AVG',
    'YEARS_BEGINEXPLUATATION_AVG',
    'YEARS_BUILD_AVG',
    'COMMONAREA_AVG',
    'ELEVATORS_AVG',
    'ENTRANCES_AVG',
    'FLOORSMAX_AVG',
    'FLAG_PHONE',
    'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY',
    'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY',
    'HOUR_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION',
    'NAME_TYPE_SUITE',
    'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE',
    'WEEKDAY_APPR_PROCESS_START'
]

    drop_cols = [col for col in drop_cols if col in df.columns]

    df.drop(columns=drop_cols,inplace=True)

    print("\nDropped Columns:", len(drop_cols))
    print("Shape After Dropping:", df.shape)


    # Missing Value Handling
    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].fillna(df[col].mode()[0])

        else:

            try:
                df[col] = df[col].fillna(df[col].median())

            except:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Target Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = TargetEncoder(cols=cat_cols)
    df[cat_cols] = encoder.fit_transform(df[cat_cols],df['TARGET'])

    return df


if __name__ == "__main__":
    df = preprocess()

    print("\nPreprocessing Completed")
    print("Final Shape:", df.shape)

    # Save output
    df.to_csv("data/preprocessed_data.csv", index=False)

    print("Saved: data/preprocessed_data.csv")