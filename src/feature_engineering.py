import numpy as np

def Features(df):
    df['AGE'] = abs(df['DAYS_BIRTH']) / 365
    df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365
    df['EMPLOYED_BIRTH_RATIO'] = (df['DAYS_EMPLOYED'] /(df['DAYS_BIRTH'] + 1))
    df['GOODS_INCOME_RATIO'] = (df['AMT_GOODS_PRICE'] /(df['AMT_INCOME_TOTAL'] + 1))
    df['EXT_SOURCE_RANGE'] = (df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)-df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1))

    # ratios
    df['CREDIT_INCOME_RATIO'] = (df['AMT_CREDIT'] /(df['AMT_INCOME_TOTAL'] + 1))
    df['ANNUITY_INCOME_RATIO'] = (df['AMT_ANNUITY'] /(df['AMT_INCOME_TOTAL'] + 1))
    df['PAYMENT_RATE'] = (df['AMT_ANNUITY'] /(df['AMT_CREDIT'] + 1))
    df['GOODS_CREDIT_RATIO'] = (df['AMT_GOODS_PRICE'] /(df['AMT_CREDIT'] + 1))

    # ext source features
    ext_cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    df['EXT_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_STD'] = df[ext_cols].std(axis=1)
    df['EXT_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SUM'] = df[ext_cols].sum(axis=1)

    # bureau features
    if 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN' in df.columns:
        df['BUREAU_DEBT_RATIO'] = (df['BUREAU_AMT_CREDIT_SUM_DEBT_MEAN'] /(df['BUREAU_AMT_CREDIT_SUM_MEAN'] + 1))

    if 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN' in df.columns:
        df['BUREAU_OVERDUE_RATIO'] = (df['BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN'] /(df['BUREAU_AMT_CREDIT_SUM_MEAN'] + 1))


    return df

