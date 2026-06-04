import numpy as np

def feature_selection(df):

    ## Log Transformation
    log_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']
    for col in log_cols:
        df[col] = np.log1p(df[col])
    
    X = df.drop(['TARGET', 'SK_ID_CURR'],axis=1)
    y = df['TARGET']
    print("\nFinal Shape:")
    print(X.shape)

    ## Remove weak features

    corr = abs(df.corr()['TARGET'])
    remove_cols = corr[corr < 0.005].index
    remove_cols = [col for col in remove_cols if col != 'TARGET']
    df.drop(columns=remove_cols,inplace=True)
    print("\nRemoved Columns:")
    print(len(remove_cols))

    ## Remove Highly correlated features

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))

    drop_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    drop_cols = [col for col in drop_cols if col != 'TARGET']
    df.drop(columns=drop_cols,inplace=True)
    print("\nHighly Correlated Features Removed:")
    print(len(drop_cols))

    return df