import pandas as pd
import numpy as np

def preprocess_df(df) : 
    mica = False
    if ('Card No.' not in df.columns) and ('Transaction Date' not in df.columns) : 
        mica = True
        df['Card No.'] = 8771
    elif ('Card No.' not in df.columns) and ('Transaction Date' in df.columns) :
        df['Card No.'] = 1436
    df = df[df['Category'] != "Payment/Credit"]
    df.columns = [x.lower().replace(" ","_").replace(".","") for x in df.columns]    
    df[['debit','credit']] = df[['debit','credit']].apply(lambda x : x.fillna(0), 
                                axis=1)
    if not mica : 
        df['amount'] = df['debit'].values - df['credit'].values
    else : 
        df['amount'] = df['debit']
    gf = ['posted_date','card_no','description','category']
    sf = ['amount',]

    T_df = (df[gf + sf]
            .groupby(gf)
            .agg({'amount' : np.sum,})
            .reset_index()
        )
    T_df['day'] = pd.to_datetime(T_df['posted_date']).dt.dayofweek
    T_df['description'] = T_df['description'].apply(lambda x: x.lower())
    card_map = {1436:0,8771:1}
    T_df['mica_card'] = T_df['card_no'].replace(card_map)

    return T_df[['description','category','mica_card','amount','day',]]
