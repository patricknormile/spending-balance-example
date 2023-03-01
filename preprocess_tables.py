import pandas as pd
import numpy as np
import os
from process_functions import preprocess_df
if __name__ == '__main__' : 
    cur_path = os.path.abspath('.')
    files = [x for x in os.listdir(os.path.join(cur_path,'raw'))
            if ".csv" in x and "NOPE" not in x]

    dfs =  [preprocess_df(pd.read_csv(os.path.join(cur_path,'raw',fl))) 
            for fl in files]
    df = pd.concat(dfs, axis=0)
    df.to_csv(os.path.join(cur_path,'transformed','in_df_transformed_202303.csv'),
              index=False)
