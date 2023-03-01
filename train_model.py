import pandas as pd
import numpy as np
import os, pickle
from dirty_cat import (SimilarityEncoder, TargetEncoder,
                       MinHashEncoder, GapEncoder)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == '__main__' : 
    cur_path = os.path.abspath('.')
    df_path = os.path.join(cur_path,'transformed','in_df_transformed_202303.csv')
    df = pd.read_csv(df_path)
    df_train, df_test = train_test_split(df, test_size=0.25, 
                                        random_state=3929032919)
    X_train = df_train.drop('shared',axis=1)
    X_test = df_test.drop('shared',axis=1)
    y_train = df_train['shared']
    y_test = df_test['shared']
    one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoders = {
        'one-hot': one_hot,
        'similarity': SimilarityEncoder(),
        'target': TargetEncoder(handle_unknown='ignore'),
        'minhash': MinHashEncoder(n_components=100),
        'gap': GapEncoder(n_components=100),
    }


    dt = DecisionTreeClassifier(
        random_state=20221014,
        min_samples_split=5,
        min_samples_leaf=0.01,
        max_depth=10,   
    )
    lr = LogisticRegression(max_iter=500)
    all_scores = dict()

    for name, method in encoders.items():
        encoder = make_column_transformer(
            (one_hot, ['mica_card', ]),
            ('passthrough', ['amount','day']),
            # Last but not least, our dirty column
            (method, ['description','category']),
            remainder='drop',
        )

        pipeline = make_pipeline(encoder, HistGradientBoostingClassifier())
        scores = cross_val_score(pipeline, X_train, y_train)
        print(f'{name} encoding')
        print(f'score:  mean: {np.mean(scores):.3f}; '
            f'std: {np.std(scores):.3f}\n')
        all_scores[name] = scores

    method = MinHashEncoder(n_components=100)
    encoder = make_column_transformer(
            (one_hot, ['mica_card', ]),
            ('passthrough', ['amount','day']),
            # Last but not least, our dirty column
            (method, ['description','category']),
            remainder='drop',
        )
    pipeline = make_pipeline(encoder, HistGradientBoostingClassifier())
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test,y_predict)}")
    print(confusion_matrix(y_test, y_predict))

    model_path = os.path.join(cur_path,'model','gbct_model_20230228.pkl')
    with open(model_path, 'wb') as file : 
        pickle.dump(pipeline, file, pickle.HIGHEST_PROTOCOL)