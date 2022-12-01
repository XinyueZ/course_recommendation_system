import pandas as pd
import numpy as np

from surprise import NMF
from surprise import Dataset, Reader


def nmf_similarity_prediction(user_id, rating_df, n_factors):
    reader = Reader(
        line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
    trainset = Dataset.load_from_df(
        rating_df, reader=reader).build_full_trainset()
    nmf = NMF(n_factors=n_factors, random_state=123)
    nmf.fit(trainset)

    user_rating_df = rating_df[rating_df['user'] == user_id]
    user_unseen_item_df = rating_df[~rating_df['item'].isin(
        user_rating_df['item'])]
    user_unseen_item_ids = user_unseen_item_df['item'].to_list()
    # Recommend item for specific user the specific item comparing to threshold
    res = dict()
    for unseen_item_id in user_unseen_item_ids:
        pred = nmf.predict(user_id, unseen_item_id)
        if not pred.details['was_impossible']:
            res[unseen_item_id] = pred.est
        else:
            print("estimate rating", pred.est,
                  ",impossible for", pred.details['reason'])

    del user_rating_df, user_unseen_item_df
    return {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
