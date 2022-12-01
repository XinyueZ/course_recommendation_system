from surprise import KNNBasic
from surprise import Dataset, Reader


def knn_similarity_prediction(user_id, K_range_tuple, name, user_based, rating_df):
    knn = KNNBasic(
        min_k=K_range_tuple[0],
        k=K_range_tuple[1],
        sim_options={"name": name, "user_based": user_based}, verbose=False)
    reader = Reader(
        line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
    trainset = Dataset.load_from_df(
        rating_df, reader=reader).build_full_trainset()
    knn.fit(trainset)

    user_rating_df = rating_df[rating_df['user'] == user_id]
    rated_item_ids = user_rating_df['item'].to_list()

    unseen_item_ids = list(
        set(rating_df['item'].to_list()) - set(rated_item_ids))

    # Recommend item for specific user the specific item comparing to threshold
    res = dict()
    for unseen_item_id in unseen_item_ids:
        pred = knn.predict(user_id, unseen_item_id)
        if not pred.details['was_impossible']:
            res[unseen_item_id] = pred.est
        else:
            print("estimate rating", pred.est,
                  ",impossible for", pred.details['reason'])

    del unseen_item_ids, user_rating_df, rated_item_ids
    return {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
