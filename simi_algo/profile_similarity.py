def profile_similarity_prediction(user_id, all_items_df, bow_df, rating_df):
    # User-rated-items sparse matrix
    rating_sparse_df = rating_df[rating_df['user'] == user_id].pivot(
        index='user', columns='item', values='rating').fillna(0)
    rated_item_ids = sorted(
        rating_sparse_df.columns.values.tolist())
    rating_sparse_df = rating_sparse_df.reindex(
        rated_item_ids, axis=1)

    # Item-genre sparse matrix
    all_items_genre_df = bow_df.pivot(index='doc_id', columns='token', values='bow').fillna(
        0).rename_axis(index=None, columns=None)
    all_items_genre_df = (all_items_genre_df > 0).astype(int)

    # Select item-genres that are rated by user
    rated_item_genre_df = all_items_genre_df.loc[rated_item_ids]
    rated_item_genre_df.sort_index(inplace=True)

    # User profile, interest vector
    user_profile_vector = rating_sparse_df.dot(
        rated_item_genre_df)

    # Find unseen item-genres
    unseen_items_df = all_items_df[~all_items_df['COURSE_ID'].isin(
        rated_item_ids)]
    unseen_item_ids = unseen_items_df['COURSE_ID'].values
    unseen_item_genre_df = all_items_genre_df.loc[unseen_item_ids]

    unseen_item_scores_df = unseen_item_genre_df.dot(
        user_profile_vector.T).T.reset_index(drop=False)
    del unseen_item_ids, unseen_item_genre_df, unseen_items_df, user_profile_vector, rated_item_genre_df, all_items_genre_df, rating_sparse_df

    # Recommend item for specific user the specific item comparing to threshold
    res = dict()
    for unseen_item_id in unseen_item_scores_df.columns:
        if unseen_item_id != 'index':
            score = unseen_item_scores_df[unseen_item_id].values[0]
            res[unseen_item_id] = score
    return {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
