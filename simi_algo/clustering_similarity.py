import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_similarity_prediction(user_id, bow_df, rating_df, n_clusters, n_components=None):
    # User-rated-items sparse matrix
    bow_sparse_df = bow_df.pivot(index='doc_id', columns='token', values='bow').fillna(
        0).rename_axis(index=None, columns=None).reset_index().rename(columns={'index': 'doc_id'})

    # User profile with item features
    all_users_profile_df = rating_df.merge(
        bow_sparse_df, left_on='item', right_on="doc_id").drop(columns=['doc_id', 'item', 'rating'])
    all_users_feature_df = all_users_profile_df.groupby(
        'user_x').sum().reset_index()

    # Clustering based on user profile (feature vector)
    features = all_users_feature_df.iloc[:, 1:].values

    # PCA if possible
    if n_components is not None:
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)

    labels = KMeans(n_clusters=n_clusters,
                    random_state=23).fit_predict(features)
    label_df = pd.DataFrame(labels, columns=['cluster_group'])

    # Connect user with cluster group
    all_users_cluster_df = pd.concat([all_users_feature_df, label_df], axis=1)
    all_users_cluster_df = all_users_cluster_df[['user_x', 'cluster_group']]
    all_users_cluster_df = all_users_cluster_df.rename(
        columns={'user_x': 'user'})

    # Find user_id's user cluster group
    cluster_group = all_users_cluster_df[all_users_cluster_df['user']
                                         == user_id]['cluster_group'].values[0]

    # Connect user rated item to user and cluster group
    all_users_item_and_cluster_df = rating_df.merge(
        all_users_cluster_df, left_on='user', right_on="user")
    all_users_item_and_cluster_df = all_users_item_and_cluster_df[[
        'user', 'item', 'rating', 'cluster_group']]

    # Popularity-counting for items in the same cluster group
    rated_items = rating_df[rating_df['user'] == user_id]['item'].values

    # Get popularity of items in the same cluster group
    all_users_item_and_cluster_df['count'] = [
        1] * len(all_users_item_and_cluster_df)
    cluster_item_popularity_df = all_users_item_and_cluster_df[all_users_item_and_cluster_df['cluster_group'] == cluster_group].groupby(
        ['item']).aggregate(popularity=('count', 'sum')).reset_index()
    cluster_item_popularity_df = cluster_item_popularity_df.sort_values(
        by='popularity', ascending=False)

    # Find items of user_id's user for unseen item popularity in the same cluster group
    unseen_item_popularity_df = cluster_item_popularity_df[~cluster_item_popularity_df['item'].isin(
        rated_items)]

    res = dict()
    for _, row in unseen_item_popularity_df.iterrows():
        unseen_item_id = row['item']
        res[unseen_item_id] = row['popularity']

    del bow_sparse_df, all_users_profile_df, label_df, all_users_cluster_df, all_users_item_and_cluster_df, cluster_item_popularity_df, unseen_item_popularity_df
    return {k: v for k, v in sorted(
            res.items(), key=lambda item: item[1], reverse=True)}
