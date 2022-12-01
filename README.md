# Recommdation system for courses

Use different algorithms to build a recommendation system for courses.

> Check `simi_algo` folder for the algorithms.

- `cluster_similarity_prediction(user_id, bow_df, rating_df, n_clusters, n_components=None)`
- `item_similarity_prediction(user_id, threshold, idx_id_dict, id_idx_dict, sim_matrix, users, courses, scores, load_rating_func)`
- `profile_similarity_prediction(user_id, all_items_df, bow_df, rating_df)`
- `knn_similarity_prediction(user_id, K_range_tuple, name, user_based, rating_df)`
- `nmf_similarity_prediction(user_id, rating_df, n_factors)`
- `nn_similarity_prediction(nn_model, user_id, rating_df, user_id2idx_dict, item_id2idx_dict)`