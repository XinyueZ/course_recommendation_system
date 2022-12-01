from os import access
import pandas as pd
import tensorflow as tf
from simi_algo.item_similarity import (
    get_doc_dicts, item_similarity_prediction)
from simi_algo.knn_similarity import knn_similarity_prediction
from simi_algo.profile_similarity import profile_similarity_prediction
from simi_algo.nn import get_num_users_items, create_train_val_test_datasets, fit, nn_similarity_prediction
from simi_algo.clustering_similarity import cluster_similarity_prediction
from simi_algo.nmf_similarity import nmf_similarity_prediction

models = (
    "Course Similarity",  # done 0
    "User Profile",  # done 1
    "Clustering",  # done 2
    "Clustering with PCA",  # done 3
    "KNN",  # done 4
    "NMF",  # done 5
    "Neural Network",  # done 6
    # "Regression with Embedding Features",  # 7
    # "Classification with Embedding Features"  # 8
)


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Prediction
def predict(model_name, user_ids, params):
    threshold = 0.6
    top_courses = 10
    if "threshold" in params:
        threshold = params["threshold"]
    if "top_courses" in params:
        top_courses = params["top_courses"]
    idx_id_dict, id_idx_dict = get_doc_dicts(load_bow)
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            threshold = threshold / 100.0
            item_similarity_prediction(
                user_id, threshold, idx_id_dict, id_idx_dict, sim_matrix, users, courses, scores, load_ratings)
        # TODO: Add prediction model code here
        # User Profile model
        if model_name == models[1]:
            threshold = threshold / 100.0
            # Use bows to build course genre profile
            all_items_df = load_courses()
            bow_df = load_bow()
            rating_df = load_ratings()

            res = profile_similarity_prediction(
                user_id, all_items_df, bow_df, rating_df)

            # Top-courses recommendation
            for idx, (item_id, score) in enumerate(res.items()):
                if idx <= top_courses and score >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(score)

            del all_items_df, bow_df, rating_df
        # Clustering model
        elif model_name == models[2]:
            rating_df = load_ratings()
            bow_df = load_bow()
            res = cluster_similarity_prediction(
                user_id, bow_df, rating_df, params["n_clusters"])
            # Top-courses recommendation
            for idx, (item_id, popularity) in enumerate(res.items()):
                if idx <= top_courses and popularity >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(popularity)
            del rating_df, bow_df
        # Clustering with PCA model
        elif model_name == models[3]:
            rating_df = load_ratings()
            bow_df = load_bow()
            res = cluster_similarity_prediction(
                user_id, bow_df, rating_df, params["n_clusters"], params['n_components'])
            # Top-courses recommendation
            for idx, (item_id, popularity) in enumerate(res.items()):
                if idx <= top_courses and popularity >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(popularity)
            del rating_df, bow_df
            pass
        # KNN based collaborative filtering
        elif model_name == models[4]:
            rating_df = load_ratings()
            k_range_tuple = params["k_range_tuple"]
            name = params["name"]
            user_based = params["user_based"]

            res = knn_similarity_prediction(user_id,
                                            k_range_tuple,
                                            name,
                                            user_based,
                                            rating_df)

            # Top-courses recommendation
            for idx, (item_id, est) in enumerate(res.items()):
                if idx <= top_courses and est >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(est)

            del rating_df
        # NMF based collaborative filtering
        elif model_name == models[5]:
            rating_df = load_ratings()
            res = nmf_similarity_prediction(user_id,
                                            rating_df,
                                            params["n_factors"])

            # Top-courses recommendation
            for idx, (item_id, est) in enumerate(res.items()):
                if idx <= top_courses and est >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(est)

            del rating_df
        elif model_name == models[6]:
            # Neural Network based collaborative filtering
            rating_df = load_ratings()

            embedding_size, batch_size, epochs = 32, 2**11, 100
            num_users, num_items = get_num_users_items(rating_df)
            x_train, x_val, y_train, y_val, user_id2idx_dict, item_id2idx_dict = create_train_val_test_datasets(
                rating_df)

            nn_model, _ = fit((x_train, y_train),   (x_val, y_val),
                              num_users, num_items, embedding_size,
                              batch_size, epochs,
                              activation=params["final_activation"])
            nn_model.trainable = False

            res = nn_similarity_prediction(
                nn_model, user_id, rating_df, user_id2idx_dict, item_id2idx_dict)

            # Top-courses recommendation
            for idx, (item_id, pred) in enumerate(res.items()):
                if idx <= top_courses and pred >= threshold:
                    users.append(user_id)
                    courses.append(item_id)
                    scores.append(pred)

            del rating_df
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
