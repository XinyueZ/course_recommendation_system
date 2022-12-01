# Create course id to index and index to id mappings
def get_doc_dicts(load_bow_func):
    bow_df = load_bow_func()
    grouped_df = bow_df.groupby(
        ['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def _similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
    return res


def item_similarity_prediction(user_id, threshold, idx_id_dict, id_idx_dict, sim_matrix, users, courses, scores, load_rating_func):
    rating_df = load_rating_func()
    user_rating_df = rating_df[rating_df['user'] == user_id]
    enrolled_course_ids = user_rating_df['item'].to_list()

    res = _similarity_recommendations(
        idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
    for key, score in res.items():
        if score >= threshold:
            users.append(user_id)
            courses.append(key)
            scores.append(score)
