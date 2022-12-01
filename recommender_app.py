import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache
def load_ratings():
    return backend.load_ratings()


@st.cache
def load_course_sims():
    return backend.load_course_sims()


@st.cache
def load_courses():
    return backend.load_courses()


@st.cache
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(
        enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=[
                           'COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)

    params['top_courses'] = top_courses
    params['threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    popularity_threshold = st.sidebar.slider('Popularity Threshold',
                                             min_value=1, max_value=300,
                                             value=100, step=1)
    n_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=1, max_value=50,
                                   value=29, step=1)
    params['top_courses'] = top_courses
    params['threshold'] = popularity_threshold
    params['n_clusters'] = n_clusters
# Clustering model with PCA
elif model_selection == backend.models[3]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    popularity_threshold = st.sidebar.slider('Popularity Threshold',
                                             min_value=1, max_value=50,
                                             value=100, step=1)
    n_components = st.sidebar.slider('Number of components',
                                     min_value=1, max_value=50,
                                     value=19, step=1)
    n_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=1, max_value=50,
                                   value=49, step=1)
    params['n_components'] = n_components
    params['top_courses'] = top_courses
    params['threshold'] = popularity_threshold
    params['n_clusters'] = n_clusters
# KNN
elif model_selection == backend.models[4]:
    min_k = st.sidebar.slider('min K',
                              min_value=1, max_value=100,
                              value=1, step=1)

    max_k = st.sidebar.slider('max K',
                              min_value=40, max_value=100,
                              value=40, step=1)

    simi_name = st.sidebar.selectbox(
        "Name:",
        ["cosine",
         "MSD",
         "pearson",
         "pearson_baseline"],
        index=1
    )
    user_based = st.sidebar.selectbox(
        "User based:",
        [True, False],
        index=1
    )

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    min_est_threshold = st.sidebar.slider("Min-Est for recommended",
                                          min_value=0.0, max_value=3.0,
                                          value=2.0, step=.1)
    params['k_range_tuple'] = (min_k, max_k)
    params['name'] = simi_name
    params['user_based'] = user_based
    params['top_courses'] = top_courses
    params['threshold'] = min_est_threshold
# NMF
elif model_selection == backend.models[5]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    min_est_threshold = st.sidebar.slider("Min-Est for recommended",
                                          min_value=0.0, max_value=3.0,
                                          value=2.0, step=.1)
    n_factors = st.sidebar.slider('Number of factors',
                                  min_value=10, max_value=50,
                                  value=15, step=1)
    params['n_factors'] = n_factors
    params['top_courses'] = top_courses
    params['threshold'] = min_est_threshold
# Neural Network
elif model_selection == backend.models[6]:
    activation = st.sidebar.selectbox(
        "Final activation",
        ["tanh", "sigmoid"],
        index=1
    )
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    threshold = st.sidebar.slider('Threshold',
                                  min_value=.1, max_value=1.0,
                                  value=.5, step=.1)
    params['final_activation'] = activation
    params['threshold'] = threshold
    params['top_courses'] = top_courses
else:
    pass


# Prediction
st.sidebar.subheader('3. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=[
                      "COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df.style.format({'SCORE': "{:.3f}"}))
