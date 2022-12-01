import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from keras import backend as K


def f_score(y_true, y_pred, threshold=0.1, beta=2):

    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))


def tp_score(y_true, y_pred, threshold=0.1):

    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(
                K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    return tp


def fp_score(y_true, y_pred, threshold=0.1):

    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(
                K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(
                K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    return fp


def fn_score(y_true, y_pred, threshold=0.1):

    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(
                y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    return fn


def precision_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)

    return tp / (tp + fp)


def recall_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    return tp / (tp + fn)


def process_dataset(raw_data_df):

    encoded_data_df = raw_data_df.copy()

    # Mapping user ids to indices
    user_list = encoded_data_df["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    # Mapping item ids to indices
    item_list = encoded_data_df["item"].unique().tolist()
    item_id2idx_dict = {x: i for i, x in enumerate(item_list)}
    item_idx2id_dict = {i: x for i, x in enumerate(item_list)}

    # Convert original user ids to idx
    encoded_data_df["user"] = encoded_data_df["user"].map(user_id2idx_dict)
    # Convert original item ids to idx
    encoded_data_df["item"] = encoded_data_df["item"].map(item_id2idx_dict)
    # Convert rating to int
    encoded_data_df["rating"] = encoded_data_df["rating"].values.astype("int")

    return encoded_data_df, user_id2idx_dict, item_id2idx_dict


def generate_train_val_datasets(encoded_rating_df, scale=True):
    min_rating = min(encoded_rating_df["rating"])
    max_rating = max(encoded_rating_df["rating"])

    encoded_rating_df = encoded_rating_df.sample(frac=1, random_state=42)
    x = encoded_rating_df[["user", "item"]].values
    if scale:
        y = encoded_rating_df["rating"].apply(lambda x: (
            x - min_rating) / (max_rating - min_rating)).values
    else:
        y = encoded_rating_df["rating"].values

    train_indices = int(0.8 * encoded_rating_df.shape[0])

    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    return x_train, x_val, y_train, y_val


class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, dropout=3.28e-2, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="item_bias")

        self.dot = layers.Dot(axes=(-1, -1))
        self.add = layers.Add()
        self.relu = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout)

        self.predictor = layers.Activation('tanh')

    def call(self, inputs):
        """
           method to be called during model fitting

           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])

        # Compute the item embedding vector
        dot_user_item = self.dot([user_vector, item_vector])
        # Add all the components (including bias)
        x = self.add([dot_user_item, user_bias, item_bias])
        x = self.relu(x)
        # Regularization with dropout
        x = self.dropout(x)

        # Sigmoid output layer to output the probability
        return self.predictor(x)


class RecommenderNet2(keras.Model):

    def __init__(self, num_users, num_items, dropout=3.28e-2, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet2, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="item_bias")

        self.dot = layers.Dot(axes=(-1, -1))
        self.add = layers.Add()
        self.dropout = layers.Dropout(dropout)
        self.bn = layers.BatchNormalization()

        self.predictor = layers.Activation('sigmoid')

    def call(self, inputs):
        """
           method to be called during model fitting

           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])

        # Compute the item embedding vector
        dot_user_item = self.dot([user_vector, item_vector])
        # Add all the components (including bias)
        x = self.add([dot_user_item, user_bias, item_bias])
        # Regularization with dropout
        x = self.dropout(x)
        x = self.bn(x)

        # Sigmoid output layer to output the probability
        return self.predictor(x)


def get_num_users_items(rating_df):
    num_users = len(rating_df["user"].unique())
    num_items = len(rating_df["item"].unique())
    return num_users, num_items


def create_train_val_test_datasets(rating_df):
    encoded_data_df, user_id2idx_dict, item_id2idx_dict = process_dataset(
        rating_df)
    x_train, x_val, y_train, y_val = generate_train_val_datasets(
        encoded_data_df,
        scale=True)
    return x_train, x_val, y_train, y_val, user_id2idx_dict, item_id2idx_dict


def fit(train_tuple, val_tuple,
        num_users, num_items, embedding_size,
        batch_size, epochs, activation="tanh"):
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,
    )

    model = RecommenderNet(num_users,
                           num_items,
                           0.6e-2,
                           embedding_size,
                           name="recommender_net_tanh") if activation == "tanh" else RecommenderNet2(num_users,
                                                                                                     num_items,
                                                                                                     3e-1,
                                                                                                     embedding_size,
                                                                                                     name="recommender_net_sigmoid")
    if activation == "tanh":
        model.compile(
            loss="mse",
            optimizer="adam",
            metrics=[keras.metrics.RootMeanSquaredError()])
    else:
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=[f_score])

    model.trainable = True
    fit = model.fit(x=train_tuple[0], y=train_tuple[1],
                      validation_data=val_tuple,
                      batch_size=batch_size,  # 2**11,
                      epochs=epochs,  # 100,
                      verbose=2,
                      callbacks=[early_stop_callback]
                      )
    return model, fit


def nn_similarity_prediction(nn_model, user_id, rating_df, user_id2idx_dict, item_id2idx_dict):
    user_rating_df = rating_df[rating_df['user'] == user_id]
    rated_item_ids = user_rating_df['item'].to_list()

    unseen_item_ids = list(
        set(rating_df['item'].to_list()) - set(rated_item_ids))

    # Recommend item for specific user the specific item comparing to threshold
    res = dict()
    for unseen_item_id in unseen_item_ids:
        user_idx = user_id2idx_dict[user_id]
        item_idx = item_id2idx_dict[unseen_item_id]
        pred = nn_model(np.array([np.array([user_idx, item_idx])]))
        res[unseen_item_id] = np.around(pred.numpy(), 3)[0][0]

    del unseen_item_ids, user_rating_df, rated_item_ids
    return {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
