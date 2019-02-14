import tensorflow as tf


def get_latent_factor(name, embedding_size, total_items, tensor_id):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name='embedding',
                                    shape=(total_items, embedding_size),
                                    trainable=True,
                                    initializer=initializer)

        item_vectors = tf.nn.embedding_lookup(embedding, tensor_id)
    return embedding, item_vectors


def get_MultiLayerFC(name, dim_item_embed, total_items, tensor_in_tensor):
    tensors = dict()

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        _in_fc1 = tf.nn.relu(tensor_in_tensor)

        _mat_fc1 = tf.get_variable('FC_1',
                                   shape=(_in_fc1.shape[1], dim_item_embed),
                                   trainable=True,
                                   initializer=tf.contrib.layers.xavier_initializer())

        _bias_fc1 = tf.get_variable('bias_1', shape=(dim_item_embed,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        _logits_fc1 = tf.matmul(_in_fc1, _mat_fc1) + _bias_fc1


        _out_fc1 = tf.nn.relu(_logits_fc1)

        _mat_fc2 = tf.get_variable('FC_2',
                                   shape=(_out_fc1.shape[1], dim_item_embed),
                                   trainable=True,
                                   initializer=tf.contrib.layers.xavier_initializer())

        _bias_fc2 = tf.get_variable('bias_2', shape=(dim_item_embed,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        _logits_fc2 = tf.matmul(_out_fc1, _mat_fc2) + _bias_fc2

        _out_fc2 = tf.nn.relu(_logits_fc2) + _out_fc1

        _mat_fc3 = tf.get_variable('FC_3',
                                   shape=(_out_fc1.shape[1], dim_item_embed),
                                   trainable=True,
                                   initializer=tf.contrib.layers.xavier_initializer())

        _bias_fc3 = tf.get_variable('bias_3', shape=(dim_item_embed,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        _logits_fc3 = tf.matmul(_out_fc2, _mat_fc3) + _bias_fc3

        _user_embedding = tf.nn.relu(_logits_fc3) + _out_fc1 + _out_fc2

        _mat_fc_last = tf.get_variable('item_embedding',
                                   shape=(dim_item_embed, total_items),
                                   trainable=True,
                                   initializer=tf.contrib.layers.xavier_initializer())

        _logits = tf.matmul(_user_embedding, _mat_fc_last)

        _item_embedding = tf.transpose(_mat_fc_last)

        tensors['logits'] = _logits
        tensors['item_embedding'] = _item_embedding
        tensors['user_embedding'] = _user_embedding

        tf.summary.histogram('logits', _logits)

        return tensors


def get_mlp_softmax(name, tensor_item_vectors, tensor_label, tensor_seq_len, max_seq_len, dim_item_embed,
                    total_items, train):
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

        # average item vectors user interacted with
        seq_mask = tf.sequence_mask(tensor_seq_len, max_seq_len, dtype=tf.float32)
        seq_vec = tf.reduce_mean(tensor_item_vectors * tf.expand_dims(seq_mask, axis=2), axis=1)

        in_tensor = tf.concat(values=[seq_vec], axis=1)

        tensors = get_MultiLayerFC(name='mlp',
                                   dim_item_embed=dim_item_embed,
                                   total_items=total_items,
                                   tensor_in_tensor=in_tensor)

        if train:
            _losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tensor_label, logits=tensors['logits'])
            tensors['losses'] = _losses

        return tensors


class RecModel(object):

    def __init__(self):
        self._train_graph = tf.Graph()
        self._serv_graph = tf.Graph()

    def build_train_model(self, batch_size, dim_item_embed, total_items, max_seq_len):
        """ build train model"""

        with self._train_graph.as_default():
            seq_item_id = tf.placeholder(tf.int32, shape=(batch_size, max_seq_len), name='seq_item_id')
            seq_len = tf.placeholder(tf.int32, shape=(batch_size,), name='seq_len')
            label = tf.placeholder(tf.int32, shape=(batch_size,), name='label')

            _, item_vectors = get_latent_factor(name='latent_factor',
                                                embedding_size=dim_item_embed,
                                                total_items=total_items,
                                                tensor_id=seq_item_id)

            tensors = get_mlp_softmax(name='mlp_softmax',
                                      tensor_item_vectors=item_vectors,
                                      tensor_label=label,
                                      tensor_seq_len=seq_len,
                                      max_seq_len=max_seq_len,
                                      dim_item_embed=dim_item_embed,
                                      total_items=total_items,
                                      train=True)

            tensors['seq_item_id'] = seq_item_id
            tensors['seq_len'] = seq_len
            tensors['label'] = label

            loss_mean = tf.reduce_mean(tensors['losses'])

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            backprop = optimizer.minimize(loss_mean)

            tf.summary.histogram('losses', tensors['losses'])
            summary = tf.summary.merge_all()

            tensors['loss'] = loss_mean
            tensors['backprop'] = backprop
            tensors['summary'] = summary

            return tensors

    def build_serve_model(self, dim_item_embed, total_items, max_seq_len):
        """ build model for serving and evaluation"""

        with self._serv_graph.as_default():
            seq_item_id = tf.placeholder(tf.int32, shape=(None, max_seq_len), name='seq_item_id')
            seq_len = tf.placeholder(tf.int32, shape=(None,), name='seq_len')

            _, item_vectors = get_latent_factor(name='latent_factor',
                                           embedding_size=dim_item_embed,
                                           total_items=total_items,
                                           tensor_id=seq_item_id)

            tensors = get_mlp_softmax(name='mlp_softmax',
                                      tensor_item_vectors=item_vectors,
                                      tensor_label=None,
                                      tensor_seq_len=seq_len,
                                      max_seq_len=max_seq_len,
                                      dim_item_embed=dim_item_embed,
                                      total_items=total_items,
                                      train=False)

            tensors['item_vectors'] = item_vectors
            tensors['seq_item_id'] = seq_item_id
            tensors['seq_len'] = seq_len

            return tensors

    def get_train_graph(self):
        return self._train_graph

    def get_serve_graph(self):
        return self._serv_graph
