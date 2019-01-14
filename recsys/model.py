import tensorflow as tf


class Model(object):

    def __init__(self):
        pass

    def build_train_model(self):
        seq_item_id = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len), name='seq_item_id')
        seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,), name='seq_len')
        label = tf.placeholder(tf.int32, shape=(self.batch_size,), name='label')

    def build_serv_model(self):
        seq_item_id = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len), name='seq_item_id')
        seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,), name='seq_len')

    def get_latent_factor(self, name, embedding_size, total_items, tensor_id):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embedding',
                                        shape=(embedding_size, total_items),
                                        trainable=True,
                                        initializer=initializer)

            output = tf.nn.embedding_lookup(embedding, id)
        return embedding, output

    def get_mlp_softmax(self, name, tensor_item, tensor_seq_len, max_seq_len, dim_item_embed, total_items):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            # average item vectors user interacted with
            seq_mask = tf.sequence_mask(tensor_seq_len, max_seq_len, dtype=tf.float32)
            item = tf.reduce_mean(tensor_item * tf.expand_dims(seq_mask, axis=2), axis=1)

            in_tensor = tf.concat(values=[item], axis=1)

            _logits = get_MultiLayerFC(name='mlp', dim_item_embed=dim_item_embed, total_items=total_items, tensor_in_tensor=in_tensor)

            # if train:
            #     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            #

    def get_MultiLayerFC(self, name, dim_item_embed, total_items, tensor_in_tensor):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            _in = tf.nn.relu(tensor_in_tensor)

            mat = tf.get_variable('FC_1',
                                  shape=(_in.shape[1], dim_item_embed),
                                  trainable=True,
                                  initializer=tf.contrib.layers.xavier_initializer())

            _bias = tf.get_variable('bias_1', shape=(dim_item_embed,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            _logits = tf.matmul(_in, mat) + _bias

            _out = tf.nn.relu(_logits)

            _out = tf.contrib.layers.batch_norm(_out, fused=True, decay=0.95,
                                                center=True,
                                                scale=True,
                                                is_training=True,
                                                scope="bn_1",
                                                updates_collections=None)

            _in = _out

            mat = tf.get_variable('FC_2',
                                  shape=(_in.shape[1], total_items),
                                  trainable=True,
                                  initializer=tf.contrib.layers.xavier_initializer())

            _bias = tf.get_variable('bias_1', shape=(total_items,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            _logits = tf.matmul(_in, mat) + _bias

            return _logits