import tensorflow as tf


class Model(object):

    def __init__(self, save_model_dir):
        self._train_graph = tf.Graph()
        self._serv_graph = tf.Graph()
        self._save_model_dir = save_model_dir

    def build_train_model(self, batch_size, embedding_size, dim_item_embed, total_items, max_seq_len):
        """ build train model"""

        tensors = dict()
        with self._train_graph.as_default():
            seq_item_id = tf.placeholder(tf.int32, shape=(batch_size, max_seq_len), name='seq_item_id')
            seq_len = tf.placeholder(tf.int32, shape=(batch_size,), name='seq_len')
            label = tf.placeholder(tf.int32, shape=(batch_size,), name='label')

            tensors['seq_item_id'] = seq_item_id
            tensors['seq_len'] = seq_len
            tensors['label'] = label

            _, seq_vec = self.get_latent_factor(name='latent_factor',
                                                embedding_size=embedding_size,
                                                total_items=total_items,
                                                tensor_id=seq_item_id)

            losses = self.get_mlp_softmax(name='mlp',
                                          tensor_seq_vec=seq_vec,
                                          tensor_label=label,
                                          tensor_seq_len=seq_len,
                                          max_seq_len=max_seq_len,
                                          dim_item_embed=dim_item_embed,
                                          total_items=total_items,
                                          train=True)

            tensors['losses'] = losses

            return tensors

    def build_serve_model(self, embedding_size, dim_item_embed, total_items, max_seq_len):
        """ build model for serving and evaluation"""

        tensors = dict()
        with self._serv_graph.as_default():
            seq_item_id = tf.placeholder(tf.int32, shape=(None, max_seq_len), name='seq_item_id')
            seq_len = tf.placeholder(tf.int32, shape=(None,), name='seq_len')

            tensors['seq_item_id'] = seq_item_id
            tensors['seq_len'] = seq_len

            _, seq_vec = self.get_latent_factor(name='latent_factor',
                                                embedding_size=embedding_size,
                                                total_items=total_items,
                                                tensor_id=seq_item_id)

            logits = self.get_mlp_softmax(name='mlp',
                                          tensor_seq_vec=seq_vec,
                                          tensor_label=None,
                                          tensor_seq_len=seq_len,
                                          max_seq_len=max_seq_len,
                                          dim_item_embed=dim_item_embed,
                                          total_items=total_items,
                                          train=False)

            tensors['logits'] = logits

            return tensors

    def get_latent_factor(self, name,
                          embedding_size,
                          total_items,
                          tensor_id):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embedding',
                                        shape=(embedding_size, total_items),
                                        trainable=True,
                                        initializer=initializer)

            seq_vec = tf.nn.embedding_lookup(embedding, tensor_id)
        return embedding, seq_vec

    def get_mlp_softmax(self, name, tensor_seq_vec, tensor_label, tensor_seq_len, max_seq_len, dim_item_embed,
                        total_items, train):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            # average item vectors user interacted with
            seq_mask = tf.sequence_mask(tensor_seq_len, max_seq_len, dtype=tf.float32)
            item = tf.reduce_mean(tensor_seq_vec * tf.expand_dims(seq_mask, axis=2), axis=1)

            in_tensor = tf.concat(values=[item], axis=1)

            _logits = self.get_MultiLayerFC(name='mlp',
                                            dim_item_embed=dim_item_embed,
                                            total_items=total_items,
                                            tensor_in_tensor=in_tensor)

            if train:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tensor_label, logits=_logits)
                losses = tf.reduce_mean(loss)
                # tf.add_to_collection('train', losses)
                return losses
            else:
                # tf.add_to_collection('eval', _logits)
                return _logits

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

            _bias = tf.get_variable('bias_2', shape=(total_items,), trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            _logits = tf.matmul(_in, mat) + _bias

            return _logits

    def get_train_graph(self):
        return self._train_graph

    def get_serve_graph(self):
        return self._serv_graph
