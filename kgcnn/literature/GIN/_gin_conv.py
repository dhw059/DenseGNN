import tensorflow as tf
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherEmbeddingSelection, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAverage, LazyMultiply, LazyAdd
from kgcnn.layers.aggr import AggregateLocalEdgesAttention
from kgcnn.literature.NMPN._mpnn_conv import MatMulMessages
from kgcnn.literature.HamNet._hamnet_conv import HamNaiveDynMessage
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.norm import GraphBatchNormalization

@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GIN')
class GIN(GraphBaseLayer):
    r"""Convolutional unit of `Graph Isomorphism Network from: How Powerful are Graph Neural Networks?
    <https://arxiv.org/abs/1810.00826>`_.

    Computes graph convolution at step :math:`k` for node embeddings :math:`h_\nu` as:

    .. math::
        h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{k-1} + \sum_{u\in N(\nu)}) h_u^{k-1}.

    with optional learnable :math:`\epsilon^{(k)}`

    .. note::
        The non-linear mapping :math:`\phi^{(k)}`, usually an :obj:`MLP`, is not included in this layer.

    """

    def __init__(self,
                 pooling_method='sum',  #注意这里是实现邻居聚合的内射，可以理解为hash()
                 epsilon_learnable=False,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        """
        super(GIN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        # Layers
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_add = LazyAdd()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)  

    def build(self, input_shape):
        """Build layer."""
        super(GIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`

        Returns:
            tf.RaggedTensor: Node embeddings of shape `(batch, [N], F)`
        """
        node, edge_index = inputs
        ed = self.lay_gather([node, edge_index], **kwargs)
        nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        no = (1+self.eps_k)*node # eps_k  是一个可训练的参数，用于调节节点自身嵌入和其邻居嵌入的权重。在代码中，这个参数就是 eps_k。如果 epsilon_learnable 参数为 True，eps_k 就是可训练的，否则它就是一个固定的零。
        out = self.lay_add([no, nu], **kwargs) 
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config




@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GINE')
class GINE(GraphBaseLayer):
    r"""Convolutional unit of `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`_.

    Computes graph convolution with node embeddings :math:`\mathbf{h}` and compared to :obj:`GIN_conv`,
    adds edge embeddings of :math:`\mathbf{e}_{ij}`.

    .. math::
        \mathbf{h}^{\prime}_i = f_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \phi \; ( \mathbf{h}_j + \mathbf{e}_{ij} ) \right),

    with optionally learnable :math:`\epsilon`. The activation :math:`\phi` can be chosen differently
    but defaults to RELU.

    .. note::
        The final non-linear mapping :math:`f_{\mathbf{\Theta}}`, usually an :obj:`MLP`, is not included in this layer.

    """

    def __init__(self,
                 pooling_method='sum',
                 g_pooling_method='mean',
                 epsilon_learnable=True,
                 activation="swish",
                 activity_regularizer=None,  

                 edge_mlp_args: dict = None,
                 concat_args: dict = None,
                 node_mlp_args: dict = None,
                 graph_mlp_args: dict = None,

                 batch_normalization: bool = True,
                 use_bias: bool = True,
                 units: int = 128,
                 eps=1e-7,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
            activation: Activation function, such as `tf.nn.relu`, or string name of
                built-in activation function, such as "relu".
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation"). Default is None.
        """
        super(GINE, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.g_pooling_method = g_pooling_method
        self.epsilon_learnable = epsilon_learnable
        self.eps = eps
        self.batch_normalization = batch_normalization
        self.units = units
        self.use_bias = use_bias

        # Layers
        # self.layer_gather = GatherEmbeddingSelection([0, 1])
        self.layer_gather_e = GatherNodesOutgoing()
        self.layer_gather_e_in = GatherNodesIngoing()
        self.layer_gather_eu = GatherState()
        self.layer_gather_n = GatherState()
        self.layer_add_e = LazyAdd()
        self.layer_add_n = LazyAdd()
        self.layer_add_u = LazyAdd()
        self.layer_act_e = Activation(activation=activation, activity_regularizer=activity_regularizer)
        self.layer_act_n = Activation(activation=activation, activity_regularizer=activity_regularizer)
        self.layer_act_u = Activation(activation=activation, activity_regularizer=activity_regularizer)
        
        # Core component
        self.edge_trans = GraphMLP(**edge_mlp_args)
        self.node_trans = GraphMLP(**node_mlp_args)
        self.graph_trans = GraphMLP(**graph_mlp_args)
        self.node_concat = LazyConcatenate(**concat_args)
        self.ed_concat = LazyConcatenate(**concat_args)
        self.graph_concat = LazyConcatenate(**concat_args)
        self.node_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.graph_pool_e = PoolingGlobalEdges(pooling_method=self.g_pooling_method)
        self.graph_pool_n = PoolingNodes(pooling_method=self.g_pooling_method)

        # schnet
        self.lazy_multiply = LazyMultiply()
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=True)
        
        # Attention V4
        self.lay_alpha = Dense(1, activation="relu", use_bias=use_bias)
        self.lay_pool_attention = AggregateLocalEdgesAttention()
        self.lay_linear_trafo = Dense(units, activation="relu", use_bias=use_bias)
        self.lay_gather_n = GatherNodesOutgoing()

        # V2
        if batch_normalization:
            self.batch_norm_f = GraphBatchNormalization()
            self.batch_norm_s = GraphBatchNormalization()
        self.f = GraphMLP(**edge_mlp_args)
        self.s = GraphMLP(**edge_mlp_args)
        self.activation_f_layer = Activation(activation="sigmoid", activity_regularizer=activity_regularizer)
        self.activation_s_layer = Activation(activation="softplus", activity_regularizer=activity_regularizer)
        self.lazy_mult = LazyMultiply()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)
    @staticmethod
    def get_features(x):
        """Getter for edge/node/graph features.

        If the argument is a Tensor it is returned as it is.
        If the argument is a dict the value for the "features" key is returned.
        """
        if isinstance(x, dict):
            assert "features" in x.keys()
            return x["features"]
        else:
            return x

    def build(self, input_shape):
        """Build layer."""
        super(GINE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edge_index, edges]

                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`
                - edges (tf.RaggedTensor): Edge embeddings for index tensor of shape `(batch, [M], F)`

        Returns:
            tf.RaggedTensor: Node embeddings of shape `(batch, [N], F)`
        """

        #  GIN preprocess
        node, edge_index, edges, env_input = inputs
        ed  = self.layer_gather_e([node, edge_index], **kwargs)
        ed_in = self.layer_gather_e_in([node, edge_index], **kwargs)
        e_u = self.layer_gather_eu([env_input, edges], **kwargs)

        # edges operate
        # ed_out = self.ed_concat([ed, ed_in, e_u])
        # ed_out = self.lay_dense1(ed_out)
        # x = self.edge_trans(edges)
        # ed = self.lazy_multiply([ed_out, x], **kwargs)                                   
        # ed = self.layer_add_e([ed, edges])                         
        # ed = self.layer_act_e(ed)         
                          
        ed = self.ed_concat([ed_in, ed, edges, e_u])                # edges aggr
        ed = self.edge_trans(ed)                                    # edges transform
        ed = self.layer_add_e([ed,edges])                          # edges update
        ed = self.layer_act_e(ed)                                   # edges connection layers 

        # nodes operate
        # w_n = self.lay_linear_trafo(node, **kwargs) 
        # wn_out = self.lay_gather_n([w_n, edge_index], **kwargs)
        # a_ij = self.lay_alpha(ed, **kwargs) 
        # nu = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        
        nu = self.node_pool([node, ed, edge_index], **kwargs)       # nodes aggr
        v_u = self.layer_gather_n([env_input, node], **kwargs)      # nodes aggr
        nu = self.node_concat([node, nu, v_u])                      # nodes aggr
        nu = self.node_trans(nu)                                    # nodes transform 
        nu = self.layer_add_n([nu, node])                 # nodes updates
        nd = self.layer_act_n(nu)                                   # nodes connection layers
        # nu = self.lazy_multiply([nu, multiplicity])

        # graph operate
        es = self.graph_pool_e(ed, **kwargs)                        # graphs aggr
        vs = self.graph_pool_n(nd, **kwargs)                        # graphs aggr
        ub = self.graph_concat([ es, vs, env_input])                 # graphs aggr
        ub = self.graph_trans(ub)                                   # graphs transform
        ub = self.layer_add_u([ub, env_input])                      # graphs updates
        ud = self.layer_act_u(ub)                                   # graphs connection layers

        return nd,ed,ud
    
    def get_config(self):
        """Update config."""
        config = super(GINE, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        conf_act = self.layer_act_n.get_config()
        config.update({"activation": conf_act["activation"],
                       "activity_regularizer": conf_act["activity_regularizer"]})
        return config
