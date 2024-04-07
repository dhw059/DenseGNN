import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherEmbeddingSelection
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
                #  units, 
                 pooling_method='sum',
                 epsilon_learnable=True,
                 activation="relu",
                 activity_regularizer=None,  

                 edge_mlp_args: dict = None,
                 concat_args: dict = None,
                 node_mlp_args: dict = None,

                 batch_normalization: bool = True,
                 use_bias: bool = True,
                 units: int = 64,
                # node_dim = 64,
                #  use_bias=True,   
                #  kernel_regularizer=None,
                #  bias_regularizer=None,
                #  kernel_initializer='glorot_uniform',
                #  bias_initializer='zeros',kernel_constraint=None,
                #  bias_constraint=None,  activation_context="elu", 
                #  use_dropout= False,  
                #  rate= 0.5,
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
        self.epsilon_learnable = epsilon_learnable
        self.eps = eps
        self.batch_normalization = batch_normalization
        self.units = units

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        # self.layer_gather = GatherEmbeddingSelection([0, 1])
        self.layer_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.layer_add = LazyAdd()
        self.layer_act = Activation(activation=activation,
                                    activity_regularizer=activity_regularizer)
        
        # Core component
        # self.edge_trans = GraphMLP(**edge_mlp_args)
        self.node_concat = LazyConcatenate(**concat_args)
        self.ed_concat = LazyConcatenate(**concat_args)
        self.node_trans = GraphMLP(**node_mlp_args)
        if batch_normalization:
            self.batch_norm_f = GraphBatchNormalization()
            self.batch_norm_s = GraphBatchNormalization()
        # self.f = Dense(self.units, activation="linear", use_bias=use_bias,)
        # self.s = Dense(self.units, activation="linear", use_bias=use_bias, )
        self.f = GraphMLP(**edge_mlp_args)
        self.s = GraphMLP(**edge_mlp_args)
        self.activation_f_layer = Activation(activation="sigmoid", activity_regularizer=activity_regularizer)
        self.activation_s_layer = Activation(activation="softplus", activity_regularizer=activity_regularizer)
        self.lazy_mult = LazyMultiply()
        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

        # AttentiveHeadFP
        # self.lay_fc1 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        # self.lay_fc2 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        # self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args)
        # self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        # self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args)
        # self.lay_gather_in = GatherNodesIngoing()
        # self.lay_gather_out = GatherNodesOutgoing()
        # self.lay_concat = LazyConcatenate(axis=-1)
        # self.lay_pool_attention = AggregateLocalEdgesAttention()
        # self.gru = GRUUpdate(node_dim)
        # self.lay_final_activ = Activation(activation=activation_context)
        # kernel_args = {"kernel_regularizer": kernel_regularizer,
        #                "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
        #                "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
        #                "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        
        # self.hamnet = HamNaiveDynMessage(units,units_edge=units_edge,use_dropout=use_dropout,rate=rate)


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
        # node, edge_index, edges, edge_net_in, edge_net_out ,  p, q= inputs 
        node, edge_index, edges = inputs
        ed  = self.layer_gather([node, edge_index], **kwargs)
        # n_in = self.lay_gather_in([node, edge_index], **kwargs)

        # AttentiveHeadFP
        # n_i = self.lay_fc1(n_in, **kwargs)
        # ed_aggr = self.layer_add([n_out, edges], **kwargs)
        # ed_aggr = self.lay_fc2(ed_aggr, **kwargs)
        # wn_out = self.lay_linear_trafo(ed_aggr, **kwargs)
        # e_ij = self.layer_add([n_i, wn_out], **kwargs)
        # a_ij = self.lay_alpha_activation(e_ij, **kwargs)
        # a_ij = self.lay_alpha(a_ij, **kwargs)
        # h_i = self.lay_pool_attention([node, n_out, a_ij, edge_index], **kwargs)


        # GIN
        ed = self.ed_concat([ed, edges])  # add is key point for structure capture
        # ed = self.edge_trans(ed)  
        x_s, x_f = self.s(ed, **kwargs), self.f(ed, **kwargs)
        if self.batch_normalization:
            x_s, x_f = self.batch_norm_s(x_s, **kwargs), self.batch_norm_f(x_f, **kwargs)
        x_s, x_f = self.activation_s_layer(x_s, **kwargs), self.activation_f_layer(x_f, **kwargs)
        ed = self.lazy_mult([x_s, x_f], **kwargs) 

        ed = self.layer_add([ed, edges])
        ed = self.layer_act (ed) 
        nu = self.layer_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection 
        nu = self.node_concat([node, nu])
        nu = self.node_trans(nu) 
        # no = (1+self.eps_k)*node
        out = self.layer_add([node, nu], **kwargs)
        # out = self.layer_act (out)
   


        #NMPN
        # n_in = self.lay_gather_in([node, edge_index], **kwargs)
        # n_out = self.lay_gather_out([node, edge_index], **kwargs)
        # m_in = MatMulMessages()([edge_net_in, n_in])
        # m_out = MatMulMessages()([edge_net_out, n_out])
        # eu = self.layer_add([m_in, m_out])
        # eu = self.lay_alpha_activation(eu)
        # eu = self.layer_pool([node, eu, edge_index]) 

        # Hamnet
        # me_ftr = self.hamnet([n_in, n_out, edges, p, q, edge_index])
        

        # no = (1+self.eps_k)*node
        # out = self.layer_add([a_ij, ed_act, eu], **kwargs)
        # eu = self.layer_pool([node, out, edge_index]) 
        # out = self.gru([node, out]) # (useful)
        
        return out,ed



        # GATv2
        # w_n = self.lay_linear_trafo(node, **kwargs)
        # n_in = self.lay_gather_in([node, edge_index], **kwargs)
        # n_out = self.lay_gather_out([node, edge_index], **kwargs)
        # wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
        # if self.use_edge_features:
        #     e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
        # else:
        #     e_ij = self.lay_concat([n_in, n_out], **kwargs)
        # a_ij = self.lay_alpha_activation(e_ij, **kwargs)  # 这里使用了activation 
        # a_ij = self.lay_alpha(a_ij, **kwargs) # out=1
        # h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        # if self.use_final_activation:
        #     h_i = self.lay_final_activ(h_i, **kwargs)



        # AttentiveHeadFP
        #  n_in = self.lay_gather_in([node, edge_index], **kwargs)
        #     n_out = self.lay_gather_out([node, edge_index], **kwargs)
        #     n_in = self.lay_fc1(n_in, **kwargs)
        #     n_out = self.lay_concat_edge([n_out, edge], **kwargs)
        #     n_out = self.lay_fc2(n_out, **kwargs)
      
        # wn_out = self.lay_linear_trafo(n_out, **kwargs)
        # e_ij = self.lay_concat([n_in, n_out], **kwargs)
        # e_ij = self.lay_alpha_activation(e_ij, **kwargs)
        # a_ij = self.lay_alpha(e_ij, **kwargs) 
        # n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        # out = self.lay_final_activ(n_i, **kwargs)
        
        # nu =   self.layer_add([nu, h_i]) 
        # nu = MatMulMessages()([nu, h_i])


    def get_config(self):
        """Update config."""
        config = super(GINE, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        conf_act = self.layer_act.get_config()
        config.update({"activation": conf_act["activation"],
                       "activity_regularizer": conf_act["activity_regularizer"]})
        return config



        ## 方案一
        # node, edge_index, edges = inputs
        # n_in = self.lay_gather_in([node, edge_index], **kwargs)
        # ed = self.lay_gather_out([node, edge_index], **kwargs)
        # # AttentiveHeadFP
        # n_in = self.lay_fc1(n_in, **kwargs)
        # ed_aggr = self.layer_add([ed, edges], **kwargs)
        # n_out = self.lay_fc2(ed_aggr, **kwargs)
        # wn_out = self.lay_linear_trafo(n_out, **kwargs)
        # e_ij = self.layer_add([n_in, wn_out], **kwargs)
        # a_ij = self.lay_alpha_activation(e_ij, **kwargs)
        # a_ij = self.lay_alpha(a_ij, **kwargs)
        # h_i = self.lay_pool_attention([node, n_out, a_ij, edge_index], **kwargs)
        # # GIN
        # ed = self.lay_gather_out([h_i, edge_index], **kwargs)
        # ed_aggr = self.layer_add([ed, edges])  # add is key point for structure capture
        # ed_act = self.layer_act(ed_aggr)
        # nu = self.layer_pool([node, ed_act, edge_index], **kwargs)  # Summing for each node connection
        # no = (1+self.eps_k)*node
        # out = self.layer_add([no, nu], **kwargs)




        # 方案二
