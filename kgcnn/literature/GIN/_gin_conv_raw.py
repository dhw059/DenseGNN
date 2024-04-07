import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazyAdd, Activation
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherEmbeddingSelection
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAverage
from kgcnn.layers.aggr import AggregateLocalEdgesAttention

from kgcnn.literature.NMPN._mpnn_conv import MatMulMessages
from kgcnn.literature.HamNet._hamnet_conv import HamNaiveDynMessage
from kgcnn.layers.mlp import GraphMLP, MLP
 

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
                 pooling_method='max',
                 epsilon_learnable=True,
                 activation="swish",
                 activity_regularizer=None,  
    
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

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        self.layer_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.layer_add = LazyAdd()
        self.layer_act = Activation(activation=activation,
                                    activity_regularizer=activity_regularizer)

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

   
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

        node, edge_index, edges = inputs
        ed = self.layer_gather([node, edge_index], **kwargs) 
        ed = self.layer_add([ed, edges])
        ed = self.layer_act (ed) 
        nu = self.layer_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection 
        # no = (1+self.eps_k)*node
        # out = self.layer_add([no, nu], **kwargs)
        return nu,ed


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


