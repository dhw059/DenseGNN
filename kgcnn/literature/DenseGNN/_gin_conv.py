import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazyAdd, Activation
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherEmbeddingSelection
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAverage
from kgcnn.layers.mlp import GraphMLP, MLP



@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GINE')
class GINE(GraphBaseLayer):
    r"""Convolutional unit of `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`_ .

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
                 epsilon_learnable=False,
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
        ed = self.layer_gather([node, edge_index], **kwargs) # GatherNodesOutgoing
        ed = self.layer_add([ed, edges])   # this layer_add is message functions,include add and activation 
        ed = self.layer_act(ed) # message  
        nu = self.layer_pool([node, ed, edge_index], **kwargs)  # aggregate for each node connection
        no = (1+self.eps_k)*node # 可学习的非线性函数来更新节点特征
        out = self.layer_add([no, nu], **kwargs)
        return out,ed

    def get_config(self):
        """Update config."""
        config = super(GINE, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        conf_act = self.layer_act.get_config()
        config.update({"activation": conf_act["activation"],
                       "activity_regularizer": conf_act["activity_regularizer"]})
        return config



@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GINE')
class GINELITE(GraphBaseLayer):
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
                 pooling_method='mean',
                 activation="swish",
                 activity_regularizer=None,  
                 edge_mlp_args: dict = None,
                concat_args: dict = None,
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
        super(GINELITE, self).__init__(**kwargs)
        self.pooling_method = pooling_method

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        self.layer_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.layer_act = Activation(activation=activation, activity_regularizer=activity_regularizer)
        self.layer_add = LazyAdd()

        self.ed_concat = LazyConcatenate(**concat_args)
        self.edge_trans = GraphMLP(**edge_mlp_args)
        self.layer_gather_e_out = GatherNodesOutgoing()
        self.layer_gather_e_in = GatherNodesIngoing()

   
    def build(self, input_shape):
        """Build layer."""
        super(GINELITE, self).build(input_shape)

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
        
        ed_out = self.layer_gather_e_out([node, edge_index], **kwargs)
        ed_in = self.layer_gather_e_in([node, edge_index], **kwargs)
        ed = self.ed_concat([ed_in, ed_out, edges])  
        ed = self.edge_trans(ed)  

        # ed = self.layer_gather([node, edge_index], **kwargs) 
        # ed = self.ed_concat([ed, edges])
        # ed = self.edge_trans(ed)  
        # ed = self.layer_act (ed) 

        nu = self.layer_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection 
        return nu,ed 


    def get_config(self):
        """Update config."""
        config = super(GINELITE, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       })
        conf_act = self.layer_act.get_config()
        config.update({"activation": conf_act["activation"],
                       "activity_regularizer": conf_act["activity_regularizer"]})
        return config


