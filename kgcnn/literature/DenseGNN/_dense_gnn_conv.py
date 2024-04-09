import tensorflow as tf
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherEmbeddingSelection, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAverage, LazyMultiply, LazyAdd
from kgcnn.layers.aggr import AggregateLocalEdgesAttention
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.norm import GraphBatchNormalization


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DenseGNN')
class DenseGNN(GraphBaseLayer):
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
                 activation="swish",
                 activity_regularizer=None,  
                 edge_mlp_args: dict = None,
                 concat_args: dict = None,
                 node_mlp_args: dict = None,
                 graph_mlp_args: dict = None,
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
        super(DenseGNN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.g_pooling_method = g_pooling_method

        # Layers
        self.layer_gather_e_out = GatherNodesOutgoing()
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


    def build(self, input_shape):
        """Build layer."""
        super(DenseGNN, self).build(input_shape)

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

        # edges operate
        node, edge_index, edges, env_input = inputs
  
        ed_out = self.layer_gather_e_out([node, edge_index], **kwargs)
        ed_in = self.layer_gather_e_in([node, edge_index], **kwargs)
        e_u = self.layer_gather_eu([env_input, edges], **kwargs) 
        
        ed = self.ed_concat([ed_in, ed_out, edges, e_u])           # edges aggr
        ed = self.edge_trans(ed)                                   # edges transform
        ed = self.layer_add_e([ed, edges])                         # edges update
        ed = self.layer_act_e(ed)                                  # edges connection layers 

        # nodes operate
        nu = self.node_pool([node, ed, edge_index], **kwargs)       # nodes aggr
        v_u = self.layer_gather_n([env_input, node], **kwargs)      # nodes aggr
        nu = self.node_concat([node, nu, v_u])                      # nodes aggr
        nu = self.node_trans(nu)                                    # nodes transform 
        nu = self.layer_add_n([nu, node])                           # nodes updates
        nd = self.layer_act_n(nu)                                   # nodes connection layers

        # graphs operate
        es = self.graph_pool_e(ed, **kwargs)                        # graphs aggr
        vs = self.graph_pool_n(nd, **kwargs)                        # graphs aggr
        ub = self.graph_concat([es, vs, env_input])                 # graphs aggr
        ub = self.graph_trans(ub)                                   # graphs transform
        ub = self.layer_add_u([ub, env_input])                      # graphs updates
        ud = self.layer_act_u(ub)                                   # graphs connection layers

        return nd,ed,ud
    
    def get_config(self):
        """Update config."""
        config = super(DenseGNN, self).get_config()
        config.update({"pooling_method": self.pooling_method,})
        conf_act = self.layer_act_n.get_config()
        config.update({"activation": conf_act["activation"],
                       "activity_regularizer": conf_act["activity_regularizer"]})
        return config
