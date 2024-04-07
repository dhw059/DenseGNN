import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import Dense, LazyConcatenate,LazyMultiply
from kgcnn.layers.gather import GatherNodes, GatherState
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.literature.Megnet.schnet_conv import SchNetCFconv
from kgcnn.layers.modules import LazyAdd, Activation
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing

from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.norm import GraphLayerNormalization


ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='MEGnetBlock')
class MEGnetBlock(GraphBaseLayer):
    r"""Convolutional unit of `MegNet <https://github.com/materialsvirtuallab/megnet>`_ called MegNet Block.

    Args:
        node_embed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        edge_embed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        env_embed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is 'kgcnn>softplus2'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, node_embed=None,
                 edge_embed=None,
                 env_embed=None,
                 pooling_method="mean",
                 use_bias=True,
                 activation='kgcnn>softplus2',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', eps=1e-7,
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        if node_embed is None:
            node_embed = [16, 16, 16]
        if env_embed is None:
            env_embed = [16, 16, 16]
        if edge_embed is None:
            edge_embed = [16, 16, 16]
          
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.env_embed = env_embed
        self.use_bias = use_bias
        self.eps = eps
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}

        # Node
        self.lay_phi_n = Dense(units=self.node_embed[0], activation=activation, **kernel_args)
        self.lay_phi_n_1 = Dense(units=self.node_embed[1], activation=activation, **kernel_args)
        self.lay_phi_n_2 = Dense(units=self.node_embed[2], activation='linear', **kernel_args)
        self.lay_esum = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_gather_un = GatherState()
        self.lay_conc_nu = LazyConcatenate(axis=-1)

        # Edge
        self.lay_phi_e = Dense(units=self.edge_embed[0], activation=activation, **kernel_args)
        self.lay_phi_e_1 = Dense(units=self.edge_embed[1], activation=activation, **kernel_args)
        self.lay_phi_e_2 = Dense(units=self.edge_embed[2], activation='linear', **kernel_args)
        self.lay_gather_n = GatherNodesOutgoing()
        self.lay_gather_ue = GatherState()
        self.lay_conc_enu = LazyConcatenate(axis=-1)

        # Environment
        self.lay_usum_e = PoolingGlobalEdges(pooling_method=self.pooling_method)
        self.lay_usum_n = PoolingNodes(pooling_method=self.pooling_method)
        self.lay_conc_u = LazyConcatenate(axis=-1)
        self.lay_phi_u = ks.layers.Dense(units=self.env_embed[0], activation=activation, **kernel_args)
        self.lay_phi_u_1 = ks.layers.Dense(units=self.env_embed[1], activation=activation, **kernel_args)
        self.lay_phi_u_2 = ks.layers.Dense(units=self.env_embed[2], activation='linear', **kernel_args)

        # Schnet layser
        # conv_args = {"units": self.edge_embed[0], "activation": activation}
        # self.lay_cfconv = SchNetCFconv(**conv_args, **kernel_args)
        # self.lay_dense1 = Dense(units= self.node_embed[0], activation='linear' ,**kernel_args)
        self.lay_mult_e = LazyMultiply()
        self.lay_mult_n = LazyMultiply()
        self.lay_mult_u = LazyMultiply()

        # gin
        self.layer_add_e = LazyAdd()
        self.layer_add_n = LazyAdd()
        self.layer_add_u = LazyAdd()
        self.layer_act_e = Activation(activation="relu", activity_regularizer=None)
        self.layer_act_n = Activation(activation="relu", activity_regularizer=None)
        self.layer_act_u = Activation(activation="relu", activity_regularizer=None)
        self.eps_k_n = self.add_weight(name="epsilon_k", trainable=True,
                                     initializer="zeros", dtype=self.dtype)
        self.eps_k_u = self.add_weight(name="epsilon_k", trainable=True,
                                     initializer="zeros", dtype=self.dtype)
        self.eps_k_e = self.add_weight(name="epsilon_k", trainable=True,
                                     initializer="zeros", dtype=self.dtype)

        # Sage
        node_mlp_args = {"units": [64, 64, 32,32], "use_bias": True, "activation": ["relu", "relu", "relu", "linear"]}
        env_mlp_args = {"units": [64, 64,32,32], "use_bias": True, "activation": ["relu", "relu","relu", "linear"]}
        edge_mlp_args = {"units": [64,64,32,32], "use_bias": True, "activation": ["relu", "relu","relu", "linear"]} 
        self.mlp_e = GraphMLP(**edge_mlp_args)
        self.mlp_n = GraphMLP(**node_mlp_args)
        self.mlp_u = GraphMLP(**env_mlp_args)
        self.norm_n = GraphLayerNormalization()


    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - state (tf.Tensor): State information for the graph, a single tensor of shape (batch, F)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F)
        """
        node_input, edge_input, edge_index_input, env_input = inputs

        # Schnet edge update
        # x = self.lay_dense1(node_input, **kwargs)  
        # ep = self.lay_cfconv([node_input, edge_input, edge_index_input, env_input], **kwargs) 
        # ep = self.lay_phi_e_2(ep, **kwargs)
        

        # Calculate edge Update
        e_n = self.lay_gather_n([node_input, edge_index_input], **kwargs)
        e_u = self.lay_gather_ue([env_input, edge_input], **kwargs)
        ed = self.lay_conc_enu([e_n, edge_input, e_u]) # 
        ed = self.mlp_e(ed) # nonlinear transformation ;like MLP, relu, or cgcnn style;
        # ep = self.lay_mult_e([ed, e_u], **kwargs) 

        # ec = self.lay_conc_enu([e_n, edge_input, e_u], **kwargs)
        # ep = self.lay_phi_e(ec, **kwargs)  # Learning of Update Functions
        # ep = self.lay_phi_e_1(ep, **kwargs)  # Learning of Update Functions
        # ep = self.lay_phi_e_2(ep, **kwargs)  # Learning of Update Functions

        # eo = (1+self.eps_k_e)*edge_input
        ep = self.layer_add_e([edge_input, ed], **kwargs)
        ep = self.layer_act_e(ep) 
        

        # Calculate Node update
        vb = self.lay_esum([node_input, ep, edge_index_input], **kwargs)  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, node_input], **kwargs)
        vc = self.lay_conc_nu([vb, node_input, v_u])
        vc = self.mlp_n(vc)
        # vp = self.lay_mult_n([vc, v_u], **kwargs)
        # vp = self.norm_n(vc)
        # vc = self.lay_conc_nu([vb, node_input, v_u], **kwargs)  # LazyConcatenate node features with new edge updates
        # vp = self.lay_phi_n(vc, **kwargs)  # Learning of Update Functions
        # vp = self.lay_phi_n_1(vp, **kwargs)  # Learning of Update Functions
        # vp = self.lay_phi_n_2(vp, **kwargs)  # Learning of Update Functions
        
        # vo = (1+self.eps_k_n)*node_input
        vp = self.layer_add_n([node_input, vc], **kwargs)
        vp = self.layer_act_n(vp) 
        

        # Calculate environment update
        es = self.lay_usum_e(ep, **kwargs)
        vs = self.lay_usum_n(vp, **kwargs)
        ub = self.lay_conc_u([es, vs, env_input]) 
        up = self.mlp_u(ub) 
        # up = self.lay_mult_u([ub, env_input], **kwargs)

        # ub = self.lay_conc_u([es, vs, env_input], **kwargs)
        # up = self.lay_phi_u(ub, **kwargs)
        # up = self.lay_phi_u_1(up, **kwargs)
        # up = self.lay_phi_u_2(up, **kwargs)  # Learning of Update Functions
        up = self.layer_act_u(up) 
        # uo = (1+self.eps_k_u)*env_input
        up = self.layer_add_u([env_input, up], **kwargs)
        
        return vp, ep,up
  

    def get_config(self):
        config = super(MEGnetBlock, self).get_config()
        config.update({"pooling_method": self.pooling_method, "node_embed": self.node_embed, "use_bias": self.use_bias,
                       "edge_embed": self.edge_embed, "env_embed": self.env_embed})
        config_dense = self.lay_phi_n.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: config_dense[x]})
        return config
