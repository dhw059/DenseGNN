import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.modules import LazyConcatenate, Dense, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.modules import LazyAdd, Activation
from kgcnn.layers.base import GraphBaseLayer

ks = tf.keras


# ks = tf.keras
# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of INorp in `tf.keras` from paper:
# 'Interaction Networks for Learning about Objects, Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://arxiv.org/abs/1612.00222
# https://github.com/higgsfield/interaction_network_pytorch

model_default = {'name': "INorp",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                            {'shape': [], 'name': "graph_attributes", 'dtype': 'float32', 'ragged': False}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 5, "output_dim": 64},
                                     "graph": {"input_dim": 100, "output_dim": 64}},
                 'set2set_args': {"channels": 32, "T": 3, "pooling_method": "mean",
                                  "init_qstar": "mean"},
                 'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                 'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                   "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                 'pooling_args': {'pooling_method': "segment_mean"},
                 'depth': 3, 'use_set2set': False, 'verbose': 10,
                 'gather_args': {},
                 'output_embedding': 'graph', "output_to_tensor": True,

                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['relu', 'relu', 'sigmoid']},
                
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": True, "normalization_technique": "graph_batch"},

                # "output_mlp": {"activation": "linear", "units": 1},
                # "last_mlp": {"use_bias": [True, True, True], "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
                # "dropout": 0.1, 
                 }


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None, activation='relu',activity_regularizer=None,
               gather_args: dict = None,
               edge_mlp_args: dict = None,
               node_mlp_args: dict = None,
               set2set_args: dict = None,
               pooling_args: dict = None,
               use_set2set: dict = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               last_mlp: dict = None, dropout: float = None,
               output_mlp: dict = None,  gin_mlp: dict = None,

               ):
    r"""Make `INorp <https://arxiv.org/abs/1612.00222>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.INorp.model_default` .

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, state_attributes]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gather_args (dict): Dictionary of layer arguments unpacked in :obj:`GatherNodes` layer.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge updates.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for node updates.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2SetEncoder` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`AggregateLocalEdges`, :obj:`PoolingNodes`
            layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3])

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    
    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                  use_embedding=len(inputs[3]['shape']) < 1)(env_input)

    edi = edge_index_input

    # Model
    ev = GatherState(**gather_args)([uenv, n]) # node level 
    
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    ed = Dense(n_units, use_bias=True, activation='linear')(ed)

    # list_embeddings = [n]
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = LazyConcatenate(axis=-1)([eu2, eu1])
        # print(eu2.shape, ed.shape)  # (None, None, 64) (None, None, 25)
        eu = LazyConcatenate()([upd, ed])  # -------
        eu = GraphMLP(**edge_mlp_args)(eu)
        # eu = Activation(activation=activation,
        #                             activity_regularizer=activity_regularizer)(eu)  #############################3
        # Pool message
        nu = AggregateLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connection  ;node,
        
        # Add environment   #############################3
        # self = GraphBaseLayer()
        # eps_k = self.add_weight(name="epsilon_k", trainable=False,
                                    #  initializer="zeros", dtype=None)  
        # no = (1+eps_k)*n
        # ev0 = (1+eps_k)*ev
        nu = LazyConcatenate(axis=-1)(
            [n, nu, ev])  # LazyConcatenate node features with new edge updates
        # nu = LazyAdd()([n, nu]) 
        n = GraphMLP(**node_mlp_args)(nu) # node level  64 
        # list_embeddings.append(n)
    

     # node, edge_index, edges = inputs
        # ed = self.layer_gather([node, edge_index], **kwargs)
        # ed = self.layer_add([ed, edges])
        # ed = self.layer_act(ed)
        # nu = self.layer_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        # no = (1+self.eps_k)*node
        # out = self.layer_add([no, nu], **kwargs)


        # node, edge_index = inputs
        # ed = self.lay_gather([node, edge_index], **kwargs)
        # nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        # no = (1+self.eps_k)*node # eps_k  是一个可训练的参数，用于调节节点自身嵌入和其邻居嵌入的权重。在代码中，这个参数就是 eps_k。如果 epsilon_learnable 参数为 True，eps_k 就是可训练的，否则它就是一个固定的零。
        # out = self.lay_add([no, nu], **kwargs) 



    # Output embedding choice
    # if output_embedding == "graph":
    #     # out = [PoolingSet2SetEncoder(**set2set_args)(x) for x in list_embeddings]  # will return tensor
    #     out = [PoolingNodes()(x) for x in list_embeddings]
    #     out = [MLP(**last_mlp)(x) for x in out]
    #     out = [ks.layers.Dropout(dropout)(x) for x in out] # [(None, 1)]
    #     out = ks.layers.Add()(out) # (None, 1)
    #     out = MLP(**output_mlp)(out)



    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:  # False
            # output
            n = Dense(set2set_args["channels"], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)(n)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)
        out = MLP(**output_mlp)(out)

    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `INorp`")


    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=out)
    # model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    model.__kgcnn_model_version__ = __model_version__
    return model
