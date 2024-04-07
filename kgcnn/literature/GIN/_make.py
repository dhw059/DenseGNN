import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._gin_conv import GIN, GINE
from kgcnn.layers.mlp import GraphMLP, MLP
from ...layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.literature.NMPN._mpnn_conv import TrafoEdgeNetMessages, MatMulMessages
from kgcnn.layers.modules import LazyConcatenate, LazyAdd, Activation, Dense, Dropout, OptionalInputEmbedding , LazyAverage
from kgcnn.literature.Schnet._schnet_densenet_conv import SchNetInteraction
from kgcnn.literature.HamNet._hamnet_conv import HamNaiveDynMessage, HamNetGRUUnion, HamNetFingerprintGenerator
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GIN in `tf.keras` from paper:
# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826

model_default = {
    "name": "GIN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    "gin_args": {},
    "depth": 3, "dropout": 0.0, "verbose": 10,
    "last_mlp": {"use_bias": [True, True, True], "units": [64, 64, 64],
                 "activation": ["relu", "relu", "linear"]},
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               gin_args: dict = None,
               gin_mlp: dict = None,
               last_mlp: dict = None,
               dropout: float = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GIN <https://arxiv.org/abs/1810.00826>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GIN.model_default`.

    Inputs:
        list: `[node_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in :obj:`GIN` convolutional layer.
        gin_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` for convolutional layer.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        dropout (float): Dropout to use.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    assert len(inputs) == 2
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    # Model
    # Map to the required number of units.
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN(**gin_args)([n, edi])
        n = GraphMLP(**gin_mlp)(n)
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]  # will return tensor
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = MLP(**output_mlp)(out)
        
    elif output_embedding == "node":  # Node labeling
        out = n
        out = GraphMLP(**last_mlp)(out)
        out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GIN`")

    model = ks.models.Model(inputs=[node_input, edge_index_input], outputs=out)
    model.__kgcnn_model_version__ = __model_version__
    return model


model_default_edge = {
    "name": "GIN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    
    "graph_mlp": {"units": [128,128], "use_bias": True, "activation": ["relu", "relu"],
                            "use_normalization": True},

    "gin_args": {"epsilon_learnable": False},
    "depth": 3, "dropout": 0.5, "verbose": 10,
    "pooling_args": {"pooling_method": "mean"},
    "last_mlp": {"use_bias": [True, True, True], "units": [64, 64, 64],
                 "activation": ["relu", "relu", "linear"]},
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"},   
                

}


@update_model_kwargs(model_default_edge)
def make_model_edge(inputs: list = None,
                    input_embedding: dict = None, 
                    depth: int = None,
                    gin_args: dict = None,
                    gin_mlp: dict = None,
                    graph_mlp: dict = None,
                    last_mlp: dict = None,
                    dropout: float = None, 
                    output_embedding: str = None, 
                    output_mlp: dict = None, 
                    output_to_tensor: bool = None,
                    pooling_args: dict = None,

                    name: str = None, verbose: int = None,
                    # interaction_args: dict = None,
                    # attention_args: dict = None,
                    # edge_mlp: dict = None, node_dim: int = None,
                    # given_coordinates: bool = None,   message_kwargs: dict = None,  union_type_edge: str = None,
                    # gru_kwargs: dict = None,  fingerprint_kwargs: dict = None,

                    ):
    r"""Make `GINE <https://arxiv.org/abs/1905.12265>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GIN.model_default_edge`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in :obj:`GIN` convolutional layer.
        gin_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` for convolutional layer.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        dropout (float): Dropout to use.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    # assert len(inputs) == 3
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3]) # charge,graph

    # Make input embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                  use_embedding=len(inputs[3]['shape']) < 1)(env_input) # graph
    
    # Map to the required number of units.
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='relu')(n) # nodes
    ed = Dense(n_units, use_bias=True, activation='relu')(ed) # edges
    ud = Dense(n_units, use_bias=True, activation='relu')(uenv) # graph
    n_raw = n
    ed_raw = ed
    ud_raw = ud

    # list_embeddings = [n]
    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    list_embeddings_u = [ud]
    for i in range(0, depth):
        if i>0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
            ud = GraphMLP(**graph_mlp)(ud)

        np,ep,up = GINE(**gin_args)([n, edi, ed, ud])
        # np,ep = GINE(**gin_args)([n, edi, ed])
        # heads = [GINE(**gin_args)([n, edi, ed, ud]) for _ in range(2)]
        # nk = LazyAverage()([h[0] for h in heads])
        # ek = LazyAverage()([h[1] for h in heads])
        # uk = LazyAverage()([h[2] for h in heads])
        # np = Activation(activation="relu")(nk)
        # ep = Activation(activation="relu")(ek)
        # up = Activation(activation="relu")(uk)

        # n = LazyConcatenate()([n_raw, np])
        # ed = LazyConcatenate()([ed_raw, ep])
        # ud = LazyConcatenate()([ud_raw, up])
        # n_raw = np
        # ed_raw = ep
        # ud_raw = up
        # list_embeddings.append(np)

        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)
        list_embeddings_u.append(up)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        ud = LazyConcatenate()(list_embeddings_u)
       

    ## for e-form and hyper_mp_perovskites readout      out1
    # n = LazyConcatenate()(list_embeddings_n)
    # if output_embedding == 'graph':
    #     out = PoolingNodes(**pooling_args)(n)
    #     out = MLP(**output_mlp)(out)


    # concat readout                                    out2
    n = PoolingNodes(**pooling_args)(n)
    ed = PoolingGlobalEdges(**pooling_args)(ed)
    out = LazyConcatenate()([n, ed, ud])
    out = MLP(**output_mlp)(out)


    # Output embedding choice                           out3
    # if output_embedding == "graph":
    #     out = [PoolingNodes(**pooling_args)(x) for x in list_embeddings]  
    #     out = [ks.layers.Dropout(dropout)(x) for x in out]
    #     out = LazyConcatenate()(out)
    #     out = MLP(**output_mlp)(out)

    # elif output_embedding == "node":  # Node labeling
    #     out = n
    #     out = GraphMLP(**last_mlp)(out)
    #     out = GraphMLP(**output_mlp)(out)
    #     if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
    #         out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    # else:
    #     raise ValueError("Unsupported output embedding for mode `GIN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input,env_input], outputs=out)
    model.__kgcnn_model_version__ = __model_version__
    return model
