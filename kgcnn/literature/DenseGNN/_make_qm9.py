import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import NestedGraphNetwork, SequentialGraphNetwork, GraphNetwork, \
    GraphNetworkMultiplicityReadout, CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from tensorflow.keras.layers import GRUCell
from kgcnn.layers.casting import ChangeTensorType
from ._dense_gnn_conv import  DenseGNN
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, Dense, OptionalInputEmbedding,ZerosLike,LazyAdd, LazyMultiply, LazySubtract
from copy import copy

from kgcnn.literature.DenseGNN._schnet_conv import SchNetInteraction
from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice
from ...layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.layers.pooling import PoolingGlobalEdges


from math import inf
from kgcnn.literature.DenseGNN._hamnet_conv import HamNaiveDynMessage, HamNetFingerprintGenerator, HamNetGRUUnion, HamNetNaiveUnion

from kgcnn.literature.DenseGNN._painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.literature.DenseGNN._painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistanceEuclidean, BesselBasisLayer, EdgeDirectionNormalized, CosCutOffEnvelope, PositionEncodingBasisLayer
from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.aggr import AggregateLocalEdges
from tensorflow.keras.layers import Input, Concatenate
from ._gin_conv import  GINELITE

ks = tf.keras


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
       
def update_features(x, v):
        """Setter for edge/node/graph features.

        Args:
            x: Tensor/dict to update
            v: New feature value.

        Returns:
            Updated Tensor or dict.
        """
        if isinstance(x, dict):
            x_ = copy(x)
            x_["features"] = v
            return x_
        else:
            return v

def get_attribute(x, k):
        if isinstance(x, dict):
            assert k in x.keys()
            return x[k]
        else:
            raise ValueError()


def make_qm9(inputs=None,
                name=None,
                input_block_cfg=None,
                output_block_cfg = None,
                input_embedding: dict = None, 
                depth: int = None,
                gin_args: dict = None,
                gin_mlp: dict = None,
                # graph_mlp: dict = None, 
                
                n_units: int = None,
                output_mlp: dict = None, 
                g_pooling_args: dict = None,
              ):
    r"""Make connectivity optimized graph networks for crystals.

    Args:
        inputs (list): List of inputs kwargs.
        input_block_cfg (dict): Input block config.
        processing_blocks_cfg (list): List of processing block configs.
        output_block_cfg: Output block config.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    
    edge_indices = ks.Input(**inputs['range_indices']) # edges,2
    atomic_number = ks.Input(**inputs['node_number']) # nodes,

    offset = ks.Input(**inputs['node_coordinates']) 
    molecule_feature = ks.Input(**inputs['molecule_feature']) 

    pos1, pos2 = NodePosition()([offset, edge_indices])
    ed = NodeDistanceEuclidean()([pos1, pos2])
    ed = EuclideanNorm()(ed) 
    # ed = GaussBasisLayer(**gauss_args)(ed)


    node_input = {'features': atomic_number, 'AGNIFinger': molecule_feature}

    env_input = ks.Input(**inputs['graph_attributes']) 

    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                    use_embedding=len(inputs['graph_attributes']['shape']) < 1)(env_input) # graph
    # global_input = Dense(n_units, use_bias=True, activation='relu')(uenv) # graph

    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    # output_block = GraphNetworkConfigurator.get_gn_block(**output_block_cfg)
    edge_features, node_features, _, _ = crystal_input_block([ed,
                                                              node_input,
                                                            #   global_input,
                                                                None,
                                                              edge_indices])

    n = get_features(node_features)
    ed = edge_features
    # ud = global_features
    edi = edge_indices

    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    for i in range(0, depth):
        if i > 0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
          
        np,ep = GINELITE(**gin_args)([n, edi, ed])        
  
        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)

        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        
    # output
    ed = PoolingGlobalEdges(**g_pooling_args)(ep) # ed-G
    out = MLP(**output_mlp)(ed)


    # list_embeddings_n = [n]
    # list_embeddings_e = [ed]
    # list_embeddings_u = [ud]
    # for i in range(0, depth):
    #     if i>0:
    #         n = GraphMLP(**gin_mlp)(n)
    #         ed = GraphMLP(**gin_mlp)(ed)
    #         ud = GraphMLP(**graph_mlp)(ud)
    #     np,ep,up = DenseGNN(**gin_args)([n, edi, ed, ud])
    #     # be choosed
    #     list_embeddings_n.append(np)
    #     list_embeddings_e.append(ep)
    #     list_embeddings_u.append(up)
    #     n = LazyConcatenate()(list_embeddings_n)
    #     ed = LazyConcatenate()(list_embeddings_e)
    #     ud = LazyConcatenate()(list_embeddings_u)
       
    # GoGN output
    # nodes_new = update_features(node_features, np)
    # x = [ep, nodes_new, up, edi]
    # _, _, out, _ = output_block(x) 
    # out = output_block.get_features(out)

    return ks.Model(inputs=[edge_indices, atomic_number, offset, molecule_feature, env_input], outputs=out, name=name)





model_default_EGNN = {
    "name": "EGNN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 10), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": (None, 30), "name": "molecule_feature", "dtype": "float32", "ragged": True},
               ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 95, "output_dim": 64}},
    "depth": 4,
    "node_mlp_initialize": None,
    "euclidean_norm_kwargs": {"keepdims": True, "axis": 2},
    "use_edge_attributes": True,
    "edge_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "edge_attention_kwargs": None,  # {"units: 1", "activation": "sigmoid"}
    "use_normalized_difference": False,
    "expand_distance_kwargs": None,
    "coord_mlp_kwargs":  {"units": [64, 1], "activation": ["swish", "linear"]},  # option: "tanh" at the end.
    "pooling_coord_kwargs": {"pooling_method": "mean"},
    "pooling_edge_kwargs": {"pooling_method": "sum"},
    "node_normalize_kwargs": None,
    "use_node_attributes": False,
    "node_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "use_skip": True,
    "verbose": 10,
    "node_decoder_kwargs": None,
    "node_pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}, 
    
    "gin_mlp": {},
}


@update_model_kwargs(model_default_EGNN)
def model_default_EGNN(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               euclidean_norm_kwargs: dict = None,
               node_mlp_initialize: dict = None,
               use_edge_attributes: bool = None,
               edge_mlp_kwargs: dict = None,
               edge_attention_kwargs: dict = None,
               use_normalized_difference: bool = None,
               expand_distance_kwargs: dict = None,
               coord_mlp_kwargs: dict = None,
               pooling_coord_kwargs: dict = None,
               pooling_edge_kwargs: dict = None,
               node_normalize_kwargs: dict = None,
               use_node_attributes: bool = None,
               node_mlp_kwargs: dict = None,
               use_skip: bool = None,
               verbose: int = None,
               node_decoder_kwargs: dict = None,
               node_pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               gin_mlp: dict = None

               ):
    r"""Make `EGNN <https://arxiv.org/abs/2102.09844>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.EGNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_indices, edge_attributes]`
        or `[node_attributes, node_coordinates, edge_indices]` if :obj:`use_edge_attributes=False`.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, D)`.
                Can also be ignored if not needed.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Default is "EGNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        euclidean_norm_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EuclideanNorm`.
        node_mlp_initialize (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer for start embedding.
        use_edge_attributes (bool): Whether to use edge attributes including for example further edge information.
        edge_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        edge_attention_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        use_normalized_difference (bool): Whether to use a normalized difference vector for nodes.
        expand_distance_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PositionEncodingBasisLayer`.
        coord_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        pooling_coord_kwargs (dict):
        pooling_edge_kwargs (dict):
        node_normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphLayerNormalization` layer.
        use_node_attributes (bool): Whether to add node attributes before node MLP.
        node_mlp_kwargs (dict):
        use_skip (bool):
        verbose (int): Level of verbosity.
        node_decoder_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer after graph network.
        node_pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_input = ks.layers.Input(**inputs[3])
    molecule_feature = ks.layers.Input(**inputs[4])

    ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[3]['shape']) < 2)(edge_input)

    # embedding, if no feature dimension
    h0 = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    print("h0 shape:", h0.shape)

    print("molecule_feature shape:", molecule_feature.shape)

    # Feature list
    feature_list = [h0, molecule_feature]

    # Concatenate features
    # node_concat = Concatenate(axis=-1)(feature_list)
    node_concat = tf.concat(feature_list, -1)


    # Model
    h = GraphMLP(**node_mlp_initialize)(node_concat) if node_mlp_initialize else node_concat
    x = xyz_input

    # list_embeddings_n = [h]
    # list_embeddings_e = [x]
    # list_embeddings_g = [ed]
    for i in range(0, depth):

        # if i>0:
            # h = GraphMLP(**gin_mlp)(node)
            # x = GraphMLP(**gin_mlp)(coord)
            # ed = GraphMLP(**gin_mlp)(msg)

        pos1, pos2 = NodePosition()([x, edi])
        diff_x = LazySubtract()([pos1, pos2])
        norm_x = EuclideanNorm(**euclidean_norm_kwargs)(diff_x)
        # Original code has a normalize option for coord-differences.
        if use_normalized_difference: # False
            diff_x = EdgeDirectionNormalized()([pos1, pos2])
        if expand_distance_kwargs: # None
            norm_x = PositionEncodingBasisLayer()(norm_x)

        # Edge model
        h_i, h_j = GatherEmbeddingSelection([0, 1])([h, edi])
        if use_edge_attributes: # False
            m_ij = LazyConcatenate()([h_i, h_j, norm_x, ed])     
        else:
            m_ij = LazyConcatenate()([h_i, h_j, norm_x])
        if edge_mlp_kwargs:
            m_ij = GraphMLP(**edge_mlp_kwargs)(m_ij)
        if edge_attention_kwargs:
            m_att = GraphMLP(**edge_attention_kwargs)(m_ij)
            m_ij = LazyMultiply()([m_att, m_ij])

        # Coord model
        if coord_mlp_kwargs: # None
            m_ij_weights = GraphMLP(**coord_mlp_kwargs)(m_ij)
            x_trans = LazyMultiply()([m_ij_weights, diff_x])
            agg = AggregateLocalEdges(**pooling_coord_kwargs)([h, x_trans, edi]) #mean
            x = LazyAdd()([x, agg])

        # Node model
        m_i = AggregateLocalEdges(**pooling_edge_kwargs)([h, m_ij, edi]) # sum
        if node_mlp_kwargs:
            m_i = LazyConcatenate()([h, m_i])
            if use_node_attributes:  # False
                m_i = LazyConcatenate()([m_i, h0])
            m_i = GraphMLP(**node_mlp_kwargs)(m_i) 
        if node_normalize_kwargs:  # None
            h = GraphLayerNormalization(**node_normalize_kwargs)(h)
        if use_skip:
            h = LazyAdd()([h, m_i])
        else:
            h = m_i
        
        # be choosed
        # list_embeddings_n.append(h)
        # list_embeddings_e.append(x)
        # list_embeddings_g.append(m_ij)
        # node = LazyConcatenate()(list_embeddings_n)
        # coord = LazyConcatenate()(list_embeddings_e)
        # msg = LazyConcatenate()(list_embeddings_g)
     
    # Output embedding choice
    if node_decoder_kwargs:
        n = GraphMLP(**node_mlp_kwargs)(h)
    else:
        n = h

    # Final step.
    if output_embedding == 'graph':

        out = PoolingNodes(**node_pooling_kwargs)(n) # max
        out = MLP(**output_mlp)(out)

        # n = PoolingNodes(**node_pooling_kwargs)(n)  # node-G
        # ed = PoolingGlobalEdges(**node_pooling_kwargs)(m_ij) # ed-G
        # out = LazyConcatenate()([n, ed])
        # out = MLP(**output_mlp)(out)

    elif output_embedding == 'node':
        out = n
        out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet`")

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, edge_input, molecule_feature], outputs=out, name=name)

    return model







model_default_PAiNN = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
    "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
    "update_args": {"units": 128},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},

    "gin_mlp": {},
}


@update_model_kwargs(model_default_PAiNN)
def make_model_PAiNN(inputs: list = None,
               input_embedding: dict = None,
               equiv_initialize_kwargs: dict = None,
               bessel_basis: dict = None,
               depth: int = None,
               pooling_args: dict = None,
               conv_args: dict = None,
               update_args: dict = None,
               equiv_normalization: bool = None,
               node_normalization: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               gin_mlp: dict = None,
               ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices]`
        or `[node_attributes, node_coordinates, bond_indices, equiv_initial]` if a custom equivariant initialization is
        chosen other than zero.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - equiv_initial (tf.RaggedTensor): Equivariant initialization `(batch, None, 3, F)`. Optional.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        equiv_initialize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EquivariantInitialize` layer.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final :obj:`BesselBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        conv_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNUpdate` layer.
        equiv_normalization (bool): Whether to apply :obj:`GraphLayerNormalization` to equivariant tensor update.
        node_normalization (bool): Whether to apply :obj:`GraphBatchNormalization` to node tensor update.
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
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    z = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)

    if len(inputs) > 3:
        equiv_input = ks.layers.Input(**inputs[3])
    else:
        equiv_input = EquivariantInitialize(**equiv_initialize_kwargs)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    # list_embeddings_n = [z]
    # list_embeddings_e = [v]
   
    for i in range(depth):
        # if i>0:
            # z = GraphMLP(**gin_mlp)(n)
            # v = GraphMLP(**gin_mlp)(equivariant_n)

        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])

        if equiv_normalization: # False
            v = GraphLayerNormalization(axis=2)(v)
        if node_normalization: # False
            z = GraphBatchNormalization(axis=-1)(z)

        # list_embeddings_n.append(z)
        # list_embeddings_e.append(v)
        # n = LazyConcatenate()(list_embeddings_n)
        # equivariant_n = LazyConcatenate()(list_embeddings_e)
     
    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    if len(inputs) > 3:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, equiv_input], outputs=out)
    else:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input], outputs=out)
    return model



model_default_HamNet = {
    "name": "HamNet",
    "inputs": [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
               {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True}],
    "input_embedding": {
        "node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "message_kwargs": {"units": 128, "units_edge": 128},
    "fingerprint_kwargs": {"units": 128, "units_attend": 128, "depth": 2},
    "gru_kwargs": {"units": 128},
    "verbose": 10, "depth": 1,
    "union_type_node": "gru",
    "union_type_edge": "gru",
    "given_coordinates": True,
    'output_embedding': 'graph', "output_to_tensor": True,
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'linear']},  

     'node_pooling_args': {'pooling_method': 'max'},
     "gin_mlp":  {}, 'input_block_cfg':{}, 
     
     "gin_mlp_pq":  {}, 
     "gauss_args":  {}, 
    
}


@update_model_kwargs(model_default_HamNet, update_recursive=inf)
def make_model_HamNet(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               verbose: int = None,
               message_kwargs: dict = None,
               gru_kwargs: dict = None,
               fingerprint_kwargs: dict = None,
               union_type_node: str = None,
               union_type_edge: str = None,
               given_coordinates: bool = None,
               depth: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None, node_pooling_args: dict = None, 
               gin_mlp: dict = None,
               gin_mlp_pq: dict = None,
               input_block_cfg:dict = None, 
               gauss_args: dict = None,
               

               ):
    r"""Make `HamNet <https://arxiv.org/abs/2105.03688>`_ graph model via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.HamNet.model_default` .


    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`,
        or `[node_attributes, edge_attributes, edge_indices, node_coordinates]` if :obj:`given_coordinates=True`.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Euclidean coordinates of nodes of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model.
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict):  Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        verbose (int): Level of verbosity. For logging and printing.
        message_kwargs (dict): Dictionary of layer arguments unpacked in message passing layer for node updates.
        gru_kwargs (dict): Dictionary of layer arguments unpacked in gated recurrent unit update layer.
        fingerprint_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`HamNetFingerprintGenerator` layer.
        given_coordinates (bool): Whether coordinates are provided as model input, or are computed by the Model.
        union_type_edge (str): Union type of edge updates. Choose "gru", "naive" or "None".
        union_type_node (str): Union type of node updates. Choose "gru", "naive" or "None".
        depth (int): Depth or number of (message passing) layers of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    node_input = ks.layers.Input(**inputs[0]) # node_number
    edge_index_input = ks.layers.Input(**inputs[1]) # range_indices

    edi = edge_index_input

    # Generate coordinates.
    if given_coordinates:  # True
        # Case for given coordinates.
        q_ftr = ks.layers.Input(**inputs[2])  # node_coordinates
        p_ftr = ZerosLike()(q_ftr)  

    else:
        # Use Hamiltonian engine to get p, q coordinates.
        raise NotImplementedError("Hamiltonian engine not yet implemented")

    x = q_ftr
    pos1, pos2 = NodePosition()([x, edi])
    ed = NodeDistanceEuclidean()([pos1, pos2])
    ed = GaussBasisLayer(**gauss_args)(ed)

    node_in = {'features': node_input} 
    crystal_input_block = get_input_block(**input_block_cfg)
    n = crystal_input_block(node_in)

    # Initialization    
    n = Dense(units=gru_kwargs["units"], activation="swish")(n) 
    ed = Dense(units=gru_kwargs["units"], activation="swish")(ed) 
    p = p_ftr 
    q = q_ftr

    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    list_embeddings_p = [p]
    list_embeddings_q = [q]
    # Message passing.
    for i in range(depth):
        if i>0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
            p = GraphMLP(**gin_mlp_pq)(p)
            q = GraphMLP(**gin_mlp_pq)(q)

        # Message step
        nu, eu,p, q = HamNaiveDynMessage(**message_kwargs)([n, ed, p, q, edi])

        # Node updates
        if union_type_node == "gru": #T
            n = HamNetGRUUnion(**gru_kwargs)([n, nu])  
        elif union_type_node == "naive":
            n = HamNetNaiveUnion(units=gru_kwargs["units"])([n, nu])
        else:
            n = nu

        # Edge updates
        if union_type_edge == "gru":  #T
            ed = HamNetGRUUnion(**gru_kwargs)([ed, eu])
        elif union_type_edge == "naive":
            ed = HamNetNaiveUnion(units=gru_kwargs["units"])([ed, eu])
        else:
            ed = eu  # T

        # be choosed
        list_embeddings_n.append(n)
        list_embeddings_e.append(ed)
        list_embeddings_p.append(p)
        list_embeddings_q.append(q)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        p = LazyConcatenate()(list_embeddings_p)
        q = LazyConcatenate()(list_embeddings_q)

    # concat readout                                     
    # n = PoolingNodes(**node_pooling_args)(n)  # node-G
    # ed = PoolingGlobalEdges(**node_pooling_args)(ed) # ed-G
    # out = LazyConcatenate()([n, ed])
    # out = MLP(**output_mlp)(out)

    # Fingerprint generator for graph embedding.
    if output_embedding == 'graph':
        out = HamNetFingerprintGenerator(**fingerprint_kwargs)(n)
        out = ks.layers.Flatten()(out)  # will be tensor.
        out = MLP(**output_mlp)(out)
        
    # elif output_embedding == 'node':
    #     out = GraphMLP(**output_mlp)(n)
    #     if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
    #         out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    # else:
    #     raise ValueError("Unsupported output embedding for `HamNet`")

    # Make Model instance.
    if given_coordinates: # T
        model = ks.models.Model(inputs=[node_input, edge_index_input, q_ftr], outputs=out)
    else:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    return model




model_default_schnet = {
    "name": "Schnet",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],

    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},

    "make_distance": True, "expand_distance": True,
    "interaction_args": {"units": 128, "use_bias": True,
                         "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},

    "gin_mlp": None, 
    "input_block_cfg": None, 
    # "use_output_mlp": True,
    # "output_mlp": {"use_bias": [True, True], "units": [64, 1],
    #                "activation": ["kgcnn>shifted_softplus", "linear"]} ,
   
}


@update_model_kwargs(model_default_schnet)
def make_model_schnet(inputs: list = None,
                       input_embedding: dict = None,
                       input_block_cfg=None,
                    
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       interaction_args: dict = None,
                       node_pooling_args: dict = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       last_mlp: dict = None,
                    #    output_embedding: str = None,
                    #    use_output_mlp: bool = None,
                    #    output_to_tensor: bool = None,
                    #    output_mlp: dict = None,
                       gin_mlp: dict = None, 
                 

                       ):
    r"""

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        interaction_args (dict): Dictionary of layer arguments unpacked in final :obj:`SchNetInteraction` layers.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    
    # embedding, if no feature dimension
    # n = OptionalInputEmbedding(**input_embedding['node'],
    #                            use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    if make_distance:
        x = xyz_input
        pos1, pos2 = NodePosition()([x, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    
    else:
        ed = xyz_input

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    node_in = {'features': node_input} 
    crystal_input_block = get_input_block(**input_block_cfg)
    n = crystal_input_block(node_in)

    ed = Dense(interaction_args["units"], activation="linear")(ed) 
    n = Dense(interaction_args["units"], activation='linear')(n)
    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    for i in range(0, depth):
        if i>0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
        np,ep = SchNetInteraction(**interaction_args)([n, ed, edi])
        # choose
        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)

    # n = GraphMLP(**gc_mlp)(n)
    # Output embedding choice                                  
    n = PoolingNodes(**node_pooling_args)(n)  # node-G
    ed = PoolingGlobalEdges(**node_pooling_args)(ed) # ed-G
    out = LazyConcatenate()([n, ed])
    out = MLP(**last_mlp)(out)

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=out)
    return model



def get_input_block(node_size=64, 
                        atomic_mass=False, atomic_radius=False, electronegativity=False, ionization_energy=False,
                        oxidation_states=False, melting_point=False, density=False, mendeleev=False, molarvolume=False, vanderwaals_radius=False, 
                        average_cationic_radius=False, average_anionic_radius=False, velocity_sound=False, thermal_conductivity=False,
                        electrical_resistivity=False, rigidity_modulus=False,
                      
                      ):
        periodic_table = PeriodicTable()

        atom_embedding_layer = AtomEmbedding(
            atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_size},
            atomic_mass=periodic_table.get_atomic_mass() if atomic_mass else None,
            atomic_radius=periodic_table.get_atomic_radius() if atomic_radius else None,
            electronegativity=periodic_table.get_electronegativity() if electronegativity else None,
            ionization_energy=periodic_table.get_ionization_energy() if ionization_energy else None,
            oxidation_states=periodic_table.get_oxidation_states() if oxidation_states else None,
            melting_point=periodic_table.get_melting_point() if melting_point else None,
            density=periodic_table.get_density() if density else None,
            mendeleev=periodic_table.get_mendeleev() if mendeleev else None,
            molarvolume=periodic_table.get_molarvolume() if molarvolume else None,
            vanderwaals_radius=periodic_table.get_vanderwaals_radius() if vanderwaals_radius else None,
            average_cationic_radius=periodic_table.get_average_cationic_radius() if average_cationic_radius else None,
            average_anionic_radius=periodic_table.get_average_anionic_radius() if average_anionic_radius else None,
            velocity_sound=periodic_table.get_velocity_sound() if velocity_sound else None,
            thermal_conductivity=periodic_table.get_thermal_conductivity() if thermal_conductivity else None,
            electrical_resistivity=periodic_table.get_electrical_resistivity() if electrical_resistivity else None,
            rigidity_modulus=periodic_table.get_rigidity_modulus() if rigidity_modulus else None,
            )
           
        return atom_embedding_layer



class GraphNetworkConfigurator():
    def __init__(self, units=64, activation='swish', last_layer_activation='tanh',
                 edge_mlp_depth=3, node_mlp_depth=3, global_mlp_depth=3,
                 depth=4, ):
        self.units = units
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.edge_mlp_depth = edge_mlp_depth
        self.node_mlp_depth = node_mlp_depth
        self.global_mlp_depth = global_mlp_depth
        self.depth = depth

    @staticmethod
    def get_gn_block(edge_mlp={'units': [64, 64], 'activation': 'swish'},
                     node_mlp={'units': [64, 64], 'activation': 'swish'},
                     global_mlp={'units': [64, 32, 1], 'activation': ['swish', 'swish', 'linear']},
                     aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                     return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
                     edge_attention_mlp_local={'units': [1], 'activation': 'linear'},
                     edge_attention_mlp_global={'units': [1], 'activation': 'linear'},
                     node_attention_mlp={'units': [1], 'activation': 'linear'},
                     edge_gate=None, node_gate=None, global_gate=None,
                     residual_node_update=False, residual_edge_update=False, residual_global_update=False,
                     update_edges_input=[True, True, True, False],  
                     update_nodes_input=[True, False, False],  
                     update_global_input=[False, True, False],  
                     multiplicity_readout=False):
        if edge_gate == 'gru':
            edge_gate = GRUCell(edge_mlp['units'][-1])
   
        elif edge_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            edge_gate = None

        if node_gate == 'gru':
            node_gate = GRUCell(node_mlp['units'][-1])
     
        elif node_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            node_gate = None
        if global_gate == 'gru':
            global_gate = GRUCell(global_mlp['units'][-1])
     
        elif global_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            global_gate = None

        edge_mlp = MLP(**edge_mlp) if edge_mlp is not None else None
        node_mlp = MLP(**node_mlp) if node_mlp is not None else None
        global_mlp = MLP(**global_mlp) if global_mlp is not None else None
        edge_attention_mlp_local = MLP(**edge_attention_mlp_local) if edge_attention_mlp_local is not None else None
        edge_attention_mlp_global = MLP(**edge_attention_mlp_global) if edge_attention_mlp_global is not None else None
        node_attention_mlp = MLP(**node_attention_mlp) if node_attention_mlp is not None else None

        if multiplicity_readout:
            block = GraphNetworkMultiplicityReadout(edge_mlp, node_mlp, global_mlp,
                                                    aggregate_edges_local=aggregate_edges_local,
                                                    aggregate_edges_global=aggregate_edges_global,
                                                    aggregate_nodes=aggregate_nodes,
                                                    return_updated_edges=return_updated_edges,
                                                    return_updated_nodes=return_updated_nodes,
                                                    return_updated_globals=return_updated_globals,
                                                    edge_attention_mlp_local=edge_attention_mlp_local,
                                                    edge_attention_mlp_global=edge_attention_mlp_global,
                                                    node_attention_mlp=node_attention_mlp,
                                                    edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                                                    residual_edge_update=residual_edge_update,
                                                    residual_node_update=residual_node_update,
                                                    residual_global_update=residual_global_update,
                                                    update_edges_input=update_edges_input,
                                                    update_nodes_input=update_nodes_input,
                                                    update_global_input=update_global_input)
        else:
            block = GraphNetwork(edge_mlp, node_mlp, global_mlp,
                                 aggregate_edges_local=aggregate_edges_local,
                                 aggregate_edges_global=aggregate_edges_global,
                                 aggregate_nodes=aggregate_nodes,
                                 return_updated_edges=return_updated_edges,
                                 return_updated_nodes=return_updated_nodes,
                                 return_updated_globals=return_updated_globals,
                                 edge_attention_mlp_local=edge_attention_mlp_local,
                                 edge_attention_mlp_global=edge_attention_mlp_global,
                                 node_attention_mlp=node_attention_mlp,
                                 edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                                 residual_edge_update=residual_edge_update,
                                 residual_node_update=residual_node_update,
                                 residual_global_update=residual_global_update,
                                 update_edges_input=update_edges_input,
                                 update_nodes_input=update_nodes_input,
                                 update_global_input=update_global_input)
        return block

    @staticmethod
    def get_input_block(node_size=64, edge_size=64,
                        atomic_mass=False, atomic_radius=False, electronegativity=False, ionization_energy=False,
                        oxidation_states=False, melting_point=False, density=False, mendeleev=False, molarvolume=False, vanderwaals_radius=False, 
                        average_cationic_radius=False, average_anionic_radius=False, velocity_sound=False, thermal_conductivity=False,
                        electrical_resistivity=False, rigidity_modulus=False,
                        edge_embedding_args={
                            'bins_distance': 32, 'max_distance': 5., 'distance_log_base': 1.,
                            'bins_voronoi_area': None, 'max_voronoi_area': None}):
        periodic_table = PeriodicTable()

        atom_embedding_layer = AtomEmbedding(
            atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_size},
            atomic_mass=periodic_table.get_atomic_mass() if atomic_mass else None,
            atomic_radius=periodic_table.get_atomic_radius() if atomic_radius else None,
            electronegativity=periodic_table.get_electronegativity() if electronegativity else None,
            ionization_energy=periodic_table.get_ionization_energy() if ionization_energy else None,
            oxidation_states=periodic_table.get_oxidation_states() if oxidation_states else None,
            melting_point=periodic_table.get_melting_point() if melting_point else None,
            density=periodic_table.get_density() if density else None,
            mendeleev=periodic_table.get_mendeleev() if mendeleev else None,
            molarvolume=periodic_table.get_molarvolume() if molarvolume else None,
            vanderwaals_radius=periodic_table.get_vanderwaals_radius() if vanderwaals_radius else None,
            average_cationic_radius=periodic_table.get_average_cationic_radius() if average_cationic_radius else None,
            average_anionic_radius=periodic_table.get_average_anionic_radius() if average_anionic_radius else None,
            velocity_sound=periodic_table.get_velocity_sound() if velocity_sound else None,
            thermal_conductivity=periodic_table.get_thermal_conductivity() if thermal_conductivity else None,
            electrical_resistivity=periodic_table.get_electrical_resistivity() if electrical_resistivity else None,
            rigidity_modulus=periodic_table.get_rigidity_modulus() if rigidity_modulus else None,
            )
        
        edge_embedding_layer = EdgeEmbedding(**edge_embedding_args)
        crystal_input_block = CrystalInputBlock(atom_embedding_layer,
                                                edge_embedding_layer,
                                                atom_mlp=MLP([node_size]), edge_mlp=MLP([edge_size]))
        return crystal_input_block



