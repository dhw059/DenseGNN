import tensorflow as tf
from copy import copy
from kgcnn.layers.casting import ChangeTensorType
from ._schnet_densenet_conv import SchNetInteraction
from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import Dense, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from ...layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.modules import  OptionalInputEmbedding, Dense, ZerosLike
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.coGN._embedding_layers._edge_embedding import GaussBasisExpansion
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.pooling import PoolingGlobalEdges
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of Schnet in `tf.keras` from paper:
# by Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela,
# Alexandre Tkatchenko, Klaus-Robert Müller (2018)
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


model_default = {
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
    "output_embedding": "graph", "output_to_tensor": True, "gin_mlp":  {},
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]} ,
    "gc_mlp":  {}, 'input_block_cfg':{}, 
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               interaction_args: dict = None,
               node_pooling_args: dict = None,
               depth: int = None,
               name: str = None,
               verbose: int = None,
               last_mlp: dict = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices]`
        or `[node_attributes, node_coordinates, edge_indices]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

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
    # edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    # ed = OptionalInputEmbedding(**input_embedding["edge"],
    #                             use_embedding=len(inputs[1]["shape"]) < 2)(edge_input)
    edi = edge_index_input


    if make_distance:
        x = xyz_input
        pos1, pos2 = NodePosition()([x, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ed = xyz_input # node_coordinates

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, edi])

    n = GraphMLP(**last_mlp)(n)

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)(n)
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet`")

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=out)
    # model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    model.__kgcnn_model_version__ = __model_version__
    return model


model_crystal_default = {
    "name": "Schnet",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "edge_image", "dtype": "int64", "ragged": True},
               {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "make_distance": True, "expand_distance": True,
    "interaction_args": {"units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
    "output_embedding": "graph", "output_to_tensor": True,  "gin_mlp":  {},
    "use_output_mlp": True, 
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]}
}


@update_model_kwargs(model_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       interaction_args: dict = None,
                       node_pooling_args: dict = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       last_mlp: dict = None,
                       output_embedding: str = None,
                       use_output_mlp: bool = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None,
                       gin_mlp: dict = None, gc_mlp: dict = None, input_block_cfg=None,

                       ):
    r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_crystal_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, edge_image, lattice]`
        or `[node_attributes, node_coordinates, edge_indices, edge_image, lattice]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

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
    edge_image = ks.layers.Input(**inputs[3])
    # offset = ks.layers.Input(**inputs[2])
    lattice = ks.layers.Input(**inputs[4])
    inp_CrystalNNFinger = ks.layers.Input(**inputs[5])
    

    # embedding, if no feature dimension
    # n = OptionalInputEmbedding(**input_embedding['node'],
    #                            use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    if make_distance:
        x = xyz_input
        pos1, pos2 = NodePosition()([x, edi])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
        ed = NodeDistanceEuclidean()([pos1, pos2])

        # euclidean_norm = EuclideanNorm()
        # ed = euclidean_norm(offset) 
        # ed = EuclideanNorm(axis=2, keepdims=True, add_eps=False, no_nan=True)(offset)
    
    else:
        ed = xyz_input

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)
        # ed = GaussBasisExpansion.from_bounds(25, 0.0, 5.0, variance=1.0)(ed)

    # Model
    node_in = {'features': node_input, 'CrystalNNFinger': inp_CrystalNNFinger} 
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

    # if output_embedding == 'graph':
        # out = PoolingNodes(**node_pooling_args)(n)
    #     if use_output_mlp: # F
    #         out = MLP(**output_mlp)(out)
    # elif output_embedding == 'node':
    #     out = n
    #     if use_output_mlp:
    #         out = GraphMLP(**output_mlp)(out)
    #     if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
    #         out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    # else:
    #     raise ValueError("Unsupported output embedding for mode `SchNet`")

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, edge_image, lattice, inp_CrystalNNFinger], outputs=out)
    # model = ks.models.Model(inputs=[node_input, edge_index_input, offset, inp_CrystalNNFinger], outputs=out)

    model.__kgcnn_model_version__ = __model_version__
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