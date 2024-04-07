import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._gin_conv_raw import GIN, GINE
from kgcnn.layers.mlp import GraphMLP, MLP
from ...layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, Dense, OptionalInputEmbedding 
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding

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
    "gin_args": {"epsilon_learnable": False},
    "depth": 3, "dropout": 0.5, "verbose": 10,
    "node_pooling_args": {"pooling_method": "mean"},
    "last_mlp": {"use_bias": [True, True, True], "units": [64, 64, 64],
                 "activation": ["relu", "relu", "linear"]},
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"},  
    "gc_mlp" :{}, "gl_mlp" :{}, 
    'input_block_cfg':{}, 

}


@update_model_kwargs(model_default_edge)
def make_model_edge(inputs: list = None,
                    input_embedding: dict = None, 
                    depth: int = None,
                    gin_args: dict = None,
                    gin_mlp: dict = None,
                    gc_mlp: dict = None,
                    last_mlp: dict = None,
                    dropout: float = None, 
                    output_embedding: str = None, 
                    output_mlp: dict = None, 
                    output_to_tensor: bool = None,
                    node_pooling_args: dict = None,
                    name: str = None, verbose: int = None, gl_mlp: dict = None,
                    input_block_cfg=None,

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

    # Make input embedding, if no feature dimension
    # n = OptionalInputEmbedding(**input_embedding['node'],
    #                            use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    # Model
    inp_CrystalNNFinger = ks.layers.Input(**inputs[3])
    node_in = {'features': node_input, 'CrystalNNFinger': inp_CrystalNNFinger} 
    crystal_input_block = get_input_block(**input_block_cfg)
    n = crystal_input_block(node_in)

    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='relu')(n) # nodes
    ed = Dense(n_units, use_bias=True, activation='relu')(ed) # edges
    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    for i in range(0, depth):
        if i>0:
            n = GraphMLP(**gc_mlp)(n)
            ed = GraphMLP(**gc_mlp)(ed)
        np,ep = GINE(**gin_args)([n, edi, ed])
        np = GraphMLP(**gl_mlp)(np)
        ep = GraphMLP(**gl_mlp)(ep)
        # be choosed
        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        
    # concat readout                                     
    n = PoolingNodes(**node_pooling_args)(n)  # node-G
    ed = PoolingGlobalEdges(**node_pooling_args)(ed) # ed-G
    out = LazyConcatenate()([n, ed])
    out = MLP(**output_mlp)(out)

        
    # Output embedding choice
    # if output_embedding == "graph":
    #     out = [PoolingNodes(**node_pooling_args)(x) for x in list_embeddings]  
    #     # out = [GraphMLP(**last_mlp)(x) for x in out]
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

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, inp_CrystalNNFinger], outputs=out)
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