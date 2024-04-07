import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import  GraphNetwork, GraphNetworkMultiplicityReadout, CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP
from tensorflow.keras.layers import GRUCell
from kgcnn.layers.modules import Dense, OptionalInputEmbedding 
from ._dense_gnn_conv import  DenseGNN
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.modules import LazyConcatenate, Dense, OptionalInputEmbedding
from copy import copy
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

def make_model_asu(inputs=None,
                name=None,
                input_block_cfg=None,
                output_block_cfg = None,
                input_embedding: dict = None, 
                depth: int = None,
                gin_args: dict = None,
                gin_mlp: dict = None,
                graph_mlp: dict = None, n_units: int = None,
              ):
    r"""Make dense connectivity optimized graph networks for crystals.

    Args:
        inputs (list): List of inputs kwargs.
        input_block_cfg (dict): Input block config.
        processing_blocks_cfg (list): List of processing block configs.
        output_block_cfg: Output block config.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    
    edge_indices = ks.Input(**inputs['edge_indices']) 
    atomic_number = ks.Input(**inputs['atomic_number']) 
    edge_inputs, node_inputs, global_inputs = [], [atomic_number], []

    # Helper function which is a workaround for not being able to delete entries from `model_default` dict.
    def in_inputs(key):
        return key in inputs and inputs[key] is not None

    if in_inputs('offset'):  
        offset = ks.Input(**inputs['offset']) 
        edge_inputs.append(offset) 
    else:
        raise ValueError('The model needs either the "offset"\
                         or "coords" or "cell_translation", "frac_coords" and "lattice_matrix" as input.')
    
    if in_inputs('voronoi_ridge_area'): 
        inp_voronoi_ridge_area = ks.Input(**inputs['voronoi_ridge_area'])
        edge_inputs.append(inp_voronoi_ridge_area)

    if in_inputs('AGNIFinger'):  
        inp_AGNIFinger= ks.Input(**inputs['AGNIFinger'])  
        node_inputs.append(inp_AGNIFinger)

    euclidean_norm = EuclideanNorm()
    distance = euclidean_norm(offset) 

    if in_inputs('voronoi_ridge_area'):
        edge_input = (distance, inp_voronoi_ridge_area)
    else:
        edge_input = distance 

    if in_inputs('AGNIFinger'):
        node_input = {'features': atomic_number, 'AGNIFinger': inp_AGNIFinger}
    else:
        node_input = {'features': atomic_number}

    env_input = ks.Input(**inputs['charge']) 
    global_inputs.append(env_input)
    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                    use_embedding=len(inputs['charge']['shape']) < 1)(env_input) 
    global_input = Dense(n_units, use_bias=True, activation='relu')(uenv) 


    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    output_block = GraphNetworkConfigurator.get_gn_block(**output_block_cfg)
    edge_features, node_features, global_features, _ = crystal_input_block([edge_input,
                                                              node_input,
                                                              global_input,
                                                              edge_indices])
    n = get_features(node_features)
    ed = edge_features
    ud = global_features
    edi = edge_indices

    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    list_embeddings_u = [ud]
    for i in range(0, depth):
        if i>0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
            ud = GraphMLP(**graph_mlp)(ud)
        np,ep,up = DenseGNN(**gin_args)([n, edi, ed, ud])
        # be choosed
        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)
        list_embeddings_u.append(up)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        ud = LazyConcatenate()(list_embeddings_u)
       
    # output
    nodes_new = update_features(node_features, np)
    x = [ep, nodes_new, up, edi]
    _, _, out, _ = output_block(x) 
    out = output_block.get_features(out)

    input_list = edge_inputs + node_inputs + global_inputs + [edge_indices]
    return ks.Model(inputs=input_list, outputs=out, name=name)



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
                        atomic_mass=True, atomic_radius=True, electronegativity=True, ionization_energy=True,
                        oxidation_states=True, melting_point=True, density=True, mendeleev=False, molarvolume=False, vanderwaals_radius=False, 
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
