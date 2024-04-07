import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.coGN._preprocessing_layers import LineGraphAngleDecoder
from kgcnn.literature.coGN._gates import HadamardProductGate
from kgcnn.literature.coGN._graph_network.graph_networks import NestedGraphNetwork, SequentialGraphNetwork, GraphNetwork, \
    GraphNetworkMultiplicityReadout, CrystalInputBlock
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.coGN._embedding_layers._edge_embedding import EdgeEmbedding, GaussBasisExpansion
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.literature.coGN._preprocessing_layers import EdgeDisplacementVectorDecoder
from kgcnn.layers.modules import Dense, OptionalInputEmbedding 
from kgcnn.layers.casting import ChangeTensorType
from ._dense_gnn_conv import  DenseGNN
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, Dense, OptionalInputEmbedding
from copy import copy


from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.mlp import GraphMLP,MLP
from kgcnn.layers.pooling import PoolingGlobalEdges
from ...layers.pooling import PoolingNodes
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

def make_model(inputs=None,
                name=None,
                input_block_cfg=None,
                # output_block_cfg = None,
                input_embedding: dict = None, 
                depth: int = None,
                gin_args: dict = None,
                gin_mlp: dict = None,
                graph_mlp: dict = None, n_units: int = None,
                gauss_args: dict = None,
                node_pooling_args: dict = None,
                last_mlp: dict = None,

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
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_image = ks.layers.Input(**inputs[3])
    lattice = ks.layers.Input(**inputs[4])
    inp_CrystalNNFinger = ks.layers.Input(**inputs[5])
    env_input = ks.Input(**inputs[6]) 
    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                    use_embedding=len(inputs[6]['shape']) < 1)(env_input) # graph

    edi = edge_index_input
    x = xyz_input
    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
    ed = NodeDistanceEuclidean()([pos1, pos2])
    ed = GaussBasisLayer(**gauss_args)(ed)

    node_in = {'features': node_input, 'CrystalNNFinger': inp_CrystalNNFinger} 
    crystal_input_block = get_input_block(**input_block_cfg)
    n = crystal_input_block(node_in)

    
    n = Dense(n_units, use_bias=True, activation='relu')(n)
    ed = Dense(n_units, use_bias=True, activation='relu')(ed)
    ud  = Dense(n_units, use_bias=True, activation='relu')(uenv) # graph

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
       
    #  output
    n = PoolingNodes(**node_pooling_args)(np)  # node-G
    ed = PoolingGlobalEdges(**node_pooling_args)(ep) # ed-G
    out = LazyConcatenate()([n, ed, up])
    out = MLP(**last_mlp)(out)
    
    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, edge_image, lattice, inp_CrystalNNFinger,env_input], outputs=out)
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