import tensorflow as tf
from copy import copy
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding,LazyAdd,Dense,Activation
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import PoolingLocalMessages, AggregateLocalEdgesLSTM
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.literature.coGN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.literature.coGN._graph_network.graph_networks import  CrystalInputBlock
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.graph.preprocessor import ExpandDistanceGaussianBasis

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GraphSAGE in `tf.keras` from paper:
# Inductive Representation Learning on Large Graphs
# by William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216


model_default = {
    'name': "GraphSAGE",
    'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
    'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'node_ff_args': {"units": [64, 32], "activation": "relu"},
    'edge_ff_args': {"units": [64, 32], "activation": "relu"},
    'pooling_args': {'pooling_method': "segment_mean"}, 'gather_args': {},
    'concat_args': {"axis": -1},
    'use_edge_features': True, 'pooling_nodes_args': {'pooling_method': "mean"},
    'depth': 3, 'verbose': 10,
    'output_embedding': 'graph', "output_to_tensor": True,  "gin_mlp":  {},
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'sigmoid']},  
    
    'input_block_cfg':{}, 'edge_embedding_args':{},
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               node_mlp_args: dict = None,
               edge_mlp_args: dict = None,
               node_ff_args: dict = None,
               edge_ff_args: dict = None,
               pooling_args: dict = None,
               pooling_nodes_args: dict = None,
               gather_args: dict = None,
               concat_args: dict = None,
               use_edge_features: bool = None,
               depth: int = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None, gin_mlp: dict = None,
               input_block_cfg=None, edge_embedding_args=None, 
               ):
    r"""Make `GraphSAGE <http://arxiv.org/abs/1706.02216>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.GraphSAGE.model_default` .
    1. 对图中每个顶点邻居顶点进行采样

    2. 根据聚合函数聚合邻居顶点蕴含的信息

    3. 得到图中各顶点的向量表示供下游任务使用

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
        node_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for node updates.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge updates.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingLocalMessages` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        gather_args (dict): Dictionary of layer arguments unpacked in :obj:`GatherNodes` layer.
        concat_args (dict): Dictionary of layer arguments unpacked in :obj:`LazyConcatenate` layer.
        use_edge_features (bool): Whether to add edge features in message step.
        depth (int): Number of graph embedding units or depth of the network.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Make input embedding, if no feature dimension
    # n = OptionalInputEmbedding(**input_embedding['node'],
    #                            use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input


    inp_CrystalNNFinger = ks.layers.Input(**inputs[3])
    # inp_voronoi_ridge_area = ks.layers.Input(**inputs[4])
    euclidean_norm = EuclideanNorm()
    edge_info = euclidean_norm(ed) 
    edge_info = EdgeEmbedding(**edge_embedding_args)(edge_info)
    node_in = {'features': node_input, 'CrystalNNFinger': inp_CrystalNNFinger} 
    crystal_input_block = get_input_block(node_in, **input_block_cfg)
    node_features = crystal_input_block(node_in)
    ed = edge_info
    n = node_features

    n = Dense(**node_ff_args)(n)# nodes
    ed = Dense(**edge_ff_args)(ed) # edges
    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    for i in range(0, depth):
        if i>0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
        eu = GatherNodesOutgoing(**gather_args)([n, edi])   # 当然，若不考虑计算效率，我们完全可以对每个顶点利用其所有的邻居顶点进行信息聚合，这样是信息无损的。
        if use_edge_features:
            eu = LazyConcatenate(**concat_args)([eu, ed])  # 原文中没有用edges features
        eu = GraphMLP(**edge_mlp_args)(eu)
        # ed_s = LazyAdd()([eu, ed])  #
        # ed_s = Activation(activation="relu")(ed_s) #

        # Pool message      
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = AggregateLocalEdgesLSTM(**pooling_args)([n, eu, edi]) 
        else:
            nu = PoolingLocalMessages(**pooling_args)([n, eu, edi]) 

        nu = LazyConcatenate(**concat_args)([n, nu])  
        nu = GraphMLP(**node_mlp_args)(nu) 
        # n = LazyAdd()([n, nu]) # 
        # n = Activation(activation="relu")(n) #
        # n = GraphLayerNormalization()(n)

        # be choosed
        list_embeddings_n.append(nu)
        list_embeddings_e.append(eu)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
   
    
    # concat readout                                     
    n = PoolingNodes(**pooling_nodes_args)(n)  # node-G
    ed = PoolingGlobalEdges(**pooling_nodes_args)(ed) # ed-G
    out = LazyConcatenate()([n, ed])
    out = MLP(**output_mlp)(out)

    # Regression layer on output
    # if output_embedding == 'graph':
    #     out = PoolingNodes(**pooling_nodes_args)(n)
    #     out = ks.layers.Flatten()(out)  # will be tensor
    #     out = MLP(**output_mlp)(out)
    # elif output_embedding == 'node':
    #     out = GraphMLP(**output_mlp)(n)
    #     if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
    #         out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    # else:
    #     raise ValueError("Unsupported output embedding for `GraphSAGE`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, inp_CrystalNNFinger ], outputs=out)
    # model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    model.__kgcnn_model_version__ = __model_version__
    return model


def get_input_block(nodes, node_size=64, edge_size=64,
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
        
        # edge_embedding_layer = EdgeEmbedding(**edge_embedding_args)
        # crystal_input_block = CrystalInputBlock(atom_embedding_layer,
        #                                         edge_embedding_layer,
        #                                         atom_mlp=MLP([node_size]), edge_mlp=MLP([edge_size]))
        
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