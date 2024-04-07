import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from ._attentivefp_conv import AttentiveHeadFP, PoolingNodesAttentive
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.modules import Dense, Dropout, OptionalInputEmbedding, LazyConcatenate
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.pooling import PoolingNodes,PoolingGlobalEdges
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# import tensorflow.keras as ks
ks = tf.keras

# Implementation of AttentiveFP in `tf.keras` from paper:
# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li,
# Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959


model_default = {
    "name": "AttentiveFP",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "attention_args": {"units": 32},
    "depthmol": 2,
    "depthato": 2,
    "dropout": 0.1,
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}, "gin_mlp":  {}, 
    'node_pooling_args': {'pooling_method': 'mean'},
    'input_block_cfg':{}, 
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depthmol: int = None,
               depthato: int = None,
               dropout: float = None,
               attention_args: dict = None,
               name: str = None,
               node_pooling_args: dict = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None, gin_mlp: dict = None, input_block_cfg:dict = None, 
               ): 
    r"""Make `AttentiveFP <https://doi.org/10.1021/acs.jmedchem.9b00959>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.AttentiveFP.model_default`.

    The attention mechanism helps the model focus on relevant information from the local environment of each atom, improving the interpret


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
        depthato (int): Number of graph embedding units or depth of the network.
        depthmol (int): Number of graph embedding units or depth of the graph embedding.
        dropout (float): Dropout to use.
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentiveHeadFP` layer. Units parameter
            is also used in GRU-update and :obj:`PoolingNodesAttentive`.
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
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1]) #range_attributes
    edge_index_input = ks.layers.Input(**inputs[2])

    # Embedding, if no feature dimension
    # n = OptionalInputEmbedding(**input_embedding['node'],
    #                            use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    inp_CrystalNNFinger = ks.layers.Input(**inputs[3])
    node_in = {'features': node_input, 'CrystalNNFinger': inp_CrystalNNFinger} 
    crystal_input_block = get_input_block(**input_block_cfg)
    n = crystal_input_block(node_in)
    # Model
    nk = Dense(units=attention_args['units'])(n)
    ck,_ = AttentiveHeadFP(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])  # 识别局部化学环境 这点和Hamnet一致

    list_embeddings_n = [nk]
    list_embeddings_e = [ed]
    for i in range(1, depthato):  # By applying the graph attention mechanism in multiple attentive layers, the model can capture nonlocal effects and learn complex patterns and relationships within the molecule. This enables Attentive FP to extract hidden information related to molecular properties and behaviors, such as solubility, aromaticity, and conformation.
        if  i > 0:  #T
            nk = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
        ck,ek = AttentiveHeadFP(**attention_args)([nk, ed, edi])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = Dropout(rate=dropout)(nk)
        # be choosed
        list_embeddings_n.append(nk)
        list_embeddings_e.append(ek)
        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
    # n = nk
  
    # concat readout                                     
    n = PoolingNodes(**node_pooling_args)(n)  # node-G
    ed = PoolingGlobalEdges(**node_pooling_args)(ed) # ed-G
    out = LazyConcatenate()([n, ed])
    out = MLP(**output_mlp)(out)

    # Output embedding choice
    # if output_embedding == 'graph':
    #     out = PoolingNodesAttentive(units=attention_args['units'], depth=depthmol)(n)  # Tensor output.
    #     out = MLP(**output_mlp)(out)
    # elif output_embedding == 'node':
    #     out = GraphMLP(**output_mlp)(n)
    #     if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
    #         out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    # else:
    #     raise ValueError("Unsupported graph embedding for mode `AttentiveFP`")


    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, inp_CrystalNNFinger], outputs=out, name=name)
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



'''
The innovation of the Attentive FP network architecture lies in its introduction of an attention mechanism for extracting nonlocal 
effects at the intramolecular level. This attention mechanism allows the model to focus on the most relevant parts of the inputs, 
leading to better predictions. The Attentive FP model outperforms other graph neural network models, such as GCN and MPNN, in various tasks, 
including predicting the number of aromatic atoms in a molecule and calculating quantum properties of small organic molecules. 
The model's attention weights at the atom level have chemical implications that can be easily interpreted, providing insights 
into the underlying molecular properties. Additionally, the Attentive FP model successfully 
captures nonlocal effects among atoms, allowing it to learn representations related to molecular solubility and other chemical properties.

'''

'''
     Full-Molecule Embedding: To combine the individual atom state vectors into a representation for the entire molecule, 
     a super virtual node is introduced. This virtual node connects all the atoms in the molecule and undergoes 
     a similar atom embedding process as described above. The resulting state vector of the virtual node represents the learned representation of the whole molecule.
    
'''
'''
    By applying the graph attention mechanism in multiple attentive layers, the model can capture nonlocal effects and 
    learn complex patterns and relationships within the molecule. 
    This enables Attentive FP to extract hidden information related to molecular properties and behaviors, 
    such as solubility, aromaticity, and conformation.
    
'''
'''
    Atom Embedding: Each atom in the molecule has its own state vector, 
    which is initially generated using a fully connected layer that incorporates the atom and bond features. 
    The atom embedding process involves multiple stacked attentive layers, allowing the atom to aggregate information from its neighboring atoms using an attention mechanism. 
    The attention mechanism aligns and weights the state vectors of the target atom and its neighbors, producing an attention context vector. This context vector, 
    along with the current state vector of the target atom, is then passed through a gated recurrent unit (GRU) to update the state vector.

'''