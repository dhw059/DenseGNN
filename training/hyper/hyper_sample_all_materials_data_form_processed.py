hyper = {
    
    
    "Megnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Megnet",
            "class_name": "make_crystal_model",
            "config": {
                'name': "Megnet",
                'inputs': [{'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                           {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}],
       
                'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "make_distance": True, "expand_distance": True,
                'gauss_args': {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4},
                'meg_block_args': {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                   'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'},
                'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
                'node_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'edge_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'state_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'nblocks': 3, 'has_ff': True, 'dropout': None, 'use_set2set': True,
                'verbose': 10,
                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": ['kgcnn>softplus2', 'kgcnn>softplus2', 'linear']}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 512, "epochs": 100, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 100, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.scaler",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "SampleAllMPFormDataset",
                "module_name": "kgcnn.data.datasets.SampleAllMPFormDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 18}}
                ]
            },
            "data_unit": "eV"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },

    



    "Schnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Schnet",
            "class_name": "make_crystal_model",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                    {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                    {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64}
                },
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "mean"},
                "depth": 1,
                "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,  # Last MLP sets output dimension if None.
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 512, "epochs": 200, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.scaler",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "SampleAllMPFormDataset",
                "module_name": "kgcnn.data.datasets.SampleAllMPFormDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5, "max_neighbours": 18}}
                ]
            },
            "data_unit": "eV"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },



    "CGCNN.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.CGCNN",
            "config": {
                'name': 'CGCNN',
                'inputs': [
                    {'shape': (None,), 'name': 'node_number', 'dtype': 'int64', 'ragged': True},
                    {'shape': (None, 3), 'name': 'node_frac_coordinates', 'dtype': 'float64', 'ragged': True},
                    {'shape': (None, 2), 'name': 'range_indices', 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': 'graph_lattice', 'dtype': 'float64', 'ragged': False},
                    {'shape': (None, 3), 'name': 'range_image', 'dtype': 'float32', 'ragged': True},

                    # For `representation="asu"`:
                    # {'shape': (None, 1), 'name': 'multiplicities', 'dtype': 'float32', 'ragged': True},
                    # {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64', 'ragged': True},
                ],
                'input_embedding': {'node': {'input_dim': 95, 'output_dim': 64}},
                'representation': 'unit',  # None, 'asu' or 'unit'
                'expand_distance': True,
                'make_distances': True,
                'gauss_args': {'bins': 60, 'distance': 6, 'offset': 0.0, 'sigma': 0.4},
                'conv_layer_args': {
                    'units': 128,
                    'activation_s': 'kgcnn>shifted_softplus',
                    'activation_out': 'kgcnn>shifted_softplus',
                    'batch_normalization': True,
                },
                'node_pooling_args': {'pooling_method': 'mean'},
                'depth': 4,
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                'output_mlp': {'use_bias': [True, True, False], 'units': [128, 64, 1],
                               'activation': ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 500, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 6.0}}
                    
                    #   {"set_representation": {
                    #     "pre_processor": {
                    #         "class_name": "KNNUnitCell",
                    #                       "module_name": "kgcnn.crystal.preprocessor",
                    #                       "config": {"k": 4}
                    #                       },
                    #     "reset_graphs": False}},


                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },


  
    "coGN": {
        "model": {
            "module_name": "kgcnn.literature.coGN",
            "class_name": "make_model",
            "config": {
                "name": "coGN",
                "inputs": {
                    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                    "cell_translation": None,
                    "affine_matrix": None,
                    "voronoi_ridge_area": None,
                    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                    "frac_coords": None,
                    "coords": None,
                    "multiplicity": {"shape": (None,), "name": "multiplicity", "dtype": "int32", "ragged": True},
                    "lattice_matrix": None,
                    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int32", "ragged": True},
                    "line_graph_edge_indices": None,
                },
                # All default.
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 256, "epochs": 100, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    # {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    #     "learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 0, "epo": 800,
                    #     "verbose": 0}
                    #  }
                ]
            },

            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                        "class_name": "ExponentialDecay",
                        "config": {"initial_learning_rate": 0.001,
                                   "decay_steps": 5800,
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },

            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "SampleAllMPFormDataset",
                "module_name": "kgcnn.data.datasets.SampleAllMPFormDataset",
                "config": {},
                "methods": [
                    {"set_representation": {

                        "pre_processor": {
                            
                            "class_name": "KNNAsymmetricUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 18}


                                          },
                                          
                        "reset_graphs": False}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.1"
        }
    },



    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_model_asu",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": {

    #                     "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
    #                     "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
    #                     "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
    #                     "AGNIFinger": {"shape": (None,128), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
    #                     "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
    #                     "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
    #                        },

    #             "input_block_cfg" : {'node_size': 128,
    #                'edge_size': 128, 

    #                'atomic_mass': True, 
    #                'atomic_radius': True, 
    #                'electronegativity': True, 
    #                'oxidation_states': True, 

    #                'ionization_energy': True, 
    #                'melting_point':True,    
    #                 'density':True,             


    #                'edge_embedding_args': {'bins_distance': 32,
    #                                        'max_distance': 8.0,
    #                                        'distance_log_base': 1.0,
    #                                        'bins_voronoi_area': 25,
    #                                        'max_voronoi_area': 32}},


    #             "output_block_cfg" : {'edge_mlp': None,
    #                                 'node_mlp': None,
    #                                 'global_mlp': {'units': [1],
    #                                             'activation': ['linear']},
    #                                 # 'nested_blocks_cfgs': None,
    #                                 'aggregate_edges_local': 'sum',
    #                                 'aggregate_edges_global': 'mean',
    #                                 'aggregate_nodes': 'mean',
    #                                 'return_updated_edges': False,
    #                                 'return_updated_nodes': False,
    #                                 'return_updated_globals': True,
    #                                 'edge_attention_mlp_local': {'units': [32, 1],
    #                                                             'activation': ['swish', 'swish']},
    #                                 'edge_attention_mlp_global': {'units': [32, 1],
    #                                                             'activation': ['swish', 'swish']},
    #                                 'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
    #                                 'edge_gate': None,
    #                                 'node_gate': None,
    #                                 'global_gate': None,
    #                                 'residual_node_update': False,
    #                                 'residual_edge_update': False,
    #                                 'residual_global_update': False,
    #                                 'update_edges_input': [True, True, True, False],
    #                                 'update_nodes_input': [True, False, False],
    #                                 'update_global_input': [False, True, False],  
    #                                 'multiplicity_readout': False},

                    
    #             "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
    #                                 "graph": {"input_dim": 100, "output_dim": 64}
    #                                 },

    #             "depth": 5,
    #             "n_units":128,
    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
    #                         },
    #             "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
    #                         },

    #             "gin_args": {"pooling_method":"sum", "g_pooling_method":"mean",
    #                          "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
    #                          "concat_args": {"axis": -1}, 
    #                          "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
    #                          "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
    #                          },

    #         }
    #     },

    #     "training": {
    #         "fit": {"batch_size": 256, "epochs": 100, "validation_freq": 20, "verbose": 2, "callbacks": []},

    #         "compile": {
    #             "optimizer": {"class_name": "Adam",
    #                 "config": {"lr": {
    #                     "class_name": "ExponentialDecay",
    #                     "config": {"initial_learning_rate": 0.001,
    #                                "decay_steps": 5800,
    #                                "decay_rate": 0.5, "staircase":  False}
    #                     }
    #                 }
    #             },
    #             "loss": "mean_absolute_error"
    #         },

    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
    #         "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
    #     },

    #     "data": {
    #         "dataset": {
    #             "class_name": "SampleAllMPFormDataset",
    #             "module_name": "kgcnn.data.datasets.SampleAllMPFormDataset",
    #             "config": {},
    #             "methods": [
               
    #                 {"set_representation": {
    #                     "pre_processor": {

    #                         # "class_name": "KNNUnitCell",
    #                         #               "module_name": "kgcnn.crystal.preprocessor",
    #                         #               "config": {"k": 6}

    #                          "class_name": "VoronoiUnitCell",
    #                                       "module_name": "kgcnn.crystal.preprocessor",
    #                                       "config": {"min_ridge_area": 0.1}


    #                         #  "class_name": "RadiusUnitCell",
    #                         #               "module_name": "kgcnn.crystal.preprocessor",
    #                         #               "config": {"radius":5.0 } ,

    #                                       },
    #                     "reset_graphs": False}},

    #             ]
    #         },
    #         "data_unit": ""
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "",
    #         "kgcnn_version": "2.0.3"
    #     }
    # },




    "DenseGNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        "AGNIFinger": {"shape": (None,128), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                       
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128, 

                   'atomic_mass': True, 
                   'atomic_radius': True, 
                   'electronegativity': True, 
                   'oxidation_states': True, 
                   'ionization_energy': True, 
                   
                   'melting_point':True,    
                    'density':True,             

                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 25,
                                           'max_voronoi_area': 32}},


                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 1,
              
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },


                "gin_args": {"pooling_method":"mean", 
                             "edge_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]}, 
                             "concat_args": {"axis": -1}, },

                "g_pooling_args": {"pooling_method": "mean"},

                "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
                             "activation": ['swish', 'swish', 'linear']},
            }
        },
        "training": {
            "fit": {"batch_size": 256, "epochs": 100, "validation_freq": 20, "verbose": 2, "callbacks": []},
           
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                        "class_name": "ExponentialDecay",
                        "config": {"initial_learning_rate": 0.001,
                                   "decay_steps": 5800,
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
            # "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "SampleAllMPFormDataset",
                "module_name": "kgcnn.data.datasets.SampleAllMPFormDataset",
                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {


                             "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}

                            #  "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12},

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },


}
