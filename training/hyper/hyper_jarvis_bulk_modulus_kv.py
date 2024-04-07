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
                # 'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},

                'node_ff_args': {"units": [64], "activation": "kgcnn>softplus2"},
                'edge_ff_args': {"units": [64], "activation": "kgcnn>softplus2"},
                'state_ff_args': {"units": [64], "activation": "kgcnn>softplus2"},
                'nblocks': 3, 'has_ff': True, 'dropout': None, 
                # 'use_set2set': True,
                'verbose': 10,
        
                "gin_mlp": {"units": [64], "use_bias": True, "activation": ["swish"],
                            },
                "pooling_args": {"pooling_method": "mean"},

                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": ['kgcnn>softplus2', 'kgcnn>softplus2', 'linear']}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
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
   

     "Schnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Schnet",
            "class_name": "make_crystal_model",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {'shape': (None,), 'name': "node_number", 'dtype': 'int32', 'ragged': True},
                    {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},

                    # {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    # {'shape': (None,3), 'name': "offset", 'dtype': 'float32', 'ragged': True},

                    {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False},
                    {'shape': (None, 24), 'name': "sites_feature", 'dtype': 'float32', 'ragged': True},

                ],

                # "input_embedding": {
                #     "node": {"input_dim": 95, "output_dim": 64}
                # },

                "input_block_cfg" : {'node_size': 128,
                                     
                    'atomic_mass': True,   #
                   'atomic_radius': False, 
                   'electronegativity': False, 
                   'ionization_energy': False, 
                   'oxidation_states': False, 
                   'melting_point':False,    
                    'density':True,         #
                    'mendeleev':False, 
                    'molarvolume':False,         
                },

                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "swish", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "max"},
                "depth": 4,
                
                "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                    
                "gc_mlp": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3,
                            },
                    

                "last_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
                             "activation": ['swish', 'swish', 'linear']},
                # "output_embedding": "graph",  # 'kgcnn>shifted_softplus'
                # "use_output_mlp": False,
                # "output_mlp": None,  # Last MLP sets output dimension if None.
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2,
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

                    {"map_list": {"method": "set_range_periodic", "max_distance": 4.0}},

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
    

    "PAiNN.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.PAiNN",
            "class_name": "make_crystal_model",
            "config": {
                "name": "PAiNN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                "equiv_initialize_kwargs": {"dim": 3, "method": "eye"},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},

                "pooling_args": {"pooling_method": "mean"},
                "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
                "update_args": {"units": 128}, "depth": 2, "verbose": 10,
                "equiv_normalization": False, "node_normalization": False,
                
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                # "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1], "activation": ["swish", "swish", "linear"]}
            }
        },

        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupLinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-04, "learning_rate_stop": 1e-06, "epo_warmup": 25, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}},  # "clipnorm": 100.0, "clipvalue": 100.0}
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
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}
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


    "DimeNetPP.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": {
                "name": "DimeNetPP",
                "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                           ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                             "embeddings_initializer": {"class_name": "RandomUniform",
                                                                        "config": {"minval": -1.7320508075688772,
                                                                                   "maxval": 1.7320508075688772}}}},
                "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                "cutoff": 5.0, "envelope_exponent": 5,
                "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                "num_targets": 1, "extensive": False, "output_init": "zeros",
                "activation": "swish", "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": {},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 16, "epochs": 780, "validation_freq": 10, "verbose": 2, "callbacks": [],
                "validation_batch_size": 8
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 3000.0, "decay_steps": 4000000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
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
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "set_angle", "allow_multi_edges": True, "allow_reverse_edges": True}}
                ]
            },
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


    "NMPN.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.NMPN",
            "config": {
                "name": "NMPN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
                "pooling_args": {"pooling_method": "segment_mean"},
                "use_set2set": True,
                "depth": 3,
                "node_dim": 128,
                "verbose": 10,
                "geometric_edge": True, "make_distance": True, "expand_distance": True,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [25, 25, 1],
                               "activation": ["selu", "selu", "linear"]},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-04, "learning_rate_stop": 1e-05, "epo_min": 50, "epo": 700,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}},
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
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
   

    "MEGAN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64, "use_embedding": True}},
                'units': [60, 50, 40, 30],
                'importance_units': [],
                'final_units': [50, 30, 10, 1],
                "final_activation": "linear",
                'dropout_rate': 0.3,
                'final_dropout_rate': 0.00,
                'importance_channels': 3,
                'return_importances': False,
                'use_edge_features': True,
                'inputs': [{'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True}],
            }
        },
        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 500,
                "validation_freq": 20,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 50, "epo": 400,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
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
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 40}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}
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
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    # {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    #     "learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 0, "epo": 800,
                    #     "verbose": 0}
                    #  }
                ]
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                            "config": {
                                "dataset_size": 509, "batch_size": 64, "epochs": 800,
                                "lr_start": 0.0005, "lr_stop": 1.0e-05
                            }
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
                "class_name": "JarvisMepszDataset",
                "module_name": "kgcnn.data.datasets.JarvisMepszDataset",
                "config": {},
                "methods": [
                    {"set_representation": {
                        "pre_processor": {"class_name": "KNNAsymmetricUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 24}
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




    # "GIN": {
    #     "model": {
    #         "class_name": "make_model_edge",
    #         "module_name": "kgcnn.literature.GIN",
    #         "config": {
    #             "name": "GIN",
    #             "inputs": [
                    
    #                 # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
    #                 #        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}

    #                     {'shape': (None,), 'name': "node_number", 'dtype': 'int32', 'ragged': True},
    #                     {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
    #                     {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
    #                     {'shape': (None, 24), 'name': "sites_feature", 'dtype': 'float32', 'ragged': True},

    #                        ],
    #             "input_embedding": {
    #                 # "node": {"input_dim": 96, "output_dim": 64},
                                    
    #                 "graph": {"input_dim": 100, "output_dim": 64}
    #                                 },
                
    #             "input_block_cfg" : {'node_size': 128,
                                     
    #                 'atomic_mass': True,   #
    #                'atomic_radius': False, 
    #                'electronegativity': False, 
    #                'ionization_energy': False, 
    #                'oxidation_states': False, 
    #                'melting_point':False,    
    #                 'density':True,         #
    #                 'mendeleev':False, 
    #                 'molarvolume':False,         
    #             },

    #             "depth": 3,
    #             "dropout": 0.05,

    #              "gin_mlp": {"units": [128, 128], "use_bias": True, "activation": ["swish", "linear"],
    #             "use_normalization": True, "normalization_technique": "graph_batch"},

    #             "gc_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
    #                         },
    #             "gl_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
    #                         },
                

    #             "node_pooling_args": {"pooling_method": "mean"},

    #             "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
    #                          "activation": ['swish', 'swish', 'linear']},
    #         }
    #     },
    #     "training": {
    #         "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},
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
    #             "class_name": "MatProjectJdft2dDataset",
    #             "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
    #             "config": {},
    #             "methods": [
                    
    #                 {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
    #                 {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
    #                               "expand_dims": False}}
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



    "GATv2": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GATv2",
            "config": {
                "name": "GATv2",
                "inputs": [
                    # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}

                           {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},

                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 8, "output_dim": 64}},

                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "mean"},
                "depth": 4, "attention_heads_num": 4,
                "attention_heads_concat": True, "verbose": 10,
                # "output_embedding": "graph",

                "gin_mlp": {"units": [64], "use_bias": True, "activation": ["swish"],
                            },
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["swish", "swish", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 250, "epo": 500,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    # {"set_attributes": {}}
                            
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}

                            ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },



    "INorp": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.INorp",
            "config": {
                "name": "INorp",
                "inputs": [

                    # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    # {"shape": [], "name": "graph_size", "dtype": "float32", "ragged": False}

                           {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                            {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},


                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 15, "output_dim": 64},
                                    "graph": {"input_dim": 32, "output_dim": 64}},
                "set2set_args": {"channels": 64, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
                "node_mlp_args": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": [64, 64], "activation": ["relu", "linear"]},
                "pooling_args": {"pooling_method": "segment_sum"},
                "depth": 4, "use_set2set": False, "verbose": 10,
                "gather_args": {},
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 64, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 500,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    # {"set_attributes": {}}
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },



    "GraphSAGE": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphSAGE",
            "config": {
                "name": "GraphSAGE",
                "inputs": [

                    # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
                    
                    {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},

                    ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 32, "output_dim": 64}},

                # "node_mlp_args": {"units": [64, 64, 32], "use_bias": True, "activation": ["relu","relu","relu"]},
                # "edge_mlp_args": {"units": [64], "use_bias": True, "activation": ["relu"]},
                # 'node_ff_args': {"units": 32, "activation": "linear"},
                # 'edge_ff_args': {"units": 64, "activation": "linear"},

                "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3},

                'node_ff_args': {"units": 128, "activation": "linear"},
                'edge_ff_args': {"units": 128, "activation": "linear"},

                # "pooling_args": {"pooling_method": "LSTM","units": 64 }, "gather_args": {},
                "pooling_args": {"pooling_method": "sum"}, "gather_args": {},
                "concat_args": {"axis": -1},
                "use_edge_features": True,
                "pooling_nodes_args": {"pooling_method": "mean"},
                "depth": 3, "verbose": 10,
                "output_embedding": "graph",
               
                  "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],},
                "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
                               "activation": ["swish", "swish", "linear"]},
            }
        },
        "training": {
            "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2,
                    "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                                   "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                              "epo_min": 400, "epo": 500, "verbose": 0}}]
                    },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 5e-3}},
                        "loss": "mean_absolute_error"
                        },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    # {"set_attributes": {}}

                     {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },


    "AttentiveFP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFP",
            "config": {
                "name": "AttentiveFP",
                "inputs": [

                    # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    #        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    #        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}

                            {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                           
                           ],
                "input_embedding": {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                    "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                "attention_args": {"units": 64},
                "depthato": 2, 
                # "depthmol": 5,
                'node_pooling_args': {'pooling_method': 'mean'},
                "gin_mlp": {"units": [64], "use_bias": True, "activation": ["swish"],
                            },
                            
                "dropout": 0.2,
                # "verbose": 10,
                # "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [64, 32, 1],
                               "activation": ["kgcnn>leaky_relu", "kgcnn>leaky_relu", "linear"]},
            }
        },
        "training": {
            "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": 0.0031622776601683794, "decay": 1e-05
                                         }
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    # {"set_attributes": {}}
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}
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


     "HamNet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.HamNet",
            "config": {
                "name": "HamNet",
                "inputs": [
                    # {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    # {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    # {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True}

                    {'shape': (None,), 'name': "node_number", 'dtype': 'int32', 'ragged': True},
                    {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                    {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {'shape': (None, 24), 'name': "sites_feature", 'dtype': 'float32', 'ragged': True},

                ],

                "input_embedding": {
                    # "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 5, "output_dim": 64}
                    },

                "message_kwargs": {"units": 128,
                                   "units_edge": 128,
                                   "rate": 0.2, "use_dropout": True},

                "input_block_cfg" : {'node_size': 128,
                    'atomic_mass': True,   # 
                   'atomic_radius': False, 
                   'electronegativity': False, 
                   'ionization_energy': False, 
                   'oxidation_states': False, 
                   'melting_point':False,    
                    'density':True,         # 
                    'mendeleev':False, 
                    'molarvolume':False,         
                },


                "gru_kwargs": {"units": 128},
                "verbose": 10, "depth": 3,
                "union_type_node": "gru",
                "union_type_edge": "gru",
                "given_coordinates": True,
                # 'output_embedding': 'graph',

                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },
                'node_pooling_args': {'pooling_method': 'mean'},
                "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1], "activation": ["swish", "swish", "linear"]}


            }
        },
        "training": {
            "fit": {
                "batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                # "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                #                                                        "weight_decay": 1e-05}},
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.001,
                                                                       "decay": 1e-05}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    
                    # {"set_attributes": {}}

                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}

                    ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },




    "DenseGNN": {
        "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        "CrystalNNFinger": {"shape": (None,24), "name": "CrystalNNFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128, 
                   'atomic_mass': True, 
                   'atomic_radius': True, 
                   'electronegativity': False, 
                   'ionization_energy': True, 
                   'oxidation_states': True, 
                   'melting_point':True,    
                    'density':True,            

                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 32,
                                           'max_voronoi_area': 32}},


                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'nested_blocks_cfgs': None,
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128,
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", 
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },

            }
        },
        "training": {
            "fit": {"batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},
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
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "JarvisBulkModulusKvDataset",
                "module_name": "kgcnn.data.datasets.JarvisBulkModulusKvDataset",
                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {

                            # "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12},


                            "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}

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
