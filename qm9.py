
    # "DenseGNN": {
    #     "model": {
    #         "class_name": "model_default_EGNN",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": [{"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
    #                        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
    #                        {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
    #                        {"shape": (None, 1), "name": "range_attributes", "dtype": "float32", "ragged": True}],

    #             "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
    #                                 "edge": {"input_dim": 95, "output_dim": 128}},
    #             "depth": 7,
    #             "node_mlp_initialize": {"units": 128, "activation": "linear"},
    #             "euclidean_norm_kwargs": {"keepdims": True, "axis": 2, "square_norm": True},
    #             "use_edge_attributes": True, # False
    #             "edge_mlp_kwargs": {"units": [128, 128], "activation": ["swish", "swish"]},
    #             "edge_attention_kwargs": {"units": 1, "activation": "sigmoid"},
    #             "use_normalized_difference": False,
    #             "expand_distance_kwargs": None,

    #             "coord_mlp_kwargs":  {"units": [128, 1], "activation": ["swish", "linear"]}, #None, or "tanh" at the end
    #             # "coord_mlp_kwargs": None, 
    #             "pooling_coord_kwargs":  {"pooling_method": "mean"}, # None, 
           
    #             "pooling_edge_kwargs": {"pooling_method": "sum"},
    #             "node_normalize_kwargs": None,
    #             "use_node_attributes": False,
    #             "node_mlp_kwargs": {"units": [128, 128], "activation": ["swish", "linear"]},
    #             "use_skip": True,
    #             "verbose": 10,
    #             # "node_decoder_kwargs": {"units": [128, 128], "activation": ["swish", "linear"]},
    #             "node_pooling_kwargs": {"pooling_method": "mean"},
    #             "output_embedding": "graph",
    #             "output_to_tensor": True,
    #             "output_mlp": {"use_bias": [True, True], "units": [128, 1],
    #                            "activation": ["swish", "linear"]},
                
    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },
    #         }
    #     },
    #     "training": {
    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
    #         "fit": {
    #             "batch_size": 256, "epochs": 800, "validation_freq": 20, "verbose": 2,
    #             "callbacks": [
    #                 {"class_name": "kgcnn>CosineAnnealingLRScheduler", "config": {
    #                     "lr_start": 0.5e-03, "lr_min": 0.0, "epoch_max": 800, "verbose": 1}}
    #             ]
    #         },
    #         "compile": {
    #             "optimizer": {"class_name": "Adam", "config": {"lr": 0.5e-03}},
    #             "loss": "mean_absolute_error"
    #         },
    #         "scaler": {"class_name": "QMGraphLabelScaler", "config": {
    #             "scaler": [{"class_name": "ExtensiveMolecularScaler",
    #                         "config": {}}
    #                        ]
    #         }},
    #         "multi_target_indices": [13]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G' or combination
    #         # 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
    #                 {"map_list": {"method": "atomic_charge_representation"}},
    #                 {"map_list": {"method": "set_range", "max_distance": 10, "max_neighbours": 10000}}
    #             ]
    #         },
    #         "data_unit": "eV"
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "_G",
    #         "kgcnn_version": "2.1.1"
    #     }
    # }



    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_model_PAiNN",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": [
    #                 {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
    #                 {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
    #                 {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
    #             ],
    #             "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
    #             "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    #             "pooling_args": {"pooling_method": "sum"}, 
    #             "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
    #             "update_args": {"units": 128}, "depth": 3, "verbose": 10,

    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },

    #             "output_embedding": "graph",
    #             "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
    #         }
    #     },
    #     "training": {
    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},


    #         "fit": {
    #             "batch_size": 256, "epochs": 800, "validation_freq": 20, "verbose": 2, "callbacks": []
    #         },

          

    #         "compile": {
    #             "optimizer": {
    #                 "class_name": "Addons>MovingAverage", "config": {
    #                     "optimizer": {
    #                         "class_name": "Adam", "config": {
    #                             "learning_rate": {
    #                                 "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
    #                                     "learning_rate": 0.001, "warmup_steps": 3000.0, "decay_steps": 4000000.0,
    #                                     "decay_rate": 0.01
    #                                 }
    #                             }, "amsgrad": True
    #                         }
    #                     },
    #                     "average_decay": 0.999
    #                 }
    #             },
    #             "loss": "mean_absolute_error"
    #         },


    #         "scaler": {"class_name": "QMGraphLabelScaler", "config": {
    #             "scaler": [{"class_name": "ExtensiveMolecularScaler",
    #                         "config": {}}
    #                        ]
    #         }},
    #         "multi_target_indices": [4]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G'  or combination
    #         # 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
    #                 {"map_list": {"method": "set_range", "max_distance": 6, "max_neighbours": 24}},
                     
    #             ]
    #         },
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "_alpha",
    #         "kgcnn_version": "2.1.0"
    #     }
    # },
   
   
     
    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_model_HamNet",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": [

    #                 {"shape": [None, ], "name": "node_number", "dtype": "int64", "ragged": True},
    #                 {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
    #                 {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True}

    #             ],

    #             "input_embedding": {
    #                 # "node": {"input_dim": 95, "output_dim": 64},
    #                 "edge": {"input_dim": 1, "output_dim": 64}
    #                 },

    #             "message_kwargs": {"units": 128,
    #                                "units_edge": 128,
    #                                "rate": 0.2, "use_dropout": True},

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

    #             "fingerprint_kwargs": {"units": 128,
    #                                    "units_attend": 128,
    #                                    "rate": 0.5, "use_dropout": True,
    #                                    "depth": 3},

    #             "gru_kwargs": {"units": 128},
    #             "verbose": 10, 
    #             "depth": 3,
    #             "union_type_node": "None",
    #             "union_type_edge": "None",

    #             "given_coordinates": True,
    #             'output_embedding': 'graph',
    #              "gauss_args": {"bins": 20, "distance": 5, "offset": 0.0, "sigma": 0.4},

    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"] },
    #             "gin_mlp_pq": {"units": [3], "use_bias": True, "activation": ["swish"] },

    #             'node_pooling_args': {'pooling_method': 'mean'},
    #             "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1], "activation": ["swish", "swish", "linear"]}


    #         }
    #     },
    #     "training": {
    #         "fit": {
    #             "batch_size": 256, "epochs": 300, "validation_freq": 20, "verbose": 2,
    #             "callbacks": []
    #         },
    #         "compile": {
               
    #             "optimizer": {"class_name": "Adam", "config": {"lr": 0.001,
    #                                                                    "decay": 1e-05}},
    #             "loss": "mean_squared_error"
    #         },
    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},

    #         # "scaler": {"class_name": "StandardScaler",
    #         #            "config": {"with_std": True, "with_mean": True, "copy": True}},


    #         "scaler": {"class_name": "QMGraphLabelScaler", "config": {
    #             "scaler": [{"class_name": "ExtensiveMolecularScaler",
    #                         "config": {}}]}},

    #         "multi_target_indices": [5]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G'  or combination
    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
                    
    #                 # {"set_attributes": {}}

    #                 {"map_list": {"method": "set_range", "max_distance": 6, "max_neighbours": 18}}

    #                 # {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 18}},
    #                 # {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
    #                 #               "expand_dims": False}}

    #                 ]
    #         },
    #         "data_unit": "mol/L"
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "",
    #         "kgcnn_version": "2.0.3"
    #     }
    # },




    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_model_schnet",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": [
    #                 {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
    #                 {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
    #                 {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
    #             ],


    #             "input_block_cfg" : {'node_size': 256, 
                                     
    #                'atomic_mass': True, 
    #                'atomic_radius': True, 
    #                'electronegativity': True, 
    #                'ionization_energy': True, 
    #                'oxidation_states': True, 
    #                'melting_point':True,    
    #                 'density':True, 
    #                 },

    #             "input_embedding": {
    #                 "node": {"input_dim": 95, "output_dim": 64}
    #             },
    #             "last_mlp": {"use_bias": [True, True, True], "units": [256, 128, 1],
    #                          "activation": ['swish', 'swish', 'linear']},

    #             "interaction_args": {
    #                 "units": 256, "use_bias": True, "activation": "swish", "cfconv_pool": "sum"
    #             },

    #             "gin_mlp": {"units": [256], "use_bias": True, "activation": ["swish"], },

    #             "node_pooling_args": {"pooling_method": "mean"},
    #             "depth": 5,
    #             "gauss_args": {"bins": 20, "distance": 5, "offset": 0.0, "sigma": 0.4},
    #             "verbose": 10,

    #             # "output_embedding": "graph",
    #             # "use_output_mlp": False,
    #             # "output_mlp": None,
    #         }
    #     },
    #     "training": {
    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
     
    #         "fit": {
    #             "batch_size": 256, "epochs": 600, "validation_freq": 20, "verbose": 2,
    #             "callbacks": [
    #                 {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
    #                     "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
    #                     "verbose": 0}
    #                  }
    #             ]
    #         },

    #         "compile": {
    #             "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
    #             "loss": "mean_absolute_error"
    #         },

    #         "scaler": {"class_name": "QMGraphLabelScaler", "config": {
    #             "scaler": [{"class_name": "ExtensiveMolecularScaler",
    #                         "config": {}}
    #                        ]
    #         }},
    #         "multi_target_indices": [6]  
    #         # 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'Cv_atom'
    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
    #                 {"map_list": {"method": "set_range", "max_distance": 8, "max_neighbours": 30}}
    #             ]
    #         },
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "_lumo",
    #         "kgcnn_version": "2.1.0"
    #     }
    # },
    




    




    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_model",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": {

    #                     "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
    #                     "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
    #                     "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
    #                     "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
    #                     "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
    #                     "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
    #                        },

    #             "input_block_cfg" : {'node_size': 128,
    #                'edge_size': 128, 
    #                'atomic_mass': True, 
    #                'atomic_radius': True, 
    #                'electronegativity': True, 
    #                'ionization_energy': True, 
    #                'oxidation_states': True, 
    #                'melting_point':True,    
    #                 'density':True,             

    #                'edge_embedding_args': {'bins_distance': 32,
    #                                        'max_distance': 8.0,
    #                                        'distance_log_base': 1.0,
    #                                        'bins_voronoi_area': 25,
    #                                        'max_voronoi_area': 32}},


                    
    #             "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
    #                                 "graph": {"input_dim": 100, "output_dim": 64}
    #                                 },
    #             "depth": 5,
              
    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },


    #             "gin_args": {"pooling_method":"mean", 
    #                          "edge_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]}, 
    #                          "concat_args": {"axis": -1}, },

    #             "g_pooling_args": {"pooling_method": "mean"},

    #             "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
    #                          "activation": ['swish', 'swish', 'linear']},
    #         }
    #     },
    #     "training": {
    #         "fit": {"batch_size": 256, "epochs": 220, "validation_freq": 20, "verbose": 2, "callbacks": []},
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

    #        "scaler": {"class_name": "QMGraphLabelScaler", "config": {"scaler": [{"class_name": "ExtensiveMolecularScaler","config": {}}]}},
    #         "multi_target_indices": [5]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G'  or combination

    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
               
    #                 {"set_representation": {
    #                     "pre_processor": {

    #                         # "class_name": "KNNUnitCell",
    #                         #               "module_name": "kgcnn.crystal.preprocessor",
    #                         #               "config": {"k": 6},

    #                         #  "class_name": "VoronoiUnitCell",
    #                         #               "module_name": "kgcnn.crystal.preprocessor",
    #                         #               "config": {"min_ridge_area": 0.01}

    #                          {"map_list": {"method": "set_range", "max_distance": 8, "max_neighbours": 18}}

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



    # "DenseGNN": {
    #     "model": {
    #         "class_name": "make_qm9_lite",
    #         "module_name": "kgcnn.literature.DenseGNN",
    #         "config": {
    #             "name": "DenseGNN",
    #             "inputs": {

                        
    #                     "node_number": {"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
    #                     "node_coordinates": {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
    #                     "range_indices": {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
    #                     # "graph_attributes": {"shape": [2], "name": "graph_attributes", "dtype": "float32", "ragged": False},
                           
    #                        },

    #             "input_block_cfg" : {'node_size': 128,
    #                'edge_size': 128, 

    #                'atomic_mass': True, 
    #                'atomic_radius': True, 
    #                'electronegativity': True, 
    #                'ionization_energy': True, 
    #                'oxidation_states': True, 
    #                'melting_point':True,    
    #                 'density':True,             

    #                'edge_embedding_args': {'bins_distance': 32,
    #                                        'max_distance': 8.0,
    #                                        'distance_log_base': 1.0,
    #                                        'bins_voronoi_area': None,
    #                                        'max_voronoi_area': None}},

            

    #             "depth": 5,
             
    #             "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"], },


    #             "gin_args": {"pooling_method":"max", 
                             
    #                         #  "edge_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]}, 
    #                         #  "concat_args": {"axis": -1},
                            
    #                            },

    #             "g_pooling_args": {"pooling_method": "mean"},

    #             "output_mlp": {"use_bias": [True, True, False], "units": [128, 64, 1],
    #                          "activation": ['swish', 'swish', 'linear']},
    #         }
    #     },
    #     "training": {
    #         "fit": {"batch_size": 256, "epochs": 500, "validation_freq": 20, "verbose": 2, "callbacks": []},
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
        

    #         "scaler": {"class_name": "QMGraphLabelScaler", "config": {"scaler": [{"class_name": "ExtensiveMolecularScaler","config": {}}]}
    #                    },
    #         "multi_target_indices": [5]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G'  or combination

    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "QM9Dataset",
    #             "module_name": "kgcnn.data.datasets.QM9Dataset",
    #             "config": {},
    #             "methods": [
               
    #               {"map_list": {"method": "set_range", "max_distance": 8, "max_neighbours": 20}}

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
