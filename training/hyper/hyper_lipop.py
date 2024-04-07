hyper = {
    
    "DMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNN",
            "config": {
                "name": "DMPNN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 5, "output_dim": 64}
                },
                "pooling_args": {"pooling_method": "sum"},
                "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depth": 5,
                "dropout": {"rate": 0.1},
                "output_embedding": "graph",
                "output_mlp": {
                    "use_bias": [True, True, False], "units": [64, 32, 1],
                    "activation": ["relu", "relu", "linear"]
                },
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 5800,
                                             "decay_rate": 0.5, "staircase": False}
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
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

    "CMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CMPNN",
            "config": {
                "name": "CMPNN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "node_initialize": {"units": 300, "activation": "relu"},
                "edge_initialize": {"units": 300, "activation": "relu"},
                "edge_dense": {"units": 300, "activation": "linear"},
                "node_dense": {"units": 300, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "verbose": 10,
                "depth": 5,
                "dropout": None,
                "use_final_gru": True,
                "pooling_gru": {"units": 300},
                "pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_mlp": {
                    "use_bias": [True, False], "units": [300, 1],
                    "activation": ["relu", "linear"]
                }
            }
        },
        "training": {
            "fit": {"batch_size": 50, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_squared_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    
    
    "AttentiveFP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFP",
            "config": {
                "name": "AttentiveFP",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                    "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                "attention_args": {"units": 200},
                "depthato": 2, "depthmol": 3,
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [200, 1],
                               "activation": ["kgcnn>leaky_relu", "linear"]},
            }
        },
        "training": {
            "fit": {"batch_size": 200, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
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


    "PAiNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.PAiNN",
            "config": {
                "name": "PAiNN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
                "update_args": {"units": 128}, "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
            }
        },
        "training": {
            "fit": {"batch_size": 128, "epochs": 250, "validation_freq": 20, "verbose": 2,
                    "callbacks": []
                    },
            "compile": {
                # "optimizer": {
                #     "class_name": "Addons>MovingAverage", "config": {
                #         "optimizer": {
                #             "class_name": "Adam", "config": {
                #                 "learning_rate": {
                #                     "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                #                         "learning_rate": 0.0005, "warmup_steps": 30.0, "decay_steps": 40000.0,
                #                         "decay_rate": 0.01
                #                     }
                #                 }, "amsgrad": True
                #             }
                #         },
                #         "average_decay": 0.999
                #     }
                # },
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}}, 
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 10000}}
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
    
    "GIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GIN",
            "config": {
                "name": "GIN",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64}},
                "depth": 5,
                "dropout": 0.05,
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": True, "normalization_technique": "graph_batch"},
                "gin_args": {},
                "last_mlp": {"use_bias": True, "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
                "output_embedding": "graph",
                "output_mlp": {"activation": "linear", "units": 1},
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
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
    
    "INorp": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.INorp",
            "config": {
                "name": "INorp",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [], "name": "graph_size", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 32},
                                    "edge": {"input_dim": 15, "output_dim": 32},
                                    "graph": {"input_dim": 30, "output_dim": 32}},
                "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
                "node_mlp_args": {"units": [32, 32], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": [32, 32], "activation": ["relu", "linear"]},
                "pooling_args": {"pooling_method": "segment_sum"},
                "depth": 3, "use_set2set": False, "verbose": 10,
                "gather_args": {},
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [32, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 300, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 300, "epo": 500,
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
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
    
    "GAT": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GAT",
            "config": {
                "name": "GAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 8, "output_dim": 64}},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "sum"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 300, "validation_freq": 2, "verbose": 2,
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
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
    
    "GATv2": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GATv2",
            "config": {
                "name": "GATv2",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 8, "output_dim": 64}},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "sum"},
                "depth": 4, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
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
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
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
    
    "DimeNetPP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": {
                "name": "DimeNetPP",
                "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                             "embeddings_initializer": {"class_name": "RandomUniform",
                                                                        "config": {"minval": -1.7320508075688772,
                                                                                   "maxval": 1.7320508075688772}}}},
                "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                "cutoff": 5.0, "envelope_exponent": 5,
                "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                "num_targets": 128, "extensive": False, "output_init": "zeros",
                "activation": "swish", "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, False], "units": [128, 1],
                               "activation": ["swish", "linear"]},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []
            },
            "compile": {
                # "optimizer": {
                #     "class_name": "Addons>MovingAverage", "config": {
                #         "optimizer": {
                #             "class_name": "Adam", "config": {
                #                 "learning_rate": {
                #                     "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                #                         "learning_rate": 0.001, "warmup_steps": 40.0, "decay_steps": 40000.0,
                #                         "decay_rate": 0.01
                #                     }
                #                 }, "amsgrad": True
                #             }
                #         },
                #         "average_decay": 0.999
                #     }
                # },

                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}}, 
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 10000}},
                    {"map_list": {"method": "set_angle"}}
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
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "message_kwargs": {"units": 200,
                                   "units_edge": 200,
                                   "rate": 0.5, "use_dropout": True},
                "fingerprint_kwargs": {"units": 200,
                                       "units_attend": 200,
                                       "rate": 0.5, "use_dropout": True,
                                       "depth": 3},
                "gru_kwargs": {"units": 200},
                "verbose": 10, "depth": 3,
                "union_type_node": "gru",
                "union_type_edge": "None",
                "given_coordinates": True,
                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, False], "units": [200, 1],
                               "activation": ['relu', 'linear'],
                               "use_dropout": [True, False],
                               "rate": [0.5, 0.0]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 64, "epochs": 400, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [{"set_attributes": {}}]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },
    
    "Schnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
                "output_embedding": "graph",
                'output_mlp': {"use_bias": [True, True], "units": [64, 1],
                               "activation": ['kgcnn>shifted_softplus', "linear"]},
                'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 32, "epochs": 300, "validation_freq": 10, "verbose": 2,
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
            }
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 10000}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    
    "MEGAN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                'units': [60, 50, 40, 30],
                'importance_units': [],
                'final_units': [50, 30, 10, 1],
                'dropout_rate': 0.4,
                'final_dropout_rate': 0.00,
                'importance_channels': 4,
                'return_importances': False,
                'use_edge_features': True,
                'concat_heads': True,
                'inputs': [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                ],
            }
        },
        "training": {
            "fit": {
                "batch_size": 128,
                "epochs": 300,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler",
                        "config": {
                            "learning_rate_start": 1e-03,
                            "learning_rate_stop": 1e-05,
                            "epo_min": 200,
                            "epo": 400,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {"lr": 1e-03}
                },
                "loss": "mean_squared_error"
            },
            "cross_validation": {
                "class_name": "KFold",
                "config": {
                    "n_splits": 5,
                    "random_state": 42,
                    "shuffle": True
                }
            },
            "scaler": {
                "class_name": "StandardScaler",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 10000}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },

    "DGIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DGIN",
            "config": {
                "name": "DGIN",
                "inputs": [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "gin_mlp": {"units": [64,64], "use_bias": True, "activation": ["relu","linear"],
                            "use_normalization": True, "normalization_technique": "graph_layer"},
                "gin_args": {},
                "pooling_args": {"pooling_method": "sum"},
                "use_graph_state": False,
                "edge_initialize": {"units": 100, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 100, "use_bias": True, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 100, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depthDMPNN": 5,"depthGIN": 5, 
                "dropoutDMPNN": {"rate": 0.05},
                "dropoutGIN": {"rate": 0.05},
                "output_embedding": "graph", "output_to_tensor": True,
                "last_mlp": {"use_bias": [True, True], "units": [64, 32],
                             "activation": ["relu", "relu"]},
                "output_mlp": {"use_bias": True, "units": 1,
                               "activation": "linear"}
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }
,

    "MXMNet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MXMNet",
            "config": {
                "name": "MXMNet",
                "inputs": [{"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": (None, 1), "name": "edge_weights", "dtype": "float32", "ragged": True},
                           {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64", "ragged": True},
                           {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 32},
                                    "edge": {"input_dim": 5, "output_dim": 32}},
                "bessel_basis_local": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                "bessel_basis_global": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0, "envelope_exponent": 5},
                "mlp_rbf_kwargs": {"units": 32, "activation": "swish"},
                "mlp_sbf_kwargs": {"units": 32, "activation": "swish"},
                "global_mp_kwargs": {"units": 32},
                "local_mp_kwargs": {"units": 32, "output_units": 1,
                                    "output_kernel_initializer": "glorot_uniform"},
                "use_edge_attributes": False,
                "depth": 4,
                "verbose": 10,
                "node_pooling_args": {"pooling_method": "sum"},
                "output_embedding": "graph", "output_to_tensor": True,
                "use_output_mlp": False,
                "output_mlp": {"use_bias": [True], "units": [1],
                               "activation": ["linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 128, "epochs": 300, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.9961697, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 45}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03, "global_clipnorm": 1000}},
                "loss": "mean_absolute_error",
                "metrics": [
                    "mean_absolute_error", "mean_squared_error",
                    # No scaling needed.
                    {"class_name": "RootMeanSquaredError", "config": {"name": "scaled_root_mean_squared_error"}},
                    {"class_name": "MeanAbsoluteError", "config": {"name": "scaled_mean_absolute_error"}},
                ]
            },

            # "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_edge_weights_uniform"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 1000}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "jk",
                                  "angle_indices": "angle_indices_1",
                                  "angle_indices_nodes": "angle_indices_nodes_1",
                                  "angle_attributes": "angle_attributes_1"}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "ik",
                                  "allow_self_edges": True,
                                  "angle_indices": "angle_indices_2",
                                  "angle_indices_nodes": "angle_indices_nodes_2",
                                  "angle_attributes": "angle_attributes_2"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.1"
        }
    },

   
    "RGCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.RGCN",
            "config": {
                "name": "RGCN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None], "name": "edge_number", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
                "dense_relation_kwargs": {"units": 64, "num_relations": 20},
                "dense_kwargs": {"units": 64},
                "activation_kwargs": {"activation": "swish"},
                "depth": 5, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 300,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 250, "epo": 800,
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
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.0"
        }
    },


    "DenseGNN": {
        "model": {
            "class_name": "make_qm9",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "node_number": {"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                        "node_coordinates": {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                        "range_indices": {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                        "graph_attributes": {"shape": [2], "name": "graph_attributes", "dtype": "float32", "ragged": False},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'atomic_mass': True, 
                   'atomic_radius': True, 
                   'electronegativity': True, 
                   'ionization_energy': True, 
                   'oxidation_states': True, 
                   'melting_point':True,    
                    'density':True,      
                #  'mendeleev':False, 
                #    'molarvolume':False,
                #    'vanderwaals_radius':False,         

                                           },

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
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
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
          
            "fit": {
                    "batch_size": 128, "epochs": 600, "validation_freq": 20, "verbose": 2,
                    "callbacks": [
                        {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                            "verbose": 0
                        }
                        }
                    ]
                },

            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},

            # "scaler": {"class_name": "StandardScaler", "config": {"scaler": [{"class_name": "ExtensiveMolecularScaler","config": {}}]}
            #            },
            # "multi_target_indices": [6]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G'  or combination
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },


        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "module_name": "kgcnn.data.datasets.LipopDataset",
                "config": {},
                "methods": [
               
                 {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 20}}

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
