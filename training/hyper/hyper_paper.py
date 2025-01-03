hyper = {

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
                "batch_size": 64, "epochs": 50, "validation_freq": 10, "verbose": 2,
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
                                "dataset_size": 20.0, "batch_size": 64, "epochs": 800,
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
                "class_name": "CrystalDataset",
                "module_name": "kgcnn.data.crystal",
                "config": {"file_directory": "cif_files", "data_directory": "/data/ExampleSmallDataset/", "dataset_name": "ExampleSmallDataset", "file_name": "id_prop.csv"},
                "methods": [
		    {"prepare_data": {"file_column_name": "file", "overwrite": True}},
		    {"read_in_memory": {"label_column_name": "labels"}},
                    {"set_representation": {
                        "pre_processor": {"class_name": "KNNAsymmetricUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 24}
                                          },
                        "reset_graphs": False}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.1"
        }
    },



}
