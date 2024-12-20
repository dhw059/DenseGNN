from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class SampleAllMPFormIncludeLiMoreThanNEG2Dataset(MatBenchDataset2020):
    r"""Store and process :obj:`SampleAllMPFormIncludeLiMoreThanNEG2Dataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database.  just include more than -2 li compounds

    Matbench test dataset for predicting DFT PBE band gap from structure.
    Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull)
    more than 150meV and those containing noble gases. Retrieved April 2, 2019.
    For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data;
    refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 10367
        * Task type: regression
        * Input type: structure

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize 'formation energy' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        super(SampleAllMPFormIncludeLiMoreThanNEG2Dataset, self).__init__("sample_all_materials_data_form_processed_include_li_more_than_neg2", reload=reload, verbose=verbose)
        self.label_names = "e_form"
        self.label_units = "eV/atom"
