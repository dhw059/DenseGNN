from kgcnn.data.datasets.JarvisBenchDataset2021 import JarvisBenchDataset2021


class JarvisOptb88vdwTotalEnergyDataset(JarvisBenchDataset2021):
    """Store and process :obj:`MatProjectJdft2dDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_jdft2d'.

    Matbench test dataset for predicting exfoliation energies from crystal structure
    (computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT database.
    For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data;
    refer to the Automatminer/Matbench publication for more details.

        * n_train: 44569
        * n_val: 5527
        * n_test: 5527
        * Task type: regression
        * Input type: structure

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize 'optb88vdw_total_energy' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        super(JarvisOptb88vdwTotalEnergyDataset, self).__init__("optb88vdw_total_energy", reload=reload, verbose=verbose)
        self.label_names = "optb88vdw_total_energy "
        self.label_units = "Ev/atom"

