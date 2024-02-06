import os
import pandas as pd

from kgcnn.data.crystal import CrystalDataset
from kgcnn.data.utils import load_json_file, save_json_file

class JarvisBenchDataset2021(CrystalDataset):

    datasets_download_info = {

        "exfoliation_energy": {"dataset_name": "exfoliation_energy",
                            "data_directory_name": "exfoliation_energy"},
        
        "bulk_modulus_kv": {"dataset_name": "bulk_modulus_kv",
                            "data_directory_name": "bulk_modulus_kv"},

        "mepsz": {"dataset_name": "mepsz",
                            "data_directory_name": "mepsz"},

        "shear_modulus_gv": {"dataset_name": "shear_modulus_gv",
                            "data_directory_name": "shear_modulus_gv"},
                            
        "spillage": {"dataset_name": "spillage",
                            "data_directory_name": "spillage"},

        "mepsx": {"dataset_name": "mepsx",
                            "data_directory_name": "mepsx"},

        "epsz": {"dataset_name": "epsz",
                            "data_directory_name": "epsz"},
        "epsx": {"dataset_name": "epsx",
                            "data_directory_name": "epsx"},
        "magmom_oszicar": {"dataset_name": "magmom_oszicar",
                            "data_directory_name": "magmom_oszicar"},

        "slme": {"dataset_name": "slme",
                            "data_directory_name": "slme"},

        "n-Seebeck": {"dataset_name": "n-Seebeck",
                            "data_directory_name": "n-Seebeck"},     

        "formation_energy_peratom": {"dataset_name": "formation_energy_peratom",
                            "data_directory_name": "formation_energy_peratom"},   
                                      
        "ph_heat_capacity": {"dataset_name": "ph_heat_capacity",
                            "data_directory_name": "ph_heat_capacity"},  

        "ehull": {"dataset_name": "ehull",
                            "data_directory_name": "ehull"},    
        
        "avg_hole_mass": {"dataset_name": "avg_hole_mass",
                            "data_directory_name": "avg_hole_mass"},   
        
        "avg_elec_mass": {"dataset_name": "avg_elec_mass",
                            "data_directory_name": "avg_elec_mass"},  

        "dfpt_piezo_max_dielectric": {"dataset_name": "dfpt_piezo_max_dielectric",
                            "data_directory_name": "dfpt_piezo_max_dielectric"},  
        
        "encut": {"dataset_name": "encut",
                            "data_directory_name": "encut"},  

        "dfpt_piezo_max_dij": {"dataset_name": "dfpt_piezo_max_dij",
                            "data_directory_name": "dfpt_piezo_max_dij"},  
        

        "optb88vdw_total_energy": {"dataset_name": "optb88vdw_total_energy",
                            "data_directory_name": "optb88vdw_total_energy"},

        "mepsy": {"dataset_name": "mepsy",
                            "data_directory_name": "mepsy"},
        
        "n_powerfact": {"dataset_name": "n_powerfact",
                            "data_directory_name": "n_powerfact"},

        "kpoint_length_unit": {"dataset_name": "kpoint_length_unit",
                            "data_directory_name": "kpoint_length_unit"},

        "optb88vdw_bandgap": {"dataset_name": "optb88vdw_bandgap",
                            "data_directory_name": "optb88vdw_bandgap"},

        "max_efg": {"dataset_name": "max_efg",
                            "data_directory_name": "max_efg"},
        
        "epsy": {"dataset_name": "epsy",
                            "data_directory_name": "epsy"},
        
        "mbj_bandgap": {"dataset_name": "mbj_bandgap",
                            "data_directory_name": "mbj_bandgap"},

        }
                            
    datasets_read_in_memory_info = {
        
        "exfoliation_energy": {"label_column_name": "exfoliation_energy"},
        "mepsz": {"label_column_name": "mepsz"},
        "bulk_modulus_kv": {"label_column_name": "bulk_modulus_kv"},
        "shear_modulus_gv": {"label_column_name": "shear_modulus_gv"},
        "spillage": {"label_column_name": "spillage"},
        "mepsx": {"label_column_name": "mepsx"},
        "epsz": {"label_column_name": "epsz"},
        "epsx": {"label_column_name": "epsx"},
        "magmom_oszicar": {"label_column_name": "magmom_oszicar"},
        "slme": {"label_column_name": "slme"},
        "n-Seebeck": {"label_column_name": "n-Seebeck"},
        "formation_energy_peratom": {"label_column_name": "formation_energy_peratom"},
        "ph_heat_capacity": {"label_column_name": "ph_heat_capacity"},
        "ehull": {"label_column_name": "ehull"},
        "avg_hole_mass": {"label_column_name": "avg_hole_mass"},
        "avg_elec_mass": {"label_column_name": "avg_elec_mass"},
        "dfpt_piezo_max_dielectric": {"label_column_name": "dfpt_piezo_max_dielectric"},
        "encut": {"label_column_name": "encut"},
        "dfpt_piezo_max_dij": {"label_column_name": "dfpt_piezo_max_dij"},
        "optb88vdw_total_energy": {"label_column_name": "optb88vdw_total_energy"},
        "mepsy": {"label_column_name": "mepsy"},
        "n_powerfact": {"label_column_name": "n_powerfact"},
        "kpoint_length_unit": {"label_column_name": "kpoint_length_unit"},
        "optb88vdw_bandgap": {"label_column_name": "optb88vdw_bandgap"},
        "max_efg": {"label_column_name": "max_efg"},
        "epsy": {"label_column_name": "epsy"},
        "mbj_bandgap": {"label_column_name": "mbj_bandgap"},

    }

    def __init__(self, 
                 dataset_name: str, 
                 reload: bool = False, 
                 verbose: int = 10,
                 data_main_dir : str = os.path.join(os.path.expanduser("/home"), "datasets"),
                 
                 ):
        """Initialize a `GraphTUDataset` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("Please provide string identifier for TUDataset.")
        
        CrystalDataset.__init__(self, verbose=verbose, dataset_name=dataset_name, )
        
        self.data_main_dir = data_main_dir
        self.data_directory_name = self.datasets_download_info[dataset_name]['data_directory_name']
        self.data_directory = os.path.join(self.data_main_dir , 'jarvis_dft_3d_'+ self.data_directory_name) 
        self.file_name = "%s.csv" % self.datasets_download_info[dataset_name]['dataset_name']
        self.require_prepare_data = True 
        self.fits_in_memory = True
        self.file_directory = self.data_directory_name

        # jarvis .csv data not include the column name, need add column name
        file_path = os.path.join(self.data_directory, self.file_name) # .csv
        header = pd.read_csv(file_path, nrows=1).columns.tolist()
        if header == ['index',dataset_name]:
            pass
        else:
            data = pd.read_csv(file_path, names=['index', dataset_name])
            data.to_csv(file_path, index=False)

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload,file_column_name='index') # for jarvis .csv need to rewrtite the prepare_data method in subclass.
        if self.fits_in_memory:
            self.read_in_memory(**self.datasets_read_in_memory_info[self.dataset_name]) # {"label_column_name": "n"}
        
