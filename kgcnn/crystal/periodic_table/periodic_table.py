import pandas as pd
from typing import Optional
import importlib.resources
import kgcnn.crystal.periodic_table as periodic_table_module

# CSV file is downloaded from:
# https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV/?response_type=save&response_basename=PubChemElements_all

try:
    # >= Python 3.9
    periodic_table_csv = importlib.resources.files(periodic_table_module) / 'periodic_table.csv'
except:
    # < Python 3.9
    with importlib.resources.path(periodic_table_module, 'periodic_table.csv') as file_name:
        periodic_table_csv = file_name


class PeriodicTable:
    """Utility class to provide further element type information for crystal graph node embeddings."""
    
    def __init__(self, csv_path=periodic_table_csv,
                 normalize_atomic_mass=True,
                 normalize_atomic_radius=True,
                 normalize_electronegativity=True,
                 normalize_ionization_energy=True,
                 normalize_melting_point = True,
                 normalize_density = True,
                 normalize_mendeleev = True,
                 normalize_molarvolume = True,
                 normalize_vanderwaals_radius = True,
                 normalize_average_cationic_radius = True,
                 normalize_average_anionic_radius = True,
                 normalize_velocity_sound = True,
                 normalize_thermal_conductivity = True,
                 normalize_electrical_resistivity = True,
                 normalize_rigidity_modulus = True,
                 imputation_atomic_radius=209.46,  # mean value
                 imputation_electronegativity=1.18,  # educated guess (based on neighbour elements)
                 imputation_ionization_energy=8.,   # mean value
                 imputation_melting_point=1287.,  # median value
                 imputation_density=7.69,
                 imputation_mendeleev=23.0,  # 53.0  23.0
                 imputation_molarvolume=17.83,  # 17.83
                 imputation_vanderwaals_radius=2.17,  # VanDerWaalsRadius
                imputation_velocity_sound=3552, 
                imputation_thermal_conductivity=23, 
                imputation_electrical_resistivity=0, 
                imputation_rigidity_modulus=47, 

                 ):  
        self.data = pd.read_csv(csv_path)
        self.data['AtomicRadius'].fillna(imputation_atomic_radius, inplace=True)
        # Pm, Eu, Tb, Yb are inside the mp_e_form dataset, but have no electronegativity value
        self.data['Electronegativity'].fillna(imputation_electronegativity, inplace=True)
        self.data['IonizationEnergy'].fillna(imputation_ionization_energy, inplace=True)
        self.data['MeltingPoint'].fillna(imputation_melting_point, inplace=True)
        self.data['Density'].fillna(imputation_density, inplace=True)
        self.data['Mendeleev'].fillna(imputation_mendeleev, inplace=True)
        self.data['MolarVolume'].fillna(imputation_molarvolume, inplace=True)
        self.data['VanDerWaalsRadius'].fillna(imputation_vanderwaals_radius, inplace=True)
        self.data['VelocitySound'].fillna(imputation_velocity_sound, inplace=True)
        self.data['ThermalConductivity'].fillna(imputation_thermal_conductivity, inplace=True)
        self.data['ElectricalResistivity'].fillna(imputation_electrical_resistivity, inplace=True)
        self.data['RigidityModulus'].fillna(imputation_rigidity_modulus, inplace=True)


        if normalize_atomic_mass:
            self._normalize_column('AtomicMass')
        if normalize_atomic_radius:
            self._normalize_column('AtomicRadius')
        if normalize_electronegativity:
            self._normalize_column('Electronegativity')
        if normalize_ionization_energy:
            self._normalize_column('IonizationEnergy')
        if normalize_melting_point:
            self._normalize_column('MeltingPoint')
        if normalize_density:
            self._normalize_column('Density')
        if normalize_mendeleev:
            self._normalize_column('Mendeleev')
        if normalize_molarvolume:
            self._normalize_column('MolarVolume')
        if normalize_vanderwaals_radius:
            self._normalize_column('VanDerWaalsRadius')
        if normalize_average_cationic_radius:
            self._normalize_column('AverageCationicRadius')
        if normalize_average_anionic_radius:
            self._normalize_column('AverageAnionicRadius')
        if normalize_velocity_sound:
            self._normalize_column('VelocitySound')
        if normalize_thermal_conductivity:
            self._normalize_column('ThermalConductivity')
        if normalize_electrical_resistivity:
            self._normalize_column('ElectricalResistivity')
        if normalize_rigidity_modulus:
            self._normalize_column('RigidityModulus')

    def _normalize_column(self, column):
        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def get_symbol(self, z: Optional[int] = None):
        if z is None:
            return self.data['Symbol'].to_list()
        else:
            return self.data.loc[z-1]['Symbol']
    
    def get_atomic_mass(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicMass'].to_list()
        else:
            return self.data.loc[z-1]['AtomicMass']
    
    def get_rigidity_modulus(self, z: Optional[int] = None):
        if z is None:
            return self.data['RigidityModulus'].to_list()
        else:
            return self.data.loc[z-1]['RigidityModulus']
        
    def get_electrical_resistivity(self, z: Optional[int] = None):
        if z is None:
            return self.data['ElectricalResistivity'].to_list()
        else:
            return self.data.loc[z-1]['ElectricalResistivity']
        
    def get_thermal_conductivity(self, z: Optional[int] = None):
        if z is None:
            return self.data['ThermalConductivity'].to_list()
        else:
            return self.data.loc[z-1]['ThermalConductivity']
        
    def get_velocity_sound(self, z: Optional[int] = None):
        if z is None:
            return self.data['VelocitySound'].to_list()
        else:
            return self.data.loc[z-1]['VelocitySound']
        
    def get_average_cationic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['AverageCationicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AverageCationicRadius']
    
    def get_average_anionic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['AverageAnionicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AverageAnionicRadius']
    
    def get_atomic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AtomicRadius']
        
    def get_vanderwaals_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['VanDerWaalsRadius'].to_list()
        else:
            return self.data.loc[z-1]['VanDerWaalsRadius']
    
    def get_electronegativity(self, z: Optional[int] = None):
        if z is None:
            return self.data['Electronegativity'].to_list()
        else:
            return self.data.loc[z-1]['Electronegativity']
    
    def get_ionization_energy(self, z: Optional[int] = None):
        if z is None:
            return self.data['IonizationEnergy'].to_list()
        else:
            return self.data.loc[z-1]['IonizationEnergy']

    def get_oxidation_states(self, z: Optional[int] = None):
        if z is None:
            return list(map(self.parse_oxidation_state_string, self.data['OxidationStates'].to_list()))
        else:
            oxidation_states = self.data.loc[z-1]['OxidationStates']
            return self.parse_oxidation_state_string(oxidation_states, encode=True)
        
    def get_melting_point(self, z: Optional[int] = None):
        if z is None:
            return self.data['MeltingPoint'].to_list()
        else:
            return self.data.loc[z-1]['MeltingPoint']
    
    def get_density(self, z: Optional[int] = None):
        if z is None:
            return self.data['Density'].to_list()
        else:
            return self.data.loc[z-1]['Density']
    
    def get_mendeleev(self, z: Optional[int] = None):
        if z is None:
            return self.data['Mendeleev'].to_list()
        else:
            return self.data.loc[z-1]['Mendeleev']
        
    def get_molarvolume(self, z: Optional[int] = None):
        if z is None:
            return self.data['MolarVolume'].to_list()
        else:
            return self.data.loc[z-1]['MolarVolume']
        
    @staticmethod
    def parse_oxidation_state_string(s: str, encode: bool = True):
        if encode:
            oxidation_states = [0] * 14
            if isinstance(s, float):  # False
                return oxidation_states
            for i in s.split(','):
                oxidation_states[int(i)-7] = 1
        else:
            oxidation_states = []
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(','):
                oxidation_states.append(int(i))
        return oxidation_states
