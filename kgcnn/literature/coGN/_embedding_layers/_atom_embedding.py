import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import LazyMultiply

class AtomEmbedding(GraphBaseLayer):
    """Emebedding layer for atomic number and other other element type properties."""

    default_atomic_number_embedding_args = {"input_dim": 119, "output_dim": 64}

    def __init__(
        self,
        atomic_number_embedding_args=default_atomic_number_embedding_args,
        atomic_mass=None,
        atomic_radius=None,
        electronegativity=None,
        ionization_energy=None,
        oxidation_states=None,
        melting_point=None, 
        density=None, 
        mendeleev=None,
        molarvolume=None,
        vanderwaals_radius=None,
        average_cationic_radius=None,
        average_anionic_radius=None,
        velocity_sound=None,
        thermal_conductivity=None,
        electrical_resistivity=None,
        rigidity_modulus=None,
    
        **kwargs
    ):
        """Initialize the AtomEmbedding Layer.

        Args:
            atomic_number_embedding_args (dict, optional): Embedding arguments which get passed
                to the keras `Embedding` layer for the atomic number.
                Defaults to self.default_atomic_number_embedding_args.
            atomic_mass (list, optional): List of atomic mass ordered by the atomic number.
                If it is not None, the atomic mass gets included in the embedding, otherwise not.
                Defaults to None.
            atomic_radius (list, optional): List of atomic radius ordered by the atomic number.
                If it is not None, the atomic radius gets included in the embedding, otherwise not.
                Defaults to None.
            electronegativity (list, optional): List of electronegativities ordered by the atomic number.
                If it is not None, the electronegativities gets included in the embedding, otherwise not.
                Defaults to None.
            ionization_energy (list, optional): List of ionization energies ordered by the atomic number.
                If it is not None, the ionization energies  gets included in the embedding, otherwise not.
                Defaults to None.
            oxidation_states (list, optional): List of oxidation states ordered by the atomic number.
                If it is not None, the oxidation states gets included in the embedding, otherwise not.
                Defaults to None.
        """

        super().__init__(**kwargs)

        self.lazy_multiply = LazyMultiply()

        self.atomic_mass = (
            tf.constant(atomic_mass, dtype=float) if atomic_mass else None
        )
        self.atomic_radius = (
            tf.constant(atomic_radius, dtype=float) if atomic_radius else None
        )
        self.electronegativity = (
            tf.constant(electronegativity, dtype=float) if electronegativity else None
        )
        self.ionization_energy = (
            tf.constant(ionization_energy, dtype=float) if ionization_energy else None
        )
        self.oxidation_states = (
            tf.constant(oxidation_states, dtype=float) if oxidation_states else None
        )
        self.melting_point = (
            tf.constant(melting_point, dtype=float) if melting_point else None
        )
        self.density = (
            tf.constant(density, dtype=float) if density else None
        )
        self.mendeleev = (
            tf.constant(mendeleev, dtype=float) if mendeleev else None
        )
        self.molarvolume = (
            tf.constant(molarvolume, dtype=float) if molarvolume else None
        )
        self.vanderwaals_radius = (
            tf.constant(vanderwaals_radius, dtype=float) if vanderwaals_radius else None
        )
        self.average_cationic_radius = (
            tf.constant(average_cationic_radius, dtype=float) if average_cationic_radius else None
        )
        self.average_anionic_radius = (
            tf.constant(average_anionic_radius, dtype=float) if average_anionic_radius else None
        )
        self.velocity_sound = (
            tf.constant(velocity_sound, dtype=float) if velocity_sound else None
        )
        self.thermal_conductivity = (
            tf.constant(thermal_conductivity, dtype=float) if thermal_conductivity else None
        )
        self.electrical_resistivity = (
            tf.constant(electrical_resistivity, dtype=float) if electrical_resistivity else None
        )
        self.rigidity_modulus = (
            tf.constant(rigidity_modulus, dtype=float) if rigidity_modulus else None
        )

        self.atomic_number_embedding_args = atomic_number_embedding_args
        self.atomic_number_embedding_layer = ks.layers.Embedding(
            **self.atomic_number_embedding_args
        )

    @staticmethod
    def get_attribute(x, k):
        if isinstance(x, dict):
            assert k in x.keys()
            return x[k]
        else:
            raise ValueError()
        
    @staticmethod
    def get_features(x):
        """Getter for edge/node/graph features.

        If the argument is a Tensor it is returned as it is.
        If the argument is a dict the value for the "features" key is returned.
        """
        if isinstance(x, dict):
            assert "features" in x.keys()
            return x["features"]
        else:
            return x

    def call(self, inputs):

        atomic_numbers = self.get_features(inputs)
        if 'AGNIFinger' in inputs:
            AGNIFinger = self.get_attribute(inputs, 'AGNIFinger')
        else:
            AGNIFinger = None

        idxs = atomic_numbers - 1  # Shifted by one (zero-indexed)
        feature_list = []
        atomic_number_embedding = self.atomic_number_embedding_layer(idxs)
        feature_list.append(atomic_number_embedding)

        if AGNIFinger is not None:
            feature_list.append(AGNIFinger)

        if self.atomic_mass is not None:
            atomic_mass = tf.expand_dims(tf.gather(self.atomic_mass, idxs), -1)
            # atomic_mass_weighted_ = self.lazy_multiply([atomic_mass, multiplicity])
            feature_list.append(atomic_mass)

        if self.rigidity_modulus is not None:
            rigidity_modulus = tf.expand_dims(tf.gather(self.rigidity_modulus, idxs), -1)
            feature_list.append(rigidity_modulus)

        if self.electrical_resistivity is not None:
            electrical_resistivity = tf.expand_dims(tf.gather(self.electrical_resistivity, idxs), -1)
            feature_list.append(electrical_resistivity)

        if self.thermal_conductivity is not None:
            thermal_conductivity = tf.expand_dims(tf.gather(self.thermal_conductivity, idxs), -1)
            feature_list.append(thermal_conductivity)

        if self.average_cationic_radius is not None:
            average_cationic_radius = tf.expand_dims(tf.gather(self.average_cationic_radius, idxs), -1)
            feature_list.append(average_cationic_radius)
        
        if self.average_anionic_radius is not None:
            average_anionic_radius = tf.expand_dims(tf.gather(self.average_anionic_radius, idxs), -1)
            feature_list.append(average_anionic_radius)

        if self.velocity_sound is not None:
            velocity_sound = tf.expand_dims(tf.gather(self.velocity_sound, idxs), -1)
            feature_list.append(velocity_sound)

        if self.atomic_radius is not None:
            atomic_radius = tf.expand_dims(tf.gather(self.atomic_radius, idxs), -1)
            feature_list.append(atomic_radius)

        if self.electronegativity is not None:
            electronegativity = tf.expand_dims(
                tf.gather(self.electronegativity, idxs), -1
            )
            # electronegativity_weighted_ = self.lazy_multiply([electronegativity, multiplicity])
            feature_list.append(electronegativity)

        if self.ionization_energy is not None:
            ionization_energy = tf.expand_dims(
                tf.gather(self.ionization_energy, idxs), -1
            )
            # ionization_energy_weighted_ = self.lazy_multiply([ionization_energy, multiplicity])
            feature_list.append(ionization_energy)

        if self.oxidation_states is not None:
            oxidation_states = tf.gather(self.oxidation_states, idxs)
            # oxidation_states_weighted_ = self.lazy_multiply([oxidation_states, multiplicity])
            feature_list.append(oxidation_states)
        
        if self.melting_point is not None:
            melting_point = tf.expand_dims(
                tf.gather(self.melting_point, idxs), -1
            )
            feature_list.append(melting_point)
        
        if self.density is not None:
            density = tf.expand_dims(
                tf.gather(self.density, idxs), -1
            )
            feature_list.append(density)

        if self.mendeleev is not None:
            mendeleev = tf.expand_dims(
                tf.gather(self.mendeleev, idxs), -1
            )
            feature_list.append(mendeleev)

        if self.molarvolume is not None:
            molarvolume = tf.expand_dims(
                tf.gather(self.molarvolume, idxs), -1
            )
            feature_list.append(molarvolume)
        
        if self.vanderwaals_radius is not None:
            vanderwaals_radius = tf.expand_dims(
                tf.gather(self.vanderwaals_radius, idxs), -1
            )
            # vanderwaals_radius_weighted_ = self.lazy_multiply([vanderwaals_radius, multiplicity])
            feature_list.append(vanderwaals_radius)

        return tf.concat(feature_list, -1)
