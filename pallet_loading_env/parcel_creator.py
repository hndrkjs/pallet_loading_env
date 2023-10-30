#from torch.utils.data import Dataset
import numpy as np
from environment.utils import generate_normal_parcel_set


class ParcelCreator():
    """
    A class that generates a set of parcels.
    """

    def __init__(self, length, width, max_height, max_weight) -> None:
        """
        Initializes the parcel creator.

        Args:
            length (int): The length of the pallet.
            width (int): The width of the pallet.
            max_height (int): The maximum height of the pallet.
            max_weight (int): The maximum weight of the pallet.
        """
        super().__init__()
        self.length = length
        self.width = width
        self.max_height = max_height
        self.max_weight = max_weight

        self.parcel_data = []

        num_parcels = np.random.randint(50, 100)

        self.parcel_data = generate_normal_parcel_set(num_parcels)    

    def generate_new_instance(self):
        """
        Generates a new instance of parcels.
        """
        num_parcels = np.random.randint(50, 100)
        self.parcel_data = generate_normal_parcel_set(num_parcels)

        return num_parcels, self.parcel_data
        
    def reset(self):
        """
        Resets the parcel creator.
        """
        self.parcel_data = []

    def __len__(self):
        """
        Returns the number of parcels.
        """
        return len(self.parcel_data)

    def __getitem__(self, index):
        """
        Returns a parcel.
        """
        return self.parcel_data