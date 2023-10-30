import numpy as np
#import torch
import random
from environment.parcel import Parcel
from environment.parcel_types import get_parcel_type

def generate_normal_parcel_set(num_parcels=60, parcel_type="random"):
    """
    Samples parcels from a normal distribution.

    Args:
        length (int): The length of the pallet.
        width (int): The width of the pallet.
        num_parcels (int, optional): The number of parcels to sample. Defaults to 60.   

    Returns:
        parcel_dataset (list): A list of parcels.
    """
    #PARCEL_TYPES = get_parcel_type(parcel_type)
    parcel_dataset = []

    length = 0
    width = 0
    height = 0
    weight = 0
    compressibility_index = 0

    for _ in range(num_parcels):



        # sample the parcel type
        #parcel_type = random.choices(PARCEL_TYPES, weights=[t["probability"] for t in PARCEL_TYPES])[0]

        if parcel_type == "dhl_max_dims":
            # sample the parcel dimensions
            length = np.random.uniform(*parcel_type["length_range"])
            width = np.random.uniform(*parcel_type["width_range"])
            height = np.random.uniform(*parcel_type["height_range"])
            weight = np.random.uniform(*parcel_type["weight_range"])
            compressibility_index = np.random.uniform(*parcel_type["compressibility_index_range"])
        elif parcel_type == "dhl":
            # sample the parcel dimensions
            length = parcel_type["length"]
            width = parcel_type["width"]
            height = parcel_type["height"]
            weight = np.random.uniform(*parcel_type["weight_range"])
            compressibility_index = np.random.uniform(*parcel_type["compressibility_index_range"])
        elif parcel_type == "random":
            # sample the parcel dimensions as integers
            length = np.random.randint(1, 100)
            width = np.random.randint(1, 80)
            height = np.random.randint(1, 100)
            weight = np.random.randint(1, 30)
            compressibility_index = np.random.randint(60, 100)

        # create the parcel from the sampled paramters
        sampled_parcel = Parcel(length, width, height, weight, compressibility_index)
        parcel_dataset.append(sampled_parcel)

    return parcel_dataset
