import numpy as np
from pallet_loading_env.pallet import Parcel

DHL_MAX_DIMS_PARCEL_TYPES = {
    "extra_small": {"length_range": (15, 35), "width_range": (11, 25), "height_range": (1, 3), "weight_range": (0.5, 1), "compressibility_index_range": (1, 4)},
     "small":  { "length_range": (35, 60), "width_range": (15, 30), "height_range": (1, 15), "weight_range": (0.5, 2), "compressibility_index_range": (1, 4)},
      "medium": { "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (0.5, 5), "compressibility_index_range": (1, 4)},
     "large":  { "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (5, 10), "compressibility_index_range": (1, 4)},
     "extra_large": { "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (10, 31.5), "compressibility_index_range": (1, 4)}
}

DHL_PARCEL_TYPES = {
     "extra_small": {"length": 22.5, "width": 14.5, "height": 3, "weight_range": (0.5, 1), "compressibility_index_range": (1, 4), "probability": 0.15},
     "small": {"length": 25, "width": 17.5, "height": 10, "weight_range": (0.5, 2), "compressibility_index_range": (1, 4), "probability": 0.35},
     "medium": { "length": 37.5, "width": 30, "height": 13.5, "weight_range": (0.5, 5), "compressibility_index_range": (1, 4), "probability": 0.35},
     "large": { "length": 45, "width": 35, "height": 20, "weight_range": (5, 10), "compressibility_index_range": (1, 4), "probability": 0.1},
     "bottle": { "length": 38, "width": 12, "height": 12, "weight_range": (1, 3), "compressibility_index_range": (1, 4), "probability": 0.05}
}

DHL_MAX_DIMS_PARCEL_TYPES_KEYS = np.array(['extra_small', 'small', 'medium', 'large', 'extra_large'])

def generate_normal_parcel_set(num_parcels=60):
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
        parcel_type = np.random.choice(DHL_MAX_DIMS_PARCEL_TYPES_KEYS, p=[0.35,0.4,0.15,0.05,0.05])

        # sample the parcel dimensions
        length = np.random.randint(low=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["length_range"][0], high=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["length_range"][1])
        width = np.random.randint(low=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["width_range"][0], high=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["width_range"][1])
        height = np.random.randint(low=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["height_range"][0], high=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["height_range"][1])
        weight = round(np.random.uniform(low=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["weight_range"][0], high=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["weight_range"][1]), 2)
        compressibility_index = round(np.random.uniform(low=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["compressibility_index_range"][0], high=DHL_MAX_DIMS_PARCEL_TYPES[parcel_type]["compressibility_index_range"][1]),2)

        # create the parcel from the sampled paramters
        sampled_parcel = Parcel(length, width, height, weight, compressibility_index)
        parcel_dataset.append(sampled_parcel)

    return parcel_dataset

def generate_easy_parcel_set(num_parcels=20):
    '''
    Generates a set of easy parcels.

    Args:
        num_parcels (int, optional): The number of parcels to sample. Defaults to 20.

    Returns:
        parcel_dataset (list): A list of parcels.

    '''

    parcel_dataset = []

    for _ in range(num_parcels):

        # sample the parcel dimensions as integers
        dimensions = [5, 10, 15, 20, 30, 40, 45]
        dim = np.random.choice(dimensions)
        length = dim
        width = dim
        height = np.random.randint(1, 5) 
        weight = np.random.randint(1, 4)
        compressibility_index = np.random.randint(10000, 40000)

        # create the parcel from the sampled paramters
        sampled_parcel = Parcel(length, width, height, weight, compressibility_index)
        parcel_dataset.append(sampled_parcel)
    
    return parcel_dataset
