DHL_MAX_DIMS_PARCEL_TYPES = [
        {"type": "extra_small", "length_range": (15, 35), "width_range": (11, 25), "height_range": (1, 3), "weight_range": (0.5, 1), "compressibility_index_range": (1, 4), "probability": 0.6},
        {"type": "small", "length_range": (35, 60), "width_range": (15, 30), "height_range": (1, 15), "weight_range": (0.5, 2), "compressibility_index_range": (1, 4), "probability": 0.3},
        {"type": "medium", "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (0.5, 5), "compressibility_index_range": (1, 4), "probability": 0.1},
        {"type": "large", "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (5, 10), "compressibility_index_range": (1, 4), "probability": 0.05},
        {"type": "extra_large", "length_range": (60, 120), "width_range": (30, 60), "height_range": (1, 60), "weight_range": (10, 31.5), "compressibility_index_range": (1, 4), "probability": 0.05}
    ]

DHL_PARCEL_TYPES = {
     "extra_small": {"length": 22.5, "width": 14.5, "height": 3, "weight_range": (0.5, 1), "compressibility_index_range": (1, 4), "probability": 0.15},
     "small": {"length": 25, "width": 17.5, "height": 10, "weight_range": (0.5, 2), "compressibility_index_range": (1, 4), "probability": 0.35},
     "medium": { "length": 37.5, "width": 30, "height": 13.5, "weight_range": (0.5, 5), "compressibility_index_range": (1, 4), "probability": 0.35},
     "large": { "length": 45, "width": 35, "height": 20, "weight_range": (5, 10), "compressibility_index_range": (1, 4), "probability": 0.1},
     "bottle": { "length": 38, "width": 12, "height": 12, "weight_range": (1, 3), "compressibility_index_range": (1, 4), "probability": 0.05}
}
        

def get_parcel_type(parcel_type:str):
    """
    Returns the parcel type.

    Args:
        parcel_type (str): The parcel type.
    
    Returns:
        parcel_type (dict): The parcel type.
    """
    if parcel_type == "dhl_max_dims":
        return DHL_MAX_DIMS_PARCEL_TYPES
    elif parcel_type == "dhl":
        return DHL_PARCEL_TYPES
    else:
        raise ValueError("Parcel type not supported.")