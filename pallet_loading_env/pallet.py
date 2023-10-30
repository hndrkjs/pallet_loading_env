import numpy as np
from copy import deepcopy
from environment.physics_engine import PhysicsStabilityEngine

class Pallet():
    """
    This class is used to represent a pallet in the environment.
    It stores information about the pallet, such as its size, the parcels on it, and the weightmap and heightmap. 
    It can also infer whether a parcel can be placed on the pallet.
    """
    
    def __init__(self, length, width, max_height, max_weight, assess_stability=False):
        """
        Initializes the pallet.

        Args:
            length (int): The length of the pallet.
            width (int): The width of the pallet.
            max_height (int): The maximum height of the pallet.
            max_weight (int): The maximum weight of the pallet.
        """
        self.length = length
        self.width = width
        self.max_height = max_height
        self.max_weight = max_weight
        self.THRESHOLD = 0.7
        self.assess_stability = assess_stability

        if self.assess_stability:
            self.physics_engine = PhysicsStabilityEngine(self.length, self.width)
        

        # initialize pallet specific variables
        self.__heightmap = np.zeros((self.length, self.width), dtype=np.int32)
        self.__weightmap = np.zeros((self.length, self.width), dtype=np.float32)

        # initialize compresisbility map
        self.compressibility_index = float(self.max_weight / (self.length * self.width))
        self.__compression_index_map = np.ones((self.length, self.width), dtype=np.float32) * self.compressibility_index

        # rotations for parcels
        self.rotations = [[0,1], [1,0]]

        self.placed_parcels = []

    @property
    def heightmap(self):
        """
        Returns the heightmap of the pallet.
        """
        return self.__heightmap
    
    @property
    def weightmap(self):
        """
        Returns the weightmap of the pallet.
        """
        return self.__weightmap
    
    @property
    def compression_index_map(self):
        """
        Returns the compression index map of the pallet.
        """
        return self.__compression_index_map
    
    def rotate_parcel(self, parcel, orientation):
        """
        Rotates a parcel according to the chosen orientation.

        Args:
            parcel (Parcel): The parcel to be rotated.
            orientation (int): The orientation of the parcel.
        """
        # rotate parcel according to the chosen orientation
        parcel.length = parcel.size[self.rotations[orientation][0]]
        parcel.width = parcel.size[self.rotations[orientation][1]]


    def is_valid_placement(self, parcel, location, orientation):
        """
        Checks whether a parcel can be placed on the pallet.

        Args:
            parcel (Parcel): The parcel to be placed on the pallet.
            location (tuple): The location on the pallet where the parcel should be placed.
            orientation (int): The orientation of the parcel.

        Returns:
            bool: Whether the parcel can be placed on the pallet.
        """
        x,y = location
        # rotate parcel according to the chosen orientation
        proxy_parcel = deepcopy(parcel)
        self.rotate_parcel(proxy_parcel, orientation)

        ########################################
        #      Basic constraints to check:     #
        ########################################

        # check if the parcel is within the bounds of the pallet
        if x + proxy_parcel.length > self.length or y + proxy_parcel.width > self.width:
            print(1)
            return False
        
        # check if the parcel exceeds the maximum height of the pallet
        if proxy_parcel.height + self.__heightmap[x:x+proxy_parcel.length, y:y+proxy_parcel.width].max() > self.max_height:
            print(2)
            return False
        
        # check if the parcel exceeds the maximum weight of the pallet
        if proxy_parcel.weight + np.sum(self.__weightmap) > self.max_weight:
            print(3)
            return False, self.__weightmap
        
        ########################################
        #      Constraints for stability       #
        ########################################

        # general information about parcel
        parcel_base_area = parcel.base_area
        placement_height_area = self.__heightmap[x:x+proxy_parcel.length, y:y+proxy_parcel.width]
        placement_compress_area = self.__compression_index_map[x:x+proxy_parcel.length, y:y+proxy_parcel.width]

        # -------------------------------------#
        # SUPPORT CONSTRAINT FOR PARCEL 
        # sufficient support for bottom area of parcel. determined by calculating the ratio of overlapping surface area of the parcel and the parcels below it
        
        # get overlapping area
        height_values, height_counts = np.unique(placement_height_area, return_counts=True)
        highest_height_idx = np.argmax(height_values)
        overlapping_area = height_counts[highest_height_idx]

        # calculate ratio of overlapping area to parcel base area
        ratio = overlapping_area / parcel_base_area

        if ratio < self.THRESHOLD:
            print(4)
            return False
        
        # -------------------------------------#
        # COMPRESSIBILITY CONSTRAINT FOR PARCEL
        # sufficient support for weight of parcel. determined by the compressibility index of the parcel/s immediately below the placed parcel

        min_compress_index = np.min(placement_compress_area)
        parcel_pressure = proxy_parcel.weight / parcel_base_area
        if min_compress_index < parcel_pressure:
            print(5)
            return False

        self.place_parcel(parcel, location, orientation)    

        return True

    def place_parcel(self, parcel, location, orientation):
        """
        Places a parcel on the pallet.

        Args:
            parcel (Parcel): The parcel to be placed on the pallet.
            location (tuple): The location on the pallet where the parcel should be placed.
            orientation (int): The orientation of the parcel.
        """
        x,y = location
        # rotate parcel according to the chosen orientation
        self.rotate_parcel(parcel, orientation)

        # add parcel to list of placed parcels
        z = self.__heightmap[x:x+parcel.length, y:y+parcel.width].max()
        self.placed_parcels.append((parcel, (x,y,z)))

        # place parcel on pallet
        self.__heightmap[x:x+parcel.length, y:y+parcel.width] += parcel.height
        # very simpliefied assumption that the weight is evenly distributed over the area of the parcel
        self.__weightmap[x:x+parcel.length, y:y+parcel.width] += parcel.weight / parcel.base_area
        # update compressibility index map
        self.__compression_index_map[x:x+parcel.length, y:y+parcel.width] = self.__compression_index_map[x:x+parcel.length, y:y+parcel.width] - parcel.weight / parcel.base_area

        if self.assess_stability:
            self.physics_engine.place_parcel_on_pallet(parcel, (x,y,z))
    
    def assess_stability_physics(self):
        """
        Assess the physical stability of the pallet using the PyBullet physics engine.

        Returns:
            stability_score (float): The stability score of the pallet.
        """
        stability_score = self.physics_engine.run_stability_simulation()

        return stability_score

    def reset(self):
        """
        Resets the pallet.
        """
        self.__heightmap = np.zeros((self.length, self.width), dtype=np.int32)
        self.__weightmap = np.zeros((self.length, self.width), dtype=np.float32)
        self.__compression_index_map = np.ones((self.length, self.width), dtype=np.float32) * self.compressibility_index
        self.placed_parcels = []

        if self.assess_stability:
            self.physics_engine.reset()
