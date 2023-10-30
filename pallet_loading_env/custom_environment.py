import numpy as np
import copy
import gym
#from torch.utils.data import DataLoader
from environment.pallet import Pallet
from environment.parcel_creator import ParcelCreator


PALLET_SIZE = (120,80) 
MAX_PALLET_HEIGHT = 210 
MAX_PALLET_WEIGHT = 1500 
NUM_ROTATIONS = 2
ASSESS_STABILITY = True 
TRAIN = True

class PalletLoadingEnv(gym.Env):
    """
    An Environment for the pallet loading problem.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self.pallet_size = PALLET_SIZE
        self.max_pallet_height = MAX_PALLET_HEIGHT
        self.max_pallet_weight = MAX_PALLET_WEIGHT
        self.num_rotations = NUM_ROTATIONS
        self.assess_stability = ASSESS_STABILITY
        self.placed_parcels = 0

        self.pallet = Pallet(self.pallet_size[0], self.pallet_size[1], self.max_pallet_height, self.max_pallet_weight, self.assess_stability)

        # action and observation specific varaibles
        self.action_area = self.pallet_size[0] * self.pallet_size[1]
        self.observation_area = self.action_area 

        #if train and self.parcel_set is None:
        #    assert self.parcel_creator is not None, "Either parcel_set or parcel_creator must be provided."

        if TRAIN:
            self.parcel_creator = ParcelCreator(self.pallet_size[0], self.pallet_size[1], self.max_pallet_height, self.max_pallet_weight)
        
        #self.dataloader = DataLoader(self.parcel_creator, batch_size=1, shuffle=True, num_workers=0)

        self.num_parcels, self._parcels = self.parcel_creator.generate_new_instance()

        self.parcels_in_order = self.get_initial_parcel_representation()

        # action space is a tuple of three discrete spaces. Corresponding to choosing x and y coordinates and rotation
        self.action_space = {
            'x' : gym.spaces.Discrete(self.pallet_size[0]),
            'y' : gym.spaces.Discrete(self.pallet_size[1]),
            'rotation' : gym.spaces.Discrete(2)
        }

        # observation space is a dict of five continuous spaces. Corresponding to the heightmap, weightmap and parcel representation
        observation_space = {
            'heightmap' : gym.spaces.Box(low=0, high=self.max_pallet_height, shape=(self.observation_area,)),
            'weightmap': gym.spaces.Box(low=0, high=self.max_pallet_height, shape=(self.observation_area,))
        }

        # assigning the observation space
        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = gym.spaces.Dict(self.action_space)

    def step(self, action):
        """
        Performs a step in the environment, given an action.

        Args:
            action (np.array): The action to perform. Represented by a numpy array, containing in the decomposed action in the following order parcel_index, parcel_rotation, x, y.
        """
        # access the relevant information from the action
        action_variables = self._get_action_representation(action)

        location = (action_variables['x'], action_variables['y'])

        parcel_not_placed = self.parcels_in_order[action_variables['parcel_index']][4] == 0

        if parcel_not_placed:
            # check if the parcel can be placed on the pallet. If True, the function automatically places the parcel on the pallet
            success = self.pallet.is_valid_placement(action_variables['parcel'], location, action_variables['rotation'])
        else:   
            success = False
        
        stability = 1

        if success and self.assess_stability:
            # get the stability score of the pallet
            stability = self.pallet.assess_stability_physics()
        
        if success:
            self.placed_parcels += 1
            self._update_parcel_representation(action_variables['parcel_index'])
        
        done = self._get_done(success, stability)
        reward = self._get_reward(success, stability)
        info = self._get_info(success, stability)
            
            
        if done and self.placed_parcels < self.num_parcels:
            truncated = True
        else:
            truncated = False

        return self.current_observation, reward, done, truncated, info
    
    def reset(self):
        """
        Resets the environment.
        """

        self.parcel_creator.reset()
        self.pallet.reset()
        self.placed_parcels = 0

        self.num_parcels, self._parcels = self.parcel_creator.generate_new_instance()
        self.parcels_in_order = self.get_initial_parcel_representation()

        info = {
            'parcels_in_order': self.parcels_in_order
        }

        return self.current_observation, info
    
    def close(self):
        """
        Closes the environment.
        """
        pass

    def get_initial_parcel_representation(self):
        """
        Returns a representation of the current parcel.
        The parcels are represented by a list containing the length, width and height of the parcel as well as the orientation and whether the parcel is placed.
        """
        parcel_representation = np.array([[0,0,0,0,0,0] for _ in range(self.num_parcels)])

        for idx, parcel in enumerate(self._parcels):
            _parcel_repr = parcel_representation[idx]
            _parcel_repr[0] = parcel.length
            _parcel_repr[1] = parcel.width
            _parcel_repr[2] = parcel.height
            _parcel_repr[3] = parcel.weight

        return parcel_representation
    
    def possible_actions(self):
        """
        Returns a binary mask for the pallet, indicating whether an index on th pallet corresponds to a possible action.
        """
        x = self.current_parcel.length
        y = self.current_parcel.width
        z = self.current_parcel.height

        pallet_width = self.pallet_size[0]
        pallet_length = self.pallet_size[1]

        possible_actions = np.zeros((pallet_width, pallet_length), dtype=np.int32)

        for i in range(pallet_width - x + 1):
            for j in range(pallet_length - y + 1):
                if self.pallet.is_valid_placement(i, j, x, y, z):
                    possible_actions[i, j] = 1

        # if no possible action was obtained, set all actions to possible
        if np.sum(possible_actions) == 0:
            possible_actions = np.ones((pallet_width, pallet_length), dtype=np.int32)
        
        return possible_actions

    @property
    def current_observation(self):
        """
        Returns the current observation of the environment.
        The current observation of the environment contains the heightmap, weightmap and parcel representation.
        """
        heightmap = self.pallet.heightmap
        weightmap = self.pallet.weightmap

        observation = { 
            'heightmap' : heightmap,
            'weightmap' : weightmap
        }   

        return observation
        
        #return np.reshape(np.hstack((heightmap, weightmap)), newshape=(-1,))
    
    def _get_action_representation(self, action):
        """
        Returns the action representation of the action.
        
        Args:
            action (np.array): The action to perform. Represented by a numpy array, containing in the decomposed action in the following order parcel_index, parcel_rotation, x, y.
        """
        index, rotation, x_coord, y_coord  = action[0], action[1], action[2], action[3]


        parcel = self._parcels[index]


        action_representation = {
            'x' : x_coord,
            'y' : y_coord,
            'parcel' : parcel,
            'rotation' : rotation,
            'parcel_index' : index   
        }

        return action_representation
    
    def _get_done(self, success, stability):
        """
        Returns whether the episode is done.

        Args:
            success (bool): Whether the parcel was successfully placed on the pallet.
            stability (float): The stability score of the pallet.
        """
        if success:
            # if the pallet isn't stable, the episode is done
            if stability < 0.5:
                return True
            # if all parcels are placed, the episode is done
            if self.placed_parcels == self.num_parcels:
                return True
            # else the episode is not done
            return False
        else:
            return True
        
    def _get_reward(self, success, stability):
        """
        Returns the reward for the current step.

        Args:
            success (bool): Whether the parcel was successfully placed on the pallet.
            stability (float): The stability score of the pallet.
        """
        if success:
            return 1
        else:
            return 0
        
    def _get_info(self, success, stability):
        """
        Returns the info for the current step.

        Args:
            success (bool): Whether the parcel was successfully placed on the pallet.
            stability (float): The stability score of the pallet.
        """
        
        info_dict = {
            'success' : success,
            'stability' : stability,
            'parcels_in_order': self.parcels_in_order
        }
        
        return info_dict
    
    def _update_parcel_representation(self, parcel_index):
        """
        Updates the parcel representation.

        Args:
            parcel_index (int): The index of the parcel to be placed.
        """

        self.parcels_in_order[parcel_index][4] = 1
    