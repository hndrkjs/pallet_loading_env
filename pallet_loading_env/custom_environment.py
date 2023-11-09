import numpy as np
import torch
import gym
from pallet_loading_env.pallet import Pallet
from pallet_loading_env.parcel_creator import ParcelCreator


PALLET_SIZE = (120,80) 
MAX_PALLET_HEIGHT = 210 
PALLET_VOLUME = PALLET_SIZE[0] * PALLET_SIZE[1] * MAX_PALLET_HEIGHT
MAX_PALLET_WEIGHT = 1500 
NUM_ROTATIONS = 2
ASSESS_STABILITY = True
TRAIN = True
MAX_PARCELS = 100

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
        self.placed_parcel_volume = 0
        self.volume_utilisation = 0
        self.max_parcels = MAX_PARCELS
        self.centre_of_mass_x = 0.5 * PALLET_SIZE[0]
        self.centre_of_mass_y = 0.5 * PALLET_SIZE[1]
        self.centre_of_mass_z = 0
        self.total_mass = 0
        self.curr_max_height = 0.00001

        self.g_x = 1 - (np.abs(self.centre_of_mass_x - 40) / (0.5 * PALLET_SIZE[0]))
        self.g_y = 1 - (np.abs(self.centre_of_mass_y - 60) / (0.5 * PALLET_SIZE[1]))
        self.g_z = 1 - (self.centre_of_mass_z / self.curr_max_height)


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

        self.__index_mask = torch.zeros(100, dtype=torch.int32)
        self.__index_mask[0:self.num_parcels] = 1
        self.__index_mask = self.__index_mask.bool()

        # action space is a tuple of three discrete spaces. Corresponding to choosing x and y coordinates and rotation
        '''
        self.action_space = [
            gym.spaces.Discrete(100),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(self.pallet_size[0]),
            gym.spaces.Discrete(self.pallet_size[1])]
        '''
        
        self.action_space = [
            100,
            2,
            self.pallet_size[0],
            self.pallet_size[1]]

        # observation space is a dict of two continuous spaces. Corresponding to the heightmap and weightmap
        self.observation_space = {
            'heightmap' : gym.spaces.Box(low=0, high=self.max_pallet_height, shape=(self.pallet_size[0],self.pallet_size[1]), dtype=np.int32),
            'weightmap': gym.spaces.Box(low=0, high=self.max_pallet_weight, shape=(self.pallet_size[0],self.pallet_size[1]), dtype=np.float32)
        }

        # assigning the observation space
        self.observation_space = gym.spaces.Dict(self.observation_space)
        self.action_space = gym.spaces.MultiDiscrete(self.action_space)

    def step(self, action):
        """
        Performs a step in the environment, given an action.

        Args:
            action (np.array): The action to perform. Represented by a numpy array, containing in the decomposed action in the following order parcel_index, parcel_rotation, x, y.
        """
        # access the relevant information from the action
        action_variables = self._get_action_representation(action)

        location = (action_variables['x'], action_variables['y'])

        parcel_not_placed = self.__index_mask[action_variables['parcel_index']]

        if parcel_not_placed:
            # check if the parcel can be placed on the pallet. If True, the function automatically places the parcel on the pallet
            success, z = self.pallet.is_valid_placement(action_variables['parcel'], location, action_variables['rotation'])
            centre_mass_location = (location[0], location[1], z)
        else:   
            success = False
        
        stability = 1

        if success and self.assess_stability:
            # get the stability score of the pallet
            stability = self.pallet.assess_stability_physics()
        
        if success:
            self.placed_parcels += 1
            self._update_parcel_representation(action_variables['parcel_index'])
            self.update_centre_of_mass(action_variables['parcel'], centre_mass_location)
            self.placed_parcel_volume += (action_variables['parcel'].length * action_variables['parcel'].width * action_variables['parcel'].height)
            self.volume_utilisation = self.placed_parcel_volume / PALLET_VOLUME
        
        done = self._get_done(success, stability)
            
        if done and self.placed_parcels < self.num_parcels:
            truncated = True
        else:
            truncated = False
        
        reward = self._get_reward(success, stability, done, truncated)
        info = self._get_info(success, stability, done, truncated)

        return self.current_observation, reward, done, truncated, info
    
    def reset(self, *kwargs):
        """
        Resets the environment.
        """

        self.parcel_creator.reset()
        self.pallet.reset()
        self.placed_parcels = 0

        self.num_parcels, self._parcels = self.parcel_creator.generate_new_instance()
        self.parcels_in_order = self.get_initial_parcel_representation()

        self.__index_mask = torch.zeros(100, dtype=torch.int32)
        self.__index_mask[0:self.num_parcels] = 1
        self.__index_mask = self.__index_mask.bool()

        self.centre_of_mass_x = 0.5 * PALLET_SIZE[0]
        self.centre_of_mass_y = 0.5 * PALLET_SIZE[1]
        self.centre_of_mass_z = 0
        self.total_mass = 0
        self.curr_max_height = 0.00001

        self.g_x = 1 - (np.abs(self.centre_of_mass_x - 60) / (0.5 * PALLET_SIZE[0]))
        self.g_y = 1 - (np.abs(self.centre_of_mass_y - 40) / (0.5 * PALLET_SIZE[1]))
        self.g_z = 1 - (self.centre_of_mass_z / self.curr_max_height)

        self.placed_parcel_volume = 0
        self.volume_utilisation = 0

        info = {
            'parcels_in_order': self.parcels_in_order,
            'num_parcels': torch.tensor(self.num_parcels),
            'index_mask' : self.__index_mask
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
        parcel_representation = torch.tensor([[0,0,0,0] for _ in range(self.max_parcels)])

        for idx, parcel in enumerate(self._parcels):
            _parcel_repr = parcel_representation[idx]
            _parcel_repr[0] = parcel.length
            _parcel_repr[1] = parcel.width
            _parcel_repr[2] = parcel.height
            _parcel_repr[3] = parcel.weight

        return parcel_representation
    
    def possible_actions(self, idx_rot):
        """
        Returns a binary mask for the pallet, indicating whether an index on th pallet corresponds to a possible action.
        """
        index, rotation = idx_rot[0], idx_rot[1]    
        current_parcel = self._parcels[index]

        if rotation == 0:
            p_length = current_parcel.length
            p_width = current_parcel.width
        elif rotation == 1:
            p_length = current_parcel.width
            p_width = current_parcel.length    

        pallet_width = self.pallet_size[0]
        pallet_length = self.pallet_size[1]

        possible_actions_y = np.zeros((pallet_width), dtype=np.int32)
        possible_actions_x = np.zeros((pallet_length), dtype=np.int32)

        for i in range(pallet_width):
            for j in range(pallet_length):
                if i + p_width <= pallet_width and j + p_length <= pallet_length:
                    possible_actions_y[i] = 1
                    possible_actions_x[j] = 1

        # if no possible action was obtained, set all actions to possible
        if np.sum(possible_actions_y) == 0 and np.sum(possible_actions_x) == 0:
            possible_actions_y = np.ones((pallet_width), dtype=np.int32)
            possible_actions_x = np.ones((pallet_length), dtype=np.int32)

        
        return np.concatenate((possible_actions_y, possible_actions_x))

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
    
    def update_centre_of_mass(self, parcel, location):
        """
        Updates the centre of mass of the pallet.

        Args:
            parcel (Parcel): The parcel that was placed on the pallet.
            location (tuple): The location of the parcel on the pallet.
        """
        # calculate the new centre of mass
        new_mass = self.total_mass + parcel.weight
        new_x = (self.centre_of_mass_x * self.total_mass + parcel.weight * (location[0] + 0.5 * parcel.length)) / new_mass
        new_y = (self.centre_of_mass_y * self.total_mass + parcel.weight * (location[1] + 0.5 * parcel.width)) / new_mass
        new_z = (self.centre_of_mass_z * self.total_mass + parcel.weight * (location[2] + 0.5 * parcel.height)) / new_mass

        # update the centre of mass
        self.centre_of_mass_x = new_x
        self.centre_of_mass_y = new_y
        self.centre_of_mass_z = new_z
        self.total_mass = new_mass

        self.curr_max_height = self.pallet.highest_point

        # update the gravity score
        self.g_x = 1 - (np.abs(self.centre_of_mass_x - 60) / (0.5 * PALLET_SIZE[0]))
        self.g_y = 1 - (np.abs(self.centre_of_mass_y - 40) / (0.5 * PALLET_SIZE[1]))
        self.g_z = 1 - (self.centre_of_mass_z / self.curr_max_height)
    
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
        
    def _get_reward(self, success, stability, done, truncated):
        """
        Returns the reward for the current step.

        Args:
            success (bool): Whether the parcel was successfully placed on the pallet.
            stability (float): The stability score of the pallet.
        """
        critic_reward = 0

        if success and not done:
            critic_reward = 10 * self.volume_utilisation + 5 * stability
        if success and done and not truncated:
            critic_reward = self.placed_parcels/self.num_parcels + self.volume_utilisation + stability + (self.g_x + self.g_y + self.g_z)/3.0
        if done and truncated:
            critic_reward = -(1 - self.placed_parcels/self.num_parcels) - (1 - self.volume_utilisation) - (1 - stability) - (1 - (self.g_x + self.g_y + self.g_z)/3.0)

        return critic_reward
        
    def _get_info(self, success, stability, done, truncated):
        """
        Returns the info for the current step.

        Args:
            success (bool): Whether the parcel was successfully placed on the pallet.
            stability (float): The stability score of the pallet.
        """
        sequence_head_reward = 0
        rotation_head_reward = 0
        position_x_reward = 0
        position_y_reward = 0

        if success and not done:
            sequence_head_reward = self.placed_parcels/self.num_parcels + self.volume_utilisation
            rotation_head_reward = stability
            position_x_reward = self.volume_utilisation + stability
            position_y_reward = self.volume_utilisation + stability
        
        if success and done and not truncated:
            sequence_head_reward = self.placed_parcels/self.num_parcels + self.volume_utilisation
            rotation_head_reward = 1
            position_x_reward = 1 * self.volume_utilisation + stability
            position_y_reward = 1 * self.volume_utilisation + stability
        
        if done and truncated:
            sequence_head_reward = -(1 - self.placed_parcels/self.num_parcels) - (1 - self.volume_utilisation)
            rotation_head_reward = -(1-stability)
            position_x_reward = -(1 - self.volume_utilisation) - (1 - stability)
            position_y_reward = -(1 - self.volume_utilisation) - (1 - stability)

        
        info_dict = {
            'success' : success,
            'stability' : stability,
            'parcels_in_order': self.parcels_in_order,
            'num_parcels': torch.tensor(self.num_parcels),
            'volume_util': self.volume_utilisation,
            'index_mask' : self.__index_mask,
            'sequence_head_reward' : sequence_head_reward,
            'rotation_head_reward' : rotation_head_reward,
            'position_x_head_reward' : position_x_reward,
            'position_y_head_reward' : position_y_reward
        }
        
        return info_dict
    
    def _update_parcel_representation(self, parcel_index):
        """
        Updates the parcel representation.

        Args:
            parcel_index (int): The index of the parcel to be placed.
        """

        self.__index_mask[parcel_index] = False
    