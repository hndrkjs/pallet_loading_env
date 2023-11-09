import pybullet as p
import pybullet_data
import numpy as np

class PhysicsStabilityEngine():

    def __init__(self, length, width, gravity=-9.8):
        """
        Initializes the physics engine.
        
        Args:
            length (int): The length of the pallet.
            width (int): The width of the pallet.
        """
        self.client = p.connect(p.DIRECT)

        self.length = length/100.
        self.width = width/100.
        self.height = 0.15
        self.gravity = gravity

        self.mid_length = self.length/2
        self.mid_width = self.width/2

        self.parcel_ids = []
        
        self.pallet = self.create_pallet(position=(0,0,0), orientation=(0,0,0,1),lateral_friction=0.1)

        
    def transform_flb_to_center(self, flb_pos, length, width):
        """
        Transforms the front left bottom corner of the parcel to the center of the parcel.

        Args:
            flb (tuple): The front left bottom corner of the parcel.
        
        Returns:
            center (tuple): The center of the parcel.
        """

        x, y, z = flb_pos
        center_pos = (x + length/2, y + width/2)

        final_pos = (center_pos[0]-self.mid_length, center_pos[1]-self.mid_width, z)

        return final_pos

    def create_pallet(self, position=(0,0,0), orientation=(0,0,0,1),lateral_friction=0.1):
        """
        Creates a pallet.

        Args:
            length (int): The length of the pallet.
            width (int): The width of the pallet.
        
        Returns:
            pallet (int): The pallet ID.
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pallet_id = p.loadURDF("plane100.urdf")


        # Create the europalette as a rigid body (box)
        palette_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.length/2, self.width/2, self.height/2])
        palette_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=palette_id)

        # Set the position and orientation of the europalette (on the ground)
        start_pos = [0, 0, self.height/2]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Roll, pitch, yaw angles
        p.resetBasePositionAndOrientation(palette_body_id, start_pos, start_orientation)

        p.changeDynamics(palette_body_id, -1, lateralFriction=lateral_friction, physicsClientId=self.client)

        return palette_body_id

    def create_parcel(self, length, width, height, weight, position=(0,0,0), friction=0.1):
        """
        Creates a parcel.

        Args:
            length (int): The length of the parcel.
            width (int): The width of the parcel.
            height (int): The height of the parcel.
            weight (int): The weight of the parcel.
            position (tuple): The position of the parcel in (x,y,z).
        
        Returns:
            parcel (int): The parcel ID.
        """

        # modify position to place parcel on top of pallet and not inside the pallet
        position = (position[0], position[1], position[2] + self.height + height/2)

        position = self.transform_flb_to_center(position,length, width)

        parcel_visual_shape_id = p.createVisualShape( 
            shapeType=p.GEOM_BOX,
            halfExtents=(length/2, width/2, height/2),
            #rgbaColor=(1,0,0,1),
            physicsClientId=self.client
        )

        parcel_collision_shape_id = p.createCollisionShape( 
            shapeType=p.GEOM_BOX,
            halfExtents=(length/2, width/2, height/2),
            physicsClientId=self.client
        )

        parcel_id = p.createMultiBody(
            baseMass=weight,
            baseCollisionShapeIndex=parcel_collision_shape_id,
            baseVisualShapeIndex=parcel_visual_shape_id,
            basePosition=position,
            physicsClientId=self.client
        )

        #p.resetBaseVelocity(parcel_id,linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.client)
        p.changeDynamics(parcel_id, -1, lateralFriction=friction, physicsClientId=self.client)

        return parcel_id
    
    def place_parcel_on_pallet(self, parcel, location):
        """
        Places a parcel on the pallet.

        Args:
            parcel_id (Parcel): The parcel ID.
            location (tuple): The location on the pallet where the parcel should be placed.
        """
        
        x, y, z = location[0]/100., location[1]/100., location[2]/100.
        length, width, height = parcel.length/100., parcel.width/100., parcel.height/100.
        weight = parcel.weight

        # create parcel
        parcel_id = self.create_parcel(length, width, height, weight, position=(x,y,z))

        # add parcel to list of parcels
        self.parcel_ids.append(parcel_id)

    def check_stability(self, original_parcel_pos, final_parcel_pos, tol=0.1):
        """
        Checks whether the pallet is stable.

        Args:
            original_parcel_pos (tuple): The original position of the parcel.
            final_parcel_pos (tuple): The final position of the parcel.
            tol (float): The tolerance for the stability check.
        
        Returns:
            bool (list): Whether the pallet is stable.
        """
        assert len(original_parcel_pos) == len(final_parcel_pos), "The number of parcels in the original and final position must be the same."

        np_original_parcel_pos = np.array(original_parcel_pos)
        np_final_parcel_pos = np.array(final_parcel_pos)

        stability_score = []

        # for each parcel, determine whether the parcel has moved inside the tolerance
        for i in range(len(np_original_parcel_pos)):
            if np.linalg.norm(np_original_parcel_pos[i] - np_final_parcel_pos[i]) < tol:
                stability_score.append(True)
            else:
                stability_score.append(False)

        return stability_score

    def run_stability_simulation(self, num_steps=200, tol=0.1):
        """
        Runs a stability simulation.

        Args:
            num_steps (int): The number of steps to run the simulation for.
            time_step (float): The time step of the simulation.
            tol (float): The tolerance for the stability check.

        Returns:
            stability_score (float): The stability score of the pallet.
        """
        # the final stability score of the pallet
        stability_score = 0.0
        # list of indicators for every parcel whether it has moved outside the tolerance or not
        parcel_stabiltiy_score = []

        orginal_parcel_pos = []
        original_parcel_orientation = []

        final_parcel_pos = []
        final_parcel_orientation = []

        p.setGravity(0,0,self.gravity, physicsClientId=self.client)
        timeStep = 1./400
        p.setTimeStep(timeStep)

        # get parcel original positions and orientations
        for parcel_id in self.parcel_ids:
            parcel_pos, parcel_or = p.getBasePositionAndOrientation(parcel_id, physicsClientId=self.client)
            orginal_parcel_pos.append(parcel_pos)
            original_parcel_orientation.append(parcel_or)

        # running the simulation
        for i in range(num_steps):
            # step the simulation
            #p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
            p.stepSimulation()
            #time.sleep(.01)
        
        for parcel_id in self.parcel_ids:
            parcel_pos, parcel_or = p.getBasePositionAndOrientation(parcel_id, physicsClientId=self.client)
            final_parcel_pos.append(parcel_pos)
            final_parcel_orientation.append(parcel_or)
        
        parcel_stabilitiy_score = self.check_stability(orginal_parcel_pos, final_parcel_pos, tol=tol)

        # calculate stability score
        stability_score = np.sum(parcel_stabilitiy_score) / (len(self.parcel_ids) + 0.01)

        for idx, parcel_id in enumerate(self.parcel_ids):
            p.resetBasePositionAndOrientation(parcel_id, orginal_parcel_pos[idx], original_parcel_orientation[idx], physicsClientId=self.client)

        return stability_score

    def reset(self):
        """
        Resets the physics engine.
        """
        p.disconnect(self.client)
        self.client = p.connect(p.DIRECT)
        self.pallet = self.create_pallet(position=(0,0,0), orientation=(0,0,0,1),lateral_friction=0.1)
        self.parcel_ids = []

