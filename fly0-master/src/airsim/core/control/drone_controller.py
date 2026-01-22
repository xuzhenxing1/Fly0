import airsim
import numpy as np
from typing import List, Tuple, Dict, Optional

class DroneController:
    def __init__(self, drone_name: str = "Drone1"):
        self.drone_name = drone_name
        self.client = airsim.MultirotorClient()
        self._initialize()
    
    def _initialize(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
    
    def takeoff(self):
        self.client.takeoffAsync().join()
    
    def land(self):
        self.client.landAsync().join()
    
    def fly_to(self, point: List[float], velocity: float = 5.0):
        z = -point[2] if point[2] > 0 else point[2]
        self.client.moveToPositionAsync(point[0], point[1], z, velocity).join()
    
    def fly_path(self, points: List[List[float]], velocity: float = 5.0):
        if len(points) == 0:
            return
        
        airsim_points = []
        for point in points:
            z = -point[2] if point[2] > 0 else point[2]
            airsim_points.append(airsim.Vector3r(point[0], point[1], z))
        
        self.client.moveOnPathAsync(
            airsim_points, velocity, 120,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0), 20, 1
        ).join()
    
    def set_yaw(self, yaw: float, timeout: float = 5.0):
        self.client.rotateToYawAsync(yaw, timeout).join()
    
    def get_position(self) -> List[float]:
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]
    
    def get_yaw(self) -> float:
        orientation = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(orientation)[2]
    
    def get_velocity(self) -> np.ndarray:
        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        kinematics = state.kinematics_estimated
        return np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])
    
    def get_object_position(self, object_name: str) -> Optional[List[float]]:
        query_string = object_name + ".*"
        object_names = self.client.simListSceneObjects(query_string)
        
        if not object_names:
            return None
        
        pose = self.client.simGetObjectPose(object_names[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]
    
    def take_control(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
    
    def release_control(self):
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)
    
    def disconnect(self):
        self.release_control()
