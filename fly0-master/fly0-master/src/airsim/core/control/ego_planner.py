import airsim
import numpy as np
import math
import time
from scipy.spatial import KDTree
import threading
from queue import Queue
from ..detection.visual_target_detector import VisualTargetDetector


draw=True
class EgoPlanner:
    def __init__(self, drone_name="Drone1", lidar_sensors=["LidarSensor1"], visual_detector=None):
        # 初始化AirSim客户端
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        
        self.drone_name = drone_name
        self.lidar_sensors = lidar_sensors
        self.visual_detector = visual_detector
        self.max_velocity = 2.0  # 最大速度 m/s
        self.max_acceleration = 2.0  # 最大加速度 m/s²
        self.horizon_distance = 10.0  # 规划视距
        self.drone_radius = 1.0  # 无人机半径（用于碰撞检测）
        self.goal_dis = 0.0
        self.lidar_range = 30.0  # 雷达探测范围
        self.safety_distance = 1.0  # 安全距离
        self.collision_check_resolution = 0.5  # 碰撞检测分辨率
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.goal_position = np.zeros(3)
        self.is_running = False
        self.obstacles = []
        self.obstacle_kdtree = None
        
        # B样条曲线参数
        self.control_points = []  # 控制点
        self.bspline_degree = 3   # B样条曲线阶数
        
        # 历史航线点
        self.trajectory_history = []  # 存储历史轨迹点
        
        # 路径规划线程
        self.planning_thread = None
        self.command_queue = Queue()
    
    def take_control(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
    
    
    def get_drone_pose(self):
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        kinematics = drone_state.kinematics_estimated
        position = kinematics.position
        orientation = kinematics.orientation
        linear_velocity = kinematics.linear_velocity
        angular_velocity = kinematics.angular_velocity
        return {
            'position': np.array([position.x_val, position.y_val, position.z_val]),
            'orientation': np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val]),
            'linear_velocity': np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val]),
            'angular_velocity': np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])
        }
    
    def get_drone_state(self):
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        return {
            'gps_location': drone_state.gps_location,
            'timestamp': drone_state.timestamp,
            'landed_state': drone_state.landed_state,
            'rc_data': drone_state.rc_data,
            'ready': drone_state.ready,
            'ready_message': drone_state.ready_message
        }
    
    def get_lidar_data(self, sensor_name):
        try:
            lidar_data = self.client.getLidarData(sensor_name, self.drone_name)

            if len(lidar_data.point_cloud) < 3:
                return None
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            return {
                'points': points,
                'time_stamp': lidar_data.time_stamp,
                'pose': lidar_data.pose
            }
        except Exception as e:
            return None
    
    def euler_to_rotation_matrix(self, euler_angles):
        roll, pitch, yaw = euler_angles
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        return np.dot(Rz, np.dot(Ry, Rx))

    def get_all_lidar_data(self):
        all_points = []
        drone_pose = self.get_drone_pose()
        drone_position = drone_pose['position']
        drone_orientation = drone_pose['orientation']
        drone_rotation_matrix = self.quaternion_to_rotation_matrix(drone_orientation)
        for sensor in self.lidar_sensors:
            data = self.get_lidar_data(sensor)
            if data and len(data['points']) > 0:
                lidar_position = np.array([0, 0, -1.0])
                lidar_rotation = np.array([0, 0, 0]) 
                lidar_rotation_matrix = self.euler_to_rotation_matrix(lidar_rotation)
                body_points = np.dot(data['points'], lidar_rotation_matrix.T) + lidar_position
                world_points = np.dot(body_points, drone_rotation_matrix.T) + drone_position
                all_points.append(world_points)
        
        if all_points:
            ppp = np.vstack(all_points) 
            if draw:
                draw_point = []
                try:
                    for points in ppp:
                        draw_point.append(airsim.Vector3r(points[0], points[1], points[2]))
                    self.client.simPlotPoints(
                            draw_point, 
                            color_rgba=[0.0, 0.0, 1.0, 1.0], # 雷达点蓝色
                            size=5, 
                            duration=0.1, 
                            is_persistent=False)
                except Exception as e:
                    print(f"Error drawing debug points: {e}")
            return np.vstack(all_points)
        return np.array([]).reshape(0, 3)
    
    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def update_obstacles(self):
        lidar_points = self.get_all_lidar_data()
        if len(lidar_points) > 0:
            self.obstacles = lidar_points
            self.obstacle_kdtree = KDTree(self.obstacles)
    
    def check_collision(self, point):
        if self.obstacle_kdtree is None or len(self.obstacles) == 0:
            return False
        distance, _ = self.obstacle_kdtree.query(point)
        return distance < self.safety_distance
    
    def check_segment_collision(self, start, end):
        if self.obstacle_kdtree is None or len(self.obstacles) == 0:
            return False
        
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return False
        
        direction = direction / length
        num_samples = max(2, int(length / self.collision_check_resolution))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start + t * direction * length
            if self.check_collision(point):
                return True
        
        return False
    
    def find_consecutive_colliding_segments(self, control_points):
        colliding_segments = []
        
        for i in range(len(control_points) - 1):
            if self.check_segment_collision(control_points[i], control_points[i+1]):
                start_idx = i
                end_idx = i + 1
                

                while start_idx > 0 and self.check_segment_collision(control_points[start_idx-1], control_points[start_idx]):
                    start_idx -= 1
                
                while end_idx < len(control_points) - 1 and self.check_segment_collision(control_points[end_idx], control_points[end_idx+1]):
                    end_idx += 1
                
                segment = (start_idx, end_idx)
                if not colliding_segments or segment != colliding_segments[-1]:
                    colliding_segments.append(segment)
        
        return colliding_segments
    
    def path_search(self, segment_indices):
        """在环境中搜索路径（简化版A*算法）"""
        start_idx, end_idx = segment_indices
        start_point = self.control_points[start_idx]
        end_point = self.control_points[end_idx]
        
        # 简化的路径搜索，实际应用中应该使用更复杂的算法如A*
        # 这里我们简单地生成一条绕过障碍物的路径
        
        # 计算中间点
        mid_point = (start_point + end_point) / 2.0
        
        # 计算障碍物的平均位置
        if len(self.obstacles) > 0:
            obstacle_center = np.mean(self.obstacles, axis=0)
            
            # 计算避开障碍物的方向(斥力方向)
            avoid_direction = mid_point - obstacle_center
            avoid_direction = avoid_direction / np.linalg.norm(avoid_direction)
            
            # 计算避开距离
            avoid_distance = 5.0  # 固定避开距离
            
            # 生成避开点
            avoid_point = start_point + avoid_direction * avoid_distance
            if draw:
                draw_point = [airsim.Vector3r(avoid_point[0], avoid_point[1], avoid_point[2])]
                self.client.simPlotPoints(
                    draw_point, 
                    color_rgba=[0.0, 0.0, 0.0 , 1.0], # avoidpoint黑色
                    size=20, 
                    duration=0.1, 
                    is_persistent=False
                )

            return [start_point, avoid_point, end_point]
        
        return [start_point, end_point]
    
    def find_pv_pairs(self, control_point, path):
        """寻找p-v对（控制点和排斥方向向量）control_point：p of {p, v}, path：[start_point, avoid_point, end_point]"""
        pv_pairs = []
        
        # 找到路径上离控制点最近的点
        min_dist = float('inf')
        closest_point = None
        
        # path：[start_point, avoid_point, end_point]
        for i in range(len(path) - 1):
            segment_start = path[i]
            segment_end = path[i+1]
            
            # 计算控制点到线段的最近点
            segment_vec = segment_end - segment_start
            segment_len = np.linalg.norm(segment_vec)
            
            if segment_len == 0:
                continue
                
            segment_dir = segment_vec / segment_len
            t = np.dot(control_point - segment_start, segment_dir) / segment_len
            t = max(0, min(1, t))  # 限制在0到1之间
            
            closest_on_segment = segment_start + t * segment_vec
            dist = np.linalg.norm(control_point - closest_on_segment)
            
            if dist < min_dist and dist > 0:
                min_dist = dist
                closest_point = closest_on_segment
        
        if closest_point is not None and min_dist < self.safety_distance * 2:
            # 计算排斥方向（从控制点指向路径上的最近点）
            repulsive_dir = control_point - closest_point
            repulsive_dir_norm = np.linalg.norm(repulsive_dir)
            
            if repulsive_dir_norm > 0:
                repulsive_dir = repulsive_dir / repulsive_dir_norm
                pv_pairs.append((closest_point, repulsive_dir))

            if draw:
                # ② 绘制排斥方向箭头
                try:
                    direction_vector =  closest_point - control_point 
                    arrow_end = control_point + direction_vector * repulsive_dir
                    start_vec = airsim.Vector3r(control_point[0], control_point[1], control_point[2])
                    end_vec = airsim.Vector3r(arrow_end[0], arrow_end[1], arrow_end[2])
                    
                    # 绘制红色箭头，持续时间0.1秒
                    self.client.simPlotArrows(
                        [start_vec], 
                        [end_vec], 
                        color_rgba=[1.0, 0.0, 0.0, 1.0], 
                        thickness=5.0, 
                        arrow_size=10.0, 
                        duration=0.1, 
                        is_persistent=False
                    )
                except Exception as e:
                    print(f"绘制排斥方向箭头失败: {e}")
                
            
        return pv_pairs
    
    def optimize_trajectory_with_pv(self, control_points, pv_pairs_list):
        """使用p-v对优化轨迹"""
        if not pv_pairs_list or len(pv_pairs_list) != len(control_points):
            return control_points
        
        optimized_points = np.copy(control_points)
        repulsion_strength = 7.0  # 排斥强度
        for i in range(len(control_points)):
            if pv_pairs_list[i]:  # 如果有p-v对
                for p, v in pv_pairs_list[i]:
                    # 应用排斥力
                    optimized_points[i] = optimized_points[i] - v * repulsion_strength
        return optimized_points
    
    def generate_bspline_control_points(self, start_pos, goal_pos, num_points=10):
        """生成B样条曲线的控制点"""
        # 简化的控制点生成，实际应用中应该使用更复杂的算法
        control_points = [start_pos]
        if self.goal_dis <= 15.0:
            num_points = 5
        if self.goal_dis <= 10.0:
            num_points = 2
        for i in range(1, num_points - 1):
            alpha = i / (num_points - 1)
            point = start_pos * (1 - alpha) + goal_pos * alpha
            
            control_points.append(point)
        control_points.append(goal_pos)
        if draw:
            draw_points = []
            for points in control_points:
                draw_points.append(airsim.Vector3r(points[0], points[1], points[2]))
            try:
                self.client.simPlotPoints(
                    draw_points, 
                    color_rgba=[0.5, 0.5, 0.5, 1.0], # 初始B样条灰色
                    size=20, 
                    duration=0.1, 
                    is_persistent=False
                )
            except Exception as e:
                print(f"Error drawing debug points: {e}")
        
        return np.array(control_points)
    
    def bspline_interpolate(self, control_points, num_samples=100):
        """B样条曲线插值"""
        if len(control_points) < 2:
            return control_points
        
        n = len(control_points)
        k = self.bspline_degree
        
        # 生成节点向量
        knots = np.zeros(n + k)
        for i in range(k, n):
            knots[i] = (i - k + 1) / (n - k + 1)
        knots[n:] = 1.0
        
        # 采样点
        t_samples = np.linspace(0, 1, num_samples)
        trajectory = []
        
        for t in t_samples:
            # 找到t所在的区间
            span = k - 1
            while span < n - 1 and t >= knots[span + 1]:
                span += 1
            
            # 计算B样条基函数
            basis_funcs = np.zeros(k + 1)
            basis_funcs[0] = 1.0
            
            for j in range(1, k + 1):
                for i in range(j, -1, -1):
                    if i == j:
                        left = 0.0
                    else:
                        left = (t - knots[span - j + i]) / (knots[span + i + 1] - knots[span - j + i]) * basis_funcs[i]
                    
                    if i == 0:
                        right = 0.0
                    else:
                        right = (knots[span + i + 1] - t) / (knots[span + i + 1] - knots[span - j + i + 1]) * basis_funcs[i - 1]
                    
                    basis_funcs[i] = left + right
            
            # 计算曲线点
            point = np.zeros(3)
            for i in range(k + 1):
                idx = span - k + i
                if idx < 0 or idx >= n:
                    continue
                point += basis_funcs[i] * control_points[idx]
            
            trajectory.append(point)
        
        return np.array(trajectory)
    
    def safe_move_to_position(self, x, y, z, velocity, min_distance=0.5):
        """安全移动到指定位置"""
        # 获取当前位置
        pose = self.get_drone_pose()
        current_pos = pose['position']
        
        # 计算距离
        target_pos = np.array([x, y, z])
        distance = np.linalg.norm(target_pos - current_pos)
        
        # 如果距离过小，使用速度控制
        if distance < min_distance:
            if distance < 1:  # 非常接近目标
                self.client.hoverAsync(vehicle_name=self.drone_name)
                return
            
            # 计算方向向量
            direction = target_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            # 使用速度控制移动短距离
            move_time = distance / velocity
            
            self.client.moveByVelocityAsync(
                direction[0] * velocity,
                direction[1] * velocity,
                direction[2] * velocity,
                move_time,
                vehicle_name=self.drone_name,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0)
            )
        else:
            # 正常使用位置控制
            self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.drone_name, 
                                            )

    def planning_loop(self):
        """Ego-Planner规划主循环"""
        while self.is_running:
            try:
                # 更新当前状态
                pose = self.get_drone_pose()
                self.current_position = pose['position']
                self.current_velocity = pose['linear_velocity']
                
                # 记录当前位置到历史航线
                self.trajectory_history.append(self.current_position.copy())
                # 限制历史航线点数量，避免内存过度使用
                if len(self.trajectory_history) > 500:
                    self.trajectory_history = self.trajectory_history[-400:]  # 保留最近400个点
                
                # 更新障碍物信息
                self.update_obstacles()
                
                # 生成初始控制点
                self.control_points = self.generate_bspline_control_points(
                    self.current_position, self.goal_position
                )
                
                # 寻找连续碰撞段
                colliding_segments = self.find_consecutive_colliding_segments(self.control_points)
                # print(f'segments={colliding_segments}')
                trajectory = None
                if colliding_segments:
                    # 初始化p-v对列表
                    pv_pairs_list = [[] for _ in range(len(self.control_points))]
                    
                    # 处理每个碰撞段
                    for segment in colliding_segments:
                        # 在环境中搜索路径
                        path = self.path_search(segment)
                        # print(f'path={path}')
                        # 为碰撞段中的每个控制点寻找p-v对
                        for j in range(segment[0], segment[1] + 1):
                            if j < len(self.control_points):
                                pv_pairs = self.find_pv_pairs(self.control_points[j], path)
                                # print(f'--for chosen control point:{self.control_points[j]}(index{j}), \npv_pairs{pv_pairs} ')
                                pv_pairs_list[j].extend(pv_pairs)
                        # print()
                    # 使用p-v对优化控制点
                    optimized_control_points = self.optimize_trajectory_with_pv(
                        self.control_points, pv_pairs_list
                    )
                    
                    # 生成B样条轨迹
                    # trajectory = self.bspline_interpolate(optimized_control_points)
        
                    trajectory = optimized_control_points
                    if draw:
                        draw_point = []
                        for points in trajectory:
                            draw_point.append(airsim.Vector3r(points[0], points[1], points[2]))
                        self.client.simPlotPoints(
                                draw_point[:2], 
                                color_rgba=[0.0, 1.0, 1.0, 1.0], # 优化点青色
                                size=20, 
                                duration=0.1, 
                                is_persistent=False)
                # 执行轨迹
                if trajectory is not None:
                    if len(trajectory) > 1:
                        # 选择轨迹上的下一个点
                        next_point = trajectory[1]
                        # self.safe_move_to_position(next_point[0], next_point[1], next_point[2], self.max_velocity)
                        self.client.moveToPositionAsync(
                            next_point[0], next_point[1], next_point[2], 
                            self.max_velocity, 
                            vehicle_name=self.drone_name,
                            # drivetrain=airsim.DrivetrainType.ForwardOnly,
                            # yaw_mode=airsim.YawMode(False, 0)
                        )
                else:
                    next_point = airsim.Vector3r(self.control_points[1][0], self.control_points[1][1], self.control_points[1][2])
                    
                    # self.safe_move_to_position(next_point[0], next_point[1], next_point[2], self.max_velocity)
                    self.client.moveToPositionAsync(
                        next_point.x_val, next_point.y_val, next_point.z_val,
                        velocity=self.max_velocity, 
                        vehicle_name=self.drone_name,
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(False, 0)
                    )
                
                # 检查是否到达目标
                distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
                self.goal_dis = distance_to_goal
                if distance_to_goal < 1.0:
                    self.is_running = False
                    break
                
                time.sleep(0.1)  # 控制循环频率
                
            except Exception as e:
                time.sleep(1)
    
    def plan_to_position(self, goal_position):
        """规划到指定位置"""
        self.goal_position = np.array(goal_position)
        self.is_running = True
        
        # 重置历史航线，从当前位置开始记录
        pose = self.get_drone_pose()
        self.trajectory_history = [pose['position'].copy()]
        
        # 启动规划线程
        self.planning_thread = threading.Thread(target=self.planning_loop)
        self.planning_thread.daemon = True
        self.planning_thread.start()
    
    def stop_planning(self):
        """停止规划"""
        self.is_running = False
        if self.planning_thread and self.planning_thread.is_alive():
            self.planning_thread.join(timeout=1.0)
        self.client.hoverAsync(vehicle_name=self.drone_name)
    
    
  
      





