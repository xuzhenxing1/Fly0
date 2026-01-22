"""Visual target detection module using multimodal LLM"""
import airsim
import numpy as np
import math
import cv2
import base64
import re
import time
from typing import List, Tuple, Optional
from ..ui.ui_utils import Colors


class VisualTargetDetector:
    """Visual target detector using multimodal LLM"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        try:
            from openai import OpenAI
            self.openai = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.base_url = base_url
            self.api_key = api_key
            print(f"Visual detector initialized with model: {model}")
        except ImportError:
            print("Warning: openai module not installed")
            self.openai = None
        except Exception as e:
            print(f"Warning: Visual detector initialization failed: {e}")
            self.openai = None
        
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 360.0
        self.cy = 240.0
        self.is_ollama = "localhost:11434" in base_url or "11434" in base_url
    
    def capture_rgbd(self, client: airsim.MultirotorClient) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Capture RGB and depth images"""
        requests = [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
        ]
        responses = client.simGetImages(requests)
        
        rgb_response = responses[0]
        img_rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb.reshape(rgb_response.height, rgb_response.width, 3)
        
        depth_response = responses[1]
        depth_matrix = np.array(depth_response.image_data_float)
        depth_matrix = depth_matrix.reshape(depth_response.height, depth_response.width)
        
        return img_rgb, depth_matrix, rgb_response.height, rgb_response.width
    
    def encode_image(self, img_rgb: np.ndarray) -> str:
        """Convert image to base64 encoding"""
        success, buffer = cv2.imencode('.png', img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not success:
            raise Exception("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')
    
    def detect_target_in_rgb(self, img_rgb: np.ndarray, user_input: str) -> Tuple[bool, Tuple[int, int]]:
        """Detect target in RGB image using multimodal LLM"""
        if self.openai is None:
            print(f"{Colors.RED} OpenAI client is None{Colors.RESET}")
            return False, (0, 0)
        
        try:
            import os
            dir = "images"
            os.makedirs(dir, exist_ok=True)
            timestamp = int(time.time())
            
            base64_image = self.encode_image(img_rgb)
            
            prompt = f""" 
                        User command: "{user_input}"
                        
                        Find the target object in the image based on the user's complete command.
                        Image size is 720*480, with top-left corner as origin (0,0).
                        When understanding the command, note:
                        - User may include direction info (e.g., "left front", "right", "ahead"), indicating relative position in image
                        - User may include target type (e.g., "car", "ball", "door"), which is the object to detect
                        - Combine direction and type to locate the most matching target
                        Handle as follows:
                        1. If the target exists, return only the center pixel coordinate (X,Y) of the most relevant target, format as "(X,Y)"
                        2. If target does not exist, return "(0,0)"
                        Return only coordinates, no other content.
                    """
            
            if self.is_ollama:
                result = self._detect_with_ollama(prompt, base64_image)
            else:
                result = self._detect_with_openai(prompt, base64_image)
            
            print(f"{Colors.CYAN}LLM原始响应: {result}{Colors.RESET}")
            
            bbox_match = re.search(r'\((\d+),(\d+),(\d+),(\d+)\)', result)
            
            if bbox_match:
                x1, y1, x2, y2 = map(int, bbox_match.groups())
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                
                if x == 0 and y == 0:
                    print(f"{Colors.YELLOW}Target found at (0,0), treating as not found{Colors.RESET}")
                    return False, (0, 0)
                
                img_with_mark = img_rgb.copy()
                cv2.circle(img_with_mark, (x, y), 10, (0, 0, 255), 2)
                cv2.circle(img_with_mark, (x, y), 2, (0, 255, 0), -1)
                cv2.rectangle(img_with_mark, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                print(f"{Colors.GREEN} Target found at bbox: ({x1},{y1},{x2},{y2}), center: ({x},{y}){Colors.RESET}")
                return True, (x, y)
            
            center_match = re.search(r'\((\d+),(\d+)\)', result)
            if center_match:
                x, y = map(int, center_match.groups())
                if x == 0 and y == 0:
                    print(f"{Colors.YELLOW}Target found at (0,0), treating as not found{Colors.RESET}")
                    return False, (0, 0)
                
                img_with_mark = img_rgb.copy()
                cv2.circle(img_with_mark, (x, y), 10, (0, 0, 255), 2)
                cv2.circle(img_with_mark, (x, y), 2, (0, 255, 0), -1)
                
                marked_image_path = os.path.join(dir, f"detect_{timestamp}_marked.png")
                cv2.imwrite(marked_image_path, img_with_mark, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"{Colors.GREEN}Target found at center: ({x},{y}), saved to {marked_image_path}{Colors.RESET}")
                return True, (x, y)
            
            print(f"{Colors.YELLOW} No target found in result{Colors.RESET}")
            return False, (0, 0)
        except Exception as e:
            print(f"{Colors.RED}Exception in detect_target_in_rgb: {str(e)}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False, (0, 0)
    
    def _detect_with_ollama(self, prompt: str, base64_image: str) -> str:
        """Detect target using Ollama API"""
        import requests
        import json
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "images": [base64_image],
            "stream": False
        }
        
        print(f"{Colors.CYAN}调用Ollama API，模型: {self.model}{Colors.RESET}")
        print(f"{Colors.CYAN}Base64图像长度: {len(base64_image)}{Colors.RESET}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=120
            )
            
            print(f"{Colors.CYAN}Ollama API响应状态码: {response.status_code}{Colors.RESET}")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                print(f"{Colors.CYAN}Ollama API响应内容: {content}{Colors.RESET}")
                return content
            else:
                print(f"{Colors.RED}[ERROR] Ollama API error: {response.status_code}{Colors.RESET}")
                print(f"{Colors.RED}[ERROR] Response: {response.text}{Colors.RESET}")
                return ""
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Ollama API request failed: {str(e)}{Colors.RESET}")
            return ""
    
    def _detect_with_openai(self, prompt: str, base64_image: str) -> str:
        """Detect target using OpenAI-compatible API"""
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=20
        )
        
        return response.choices[0].message.content.strip()
    
    def calculate_target_position(self, img_coords: Tuple[int, int], depth_matrix: np.ndarray,
                                  client: airsim.MultirotorClient) -> Tuple[float, float, float]:
        """Calculate target world coordinates"""
        u, v = img_coords
        height, width = depth_matrix.shape
        
        if v >= height or u >= width or v < 0 or u < 0:
            return (0, 0, 0)
        
        radius_pixels = 5
        valid_depths = []
        
        min_x = max(0, int(u - radius_pixels))
        max_x = min(width - 1, int(u + radius_pixels))
        min_y = max(0, int(v - radius_pixels))
        max_y = min(height - 1, int(v + radius_pixels))
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.sqrt((x - u) ** 2 + (y - v) ** 2)
                if distance <= radius_pixels:
                    depth = depth_matrix[y, x]
                    if depth > 0 and not math.isinf(depth) and not math.isnan(depth):
                        valid_depths.append(depth)
        
        if not valid_depths:
            return (0, 0, 0)
        
        valid_depths.sort()
        depth = valid_depths[0]
        
        X_cam = depth
        Y_cam = (u - self.cx) * depth / self.fx
        Z_cam = (v - self.cy) * depth / self.fy
        
        drone_pose = client.simGetVehiclePose()
        drone_position = drone_pose.position
        drone_orientation = drone_pose.orientation
        
        q = np.array([drone_orientation.x_val, drone_orientation.y_val, 
                      drone_orientation.z_val, drone_orientation.w_val])
        rotation_matrix = self._quaternion_to_rotation_matrix(q)
        
        camera_point = np.array([X_cam, Y_cam, Z_cam])
        world_point = np.dot(rotation_matrix, camera_point)
        
        X_world = drone_position.x_val + world_point[0]
        Y_world = drone_position.y_val + world_point[1]
        Z_world = drone_position.z_val + world_point[2]
        
        return X_world, Y_world, Z_world
    
    def detect_visual_target(self, client: airsim.MultirotorClient, user_input: str, 
                            search_360: bool = True) -> Tuple[bool, Tuple[float, float, float]]:
        """Detect visual target"""
        img_rgb, depth_matrix, height, width = self.capture_rgbd(client)
        target_found, img_coords = self.detect_target_in_rgb(img_rgb, user_input)
        
        if not target_found and search_360:
            print("Target not found, starting 360-degree search...")
            search_steps = 8
            rotation_angle = 360 / search_steps
            
            for i in range(search_steps):
                self.rotate_drone(client, rotation_angle)
                img_rgb, depth_matrix, height, width = self.capture_rgbd(client)
                target_found, img_coords = self.detect_target_in_rgb(img_rgb, user_input)
                
                if target_found:
                    print(f"Target found after {i+1} rotations")
                    break
        
        if not target_found:
            print("Target not found")
            return False, (0, 0, 0)
        
        print(f"Target image coordinates: {img_coords}")
        
        target_pos = self.calculate_target_position(img_coords, depth_matrix, client)
        
        return True, target_pos
    
    def rotate_drone(self, client: airsim.MultirotorClient, degrees: float, velocity: float = 30.0):
        """Rotate drone in place"""
        current_pose = client.simGetVehiclePose()
        current_yaw = math.atan2(
            2.0 * (current_pose.orientation.w_val * current_pose.orientation.z_val + 
                   current_pose.orientation.x_val * current_pose.orientation.y_val),
            1.0 - 2.0 * (current_pose.orientation.y_val ** 2 + current_pose.orientation.z_val ** 2)
        )
        target_yaw = current_yaw + math.radians(degrees)
        client.rotateToYawAsync(target_yaw, velocity).join()
        time.sleep(1)
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
