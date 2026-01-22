from typing import Tuple
from ..ui.ui_utils import Colors


class NavigationHandler:
    def __init__(self, planner):
        self.planner = planner
    
    def navigate_to_target(self, user_command: str) -> bool:
        print(f"{Colors.YELLOW}Navigating with EgoPlanner...{Colors.RESET}")
        
        try:
            client = self.planner.client
            detector = self.planner.visual_detector
            
            # print(f"{Colors.CYAN}[DEBUG] Visual detector: {detector}{Colors.RESET}")
            
            if not detector:
                print(f"{Colors.RED}✗ Visual detector not enabled{Colors.RESET}")
                return False
            
            # print(f"{Colors.CYAN}[DEBUG] Starting visual target detection for command: {user_command}{Colors.RESET}")
            success, target_position = detector.detect_visual_target(client, user_command, search_360=True)
            
            # print(f"{Colors.CYAN}[DEBUG] Detection success: {success}, target_position: {target_position}{Colors.RESET}")
            
            if success and target_position != (0, 0, 0):
                print(f"{Colors.GREEN}✓ Target position: X={target_position[0]:.2f}, Y={target_position[1]:.2f}, Z={target_position[2]:.2f}{Colors.RESET}")
                
                goal_position = __import__('numpy').array([target_position[0], target_position[1], target_position[2]])
                
                print(f"{Colors.YELLOW}Starting path planning and navigation...{Colors.RESET}")
                self.planner.take_control()
                self.planner.plan_to_position(goal_position)
                print(f"{Colors.GREEN}✓ Navigation complete{Colors.RESET}")
                return True
            else:
                print(f"{Colors.RED}✗ Failed to get valid target position{Colors.RESET}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}✗ Navigation error: {str(e)}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False
