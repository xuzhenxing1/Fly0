from typing import Dict, Any
from ..ui.ui_utils import Colors


class CodeExecutor:
    
    def __init__(self, drone, planner):
        self.drone = drone
        self.planner = planner
    
    def execute(self, code: str) -> bool:
        try:
            local_vars = {
                'drone': self.drone,
                'planner': self.planner,
                'print': print,
                'np': __import__('numpy'),
                'math': __import__('math'),
                'airsim': __import__('airsim')
            }
            exec(code, globals(), local_vars)
            print(f"{Colors.GREEN}✓ Code executed successfully{Colors.RESET}")
            return True
        except Exception as e:
            print(f"{Colors.RED}✗ Code execution failed: {str(e)}{Colors.RESET}")
            return False
