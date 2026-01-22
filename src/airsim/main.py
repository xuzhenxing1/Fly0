import sys
import os
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.airsim.core import (
    DroneController, PathPlanner, Config, VisualTargetDetector,
    Colors, Banner, CommandClassifier, CodeExecutor,
    NavigationHandler, CommandProcessor, LLMInterface
)


class DroneAssistant:
    def __init__(self, config_path: str = "config.json", 
                 sys_prompt_path: str = "sysprompt/sysprompt.txt"):
        self.config = Config(config_path)
        
        self.drone = DroneController()
        
        vision_config = self.config.get_vision_config()
        visual_detector = None
        if vision_config["enabled"]:
            visual_detector = VisualTargetDetector(
                api_key=vision_config["api_key"],
                base_url=vision_config["base_url"],
                model=vision_config["model"]
            )
            print(f"{Colors.GREEN}✓ Vision detector enabled{Colors.RESET}")
        
        planner_config = self.config.get_planner_config()
        self.planner = PathPlanner(
            drone_name="Drone1",
            lidar_sensors=planner_config["lidar_sensors"],
            visual_detector=visual_detector
        )
        print(f"{Colors.GREEN}✓ Path planner initialized (LiDAR: {planner_config['lidar_sensors']}){Colors.RESET}")
        
        self.llm = LLMInterface(sys_prompt_path, self.config)
        self.command_classifier = CommandClassifier(self.llm)
        self.code_executor = CodeExecutor(self.drone, self.planner)
        self.navigation_handler = NavigationHandler(self.planner)
        self.command_processor = CommandProcessor(
            self.llm,
            self.command_classifier,
            self.code_executor,
            self.navigation_handler
        )
    
    def run(self):
        Banner.show_banner()
        
        print(f"{Colors.CYAN}Current mode: {self.command_processor.mode}{Colors.RESET}")
        print(f"{Colors.YELLOW}Enter command or !help for help{Colors.RESET}\n")
        
        while True:
            try:
                command = input(f"{Colors.GREEN}> {Colors.RESET}")
                self.command_processor.process(command)
                print()
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Drone Natural Language Control System")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--prompt", type=str, default="sysprompt/sysprompt.txt", help="System prompt file path")
    
    args = parser.parse_args()
    
    assistant = DroneAssistant(
        config_path=args.config,
        sys_prompt_path=args.prompt
    )
    assistant.run()


if __name__ == "__main__":
    main()
