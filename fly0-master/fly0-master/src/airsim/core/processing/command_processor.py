from typing import Optional
from ..ui.ui_utils import Colors, Banner


class CommandProcessor:
    
    def __init__(self, llm_interface, command_classifier, code_executor, navigation_handler):
        self.llm = llm_interface
        self.command_classifier = command_classifier
        self.code_executor = code_executor
        self.navigation_handler = navigation_handler
        self.mode = "flight"
    
    def process(self, command: str):
        command = command.strip()
        
        if not command:
            return
        
        if command == "!quit":
            print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
            import sys
            sys.exit(0)
        elif command == "!clear":
            self.llm.clear_history()
            print(f"{Colors.GREEN}✓ Chat history cleared{Colors.RESET}")
            return
        elif command == "!mode":
            modes = ["flight", "chat", "visual"]
            current_index = modes.index(self.mode)
            self.mode = modes[(current_index + 1) % len(modes)]
            print(f"{Colors.GREEN}✓ Mode switched to: {self.mode}{Colors.RESET}")
            return
        elif command == "!visual":
            self.mode = "visual"
            print(f"{Colors.GREEN}✓ Mode switched to: {self.mode}{Colors.RESET}")
            return
        elif command == "!help":
            Banner.show_help()
            return
        
        print(f"\n{Colors.CYAN}User:{Colors.RESET} {command}")
        
        if self.mode == "visual":
            self.navigation_handler.navigate_to_target(command)
        else:
            command_type = self.command_classifier.classify(command)
            
            if command_type == 'visual' and self.mode == "flight":
                self.navigation_handler.navigate_to_target(command)
            else:
                self._process_llm_command(command)
    
    def _process_llm_command(self, command: str):
        """Process commands that go through LLM"""
        print(f"{Colors.CYAN}Assistant:{Colors.RESET}", end="", flush=True)
        
        if self.mode == "flight":
            code = self.llm.process(command, mode="flight")
            print()
            
            if code:
                print(f"{Colors.YELLOW}Generated code:{Colors.RESET}")
                print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")
                print(code)
                print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}\n")
                
                self.code_executor.execute(code)
            else:
                print(f"{Colors.RED}✗ No valid code detected{Colors.RESET}")
        else:
            response = self.llm.process(command, mode="chat")
            print()
