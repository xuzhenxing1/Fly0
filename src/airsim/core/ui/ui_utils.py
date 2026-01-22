from typing import Dict


class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'


class Banner:
    
    @staticmethod
    def show_banner(colors: Dict[str, str] = None):
        if colors is None:
            colors = {
                'CYAN': Colors.CYAN,
                'YELLOW': Colors.YELLOW,
                'GREEN': Colors.GREEN,
                'RESET': Colors.RESET
            }
        
        banner = f"""
{colors['CYAN']}╔══════════════════════════════════════════════════════════════╗
║          🚁 Drone Natural Language Control System  🚁        ║
╚══════════════════════════════════════════════════════════════╝{colors['RESET']}
{colors['YELLOW']}Available Commands:{colors['RESET']}
  {colors['GREEN']}!quit{colors['RESET']}   - Exit program
  {colors['GREEN']}!clear{colors['RESET']}  - Clear chat history
  {colors['GREEN']}!help{colors['RESET']}   - Show help
"""
        print(banner)
    
    @staticmethod
    def show_help(colors: Dict[str, str] = None):
        """Display help information"""
        if colors is None:
            colors = {
                'CYAN': Colors.CYAN,
                'YELLOW': Colors.YELLOW,
                'GREEN': Colors.GREEN,
                'RESET': Colors.RESET
            }
        
        help_text = f"""
{colors['CYAN']}═══════════════════════════════════════════════════════════════{colors['RESET']}
{colors['YELLOW']}Help Information{colors['RESET']}
{colors['CYAN']}═══════════════════════════════════════════════════════════════{colors['RESET']}

{colors['YELLOW']}Modes:{colors['RESET']}
  {colors['GREEN']}flight{colors['RESET']}  - Flight control mode, LLM generates control code only
  {colors['GREEN']}chat{colors['RESET']}    - Chat mode, LLM uses full capabilities

{colors['YELLOW']}Commands:{colors['RESET']}
  {colors['GREEN']}!quit{colors['RESET']}   - Exit program
  {colors['GREEN']}!clear{colors['RESET']}  - Clear chat history
  {colors['GREEN']}!mode{colors['RESET']}   - Switch mode (flight/chat)
  {colors['GREEN']}!help{colors['RESET']}   - Show help

{colors['YELLOW']}Control Examples:{colors['RESET']}
  - "Fly up 10 meters"
  - "Fly forward 5 meters"
  - "Turn left 90 degrees"
  - "Fly to target"
  - "Land"
"""
        print(help_text)
