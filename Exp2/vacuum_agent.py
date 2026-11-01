
from typing import Dict, List, Tuple
import random


class Environment:
    """Represents a discrete grid/room environment."""
    def __init__(self, rooms: List[str] = None, initial_dirt: Dict[str, str] = None):
        self.rooms = rooms if rooms is not None else ['A', 'B']
        if initial_dirt:
            self.dirt_status = initial_dirt.copy()
        else:
            self.dirt_status = {r: random.choice(['Clean', 'Dirty']) for r in self.rooms}

    def is_dirty(self, location: str) -> bool:
        return self.dirt_status.get(location) == 'Dirty'

    def clean(self, location: str):
        self.dirt_status[location] = 'Clean'

    def all_clean(self) -> bool:
        return all(status == 'Clean' for status in self.dirt_status.values())

    def __str__(self) -> str:
        return ", ".join(f"[{r}: {self.dirt_status[r]}]" for r in self.rooms)


class SimpleReflexVacuumAgent:
    """
    Simple Reflex Agent:
    Selects actions based solely on current percept (location, status).
    Ignores history or internal state.
    """
    def __init__(self):
        self.name = "Simple Reflex Agent"

    def act(self, location: str, status: str) -> str:
        # Rule 1: If current location is dirty, clean it
        if status == 'Dirty':
            return 'Suck'
        # Rule 2: If in room A and clean, move Right
        elif location == 'A':
            return 'Right'
        # Rule 3: If in room B and clean, move Left
        elif location == 'B':
            return 'Left'
        return 'NoOp'


class ModelBasedVacuumAgent:
    """
    Model-Based Reflex Agent:
    Maintains internal model of room cleanliness and current position.
    Stops operating (NoOp) when all known rooms are clean.
    """
    def __init__(self, rooms: List[str]):
        self.name = "Model-Based Reflex Agent"
        self.rooms = rooms
        # Internal state memory of the world
        self.model = {r: 'Unknown' for r in rooms}

    def act(self, location: str, status: str) -> str:
        # Update internal model
        self.model[location] = status

        # If dirty, suck immediately
        if status == 'Dirty':
            self.model[location] = 'Clean'
            return 'Suck'

        # If all rooms known to be clean, power down
        if all(self.model[r] == 'Clean' for r in self.rooms):
            return 'NoOp'

        # Look for the nearest dirty or unexplored room
        curr_idx = self.rooms.index(location)
        # Find if any room to the right is dirty/unknown
        right_dirty = any(self.model[r] != 'Clean' for r in self.rooms[curr_idx + 1:])
        left_dirty = any(self.model[r] != 'Clean' for r in self.rooms[:curr_idx])

        if right_dirty:
            return 'Right'
        elif left_dirty:
            return 'Left'
        else:
            return 'NoOp'


def run_simulation(agent_type: str = "simple", 
                   start_location: str = 'A', 
                   initial_dirt: Dict[str, str] = None, 
                   max_steps: int = 10):
    """Executes a simulation run measuring agent performance."""
    rooms = ['A', 'B']
    if initial_dirt is None:
        initial_dirt = {'A': 'Dirty', 'B': 'Dirty'}

    env = Environment(rooms=rooms, initial_dirt=initial_dirt)
    
    if agent_type == "simple":
        agent = SimpleReflexVacuumAgent()
    else:
        agent = ModelBasedVacuumAgent(rooms=rooms)

    curr_location = start_location
    performance_score = 0
    energy_consumed = 0

    print(f"\n--- Simulation: {agent.name} ---")
    print(f"Start Location: {start_location}")
    print(f"Initial State : {env}")
    print("-" * 65)
    print(f"{'Step':<5} | {'Percept (Loc, Status)':<23} | {'Action':<8} | {'Score':<6} | {'Environment State'}")
    print("-" * 65)

    for step in range(1, max_steps + 1):
        status = 'Dirty' if env.is_dirty(curr_location) else 'Clean'
        action = agent.act(curr_location, status)

        # Apply action
        if action == 'Suck':
            env.clean(curr_location)
            performance_score += 10
            energy_consumed += 1
        elif action == 'Right':
            curr_location = 'B'
            performance_score -= 1
            energy_consumed += 1
        elif action == 'Left':
            curr_location = 'A'
            performance_score -= 1
            energy_consumed += 1
        elif action == 'NoOp':
            pass

        print(f"{step:<5} | ({curr_location}, {status:<6}){' ' * 10} | {action:<8} | {performance_score:<6} | {env}")

        if action == 'NoOp' and env.all_clean():
            print(f"--> Goal achieved! All rooms clean. Agent halted cleanly at step {step}.")
            break

    print("-" * 65)
    print(f"Final Score          : {performance_score}")
    print(f"Total Energy Consumed: {energy_consumed} units")
    print(f"Final Room Status    : {env}")
    return performance_score


def evaluate_all_permutations():
    """Evaluates agent performance across all 8 standard environment percept combinations."""
    print("\n" + "=" * 65)
    print("      EVALUATING AGENT PERFORMANCE OVER ALL 8 PERCEPTS        ")
    print("=" * 65)
    permutations = [
        ('A', {'A': 'Clean', 'B': 'Clean'}),
        ('A', {'A': 'Clean', 'B': 'Dirty'}),
        ('A', {'A': 'Dirty', 'B': 'Clean'}),
        ('A', {'A': 'Dirty', 'B': 'Dirty'}),
        ('B', {'A': 'Clean', 'B': 'Clean'}),
        ('B', {'A': 'Clean', 'B': 'Dirty'}),
        ('B', {'A': 'Dirty', 'B': 'Clean'}),
        ('B', {'A': 'Dirty', 'B': 'Dirty'}),
    ]

    print(f"{'Start Loc':<10} | {'Room A':<8} | {'Room B':<8} | {'Simple Agent':<14} | {'Model-Based Agent'}")
    print("-" * 65)

    for loc, dirt in permutations:
        # Simple Reflex
        env1 = Environment(rooms=['A', 'B'], initial_dirt=dirt)
        agent1 = SimpleReflexVacuumAgent()
        c_loc = loc
        score1 = 0
        for _ in range(6):
            st = 'Dirty' if env1.is_dirty(c_loc) else 'Clean'
            act = agent1.act(c_loc, st)
            if act == 'Suck':
                env1.clean(c_loc)
                score1 += 10
            elif act == 'Right':
                c_loc = 'B'
                score1 -= 1
            elif act == 'Left':
                c_loc = 'A'
                score1 -= 1

        # Model Based
        env2 = Environment(rooms=['A', 'B'], initial_dirt=dirt)
        agent2 = ModelBasedVacuumAgent(rooms=['A', 'B'])
        c_loc2 = loc
        score2 = 0
        for _ in range(6):
            st = 'Dirty' if env2.is_dirty(c_loc2) else 'Clean'
            act = agent2.act(c_loc2, st)
            if act == 'Suck':
                env2.clean(c_loc2)
                score2 += 10
            elif act == 'Right':
                c_loc2 = 'B'
                score2 -= 1
            elif act == 'Left':
                c_loc2 = 'A'
                score2 -= 1
            elif act == 'NoOp':
                break

        print(f"{loc:<10} | {dirt['A']:<8} | {dirt['B']:<8} | Score: {score1:<7} | Score: {score2:<7}")
    print("=" * 65)


def print_peas():
    """Prints PEAS framework description for Vacuum Cleaner Agent."""
    print("\n" + "=" * 65)
    print("                PEAS FRAMEWORK SPECIFICATION                  ")
    print("=" * 65)
    print("Performance Measure: +10 points per clean room cleaned,")
    print("                     -1 point penalty per movement action,")
    print("                     0 penalty for NoOp (standby mode).")
    print("Environment        : 2 or N discrete rooms (A, B, ...),")
    print("                     Each room status: Clean or Dirty.")
    print("Actuators          : Suck, Move Left, Move Right, NoOp.")
    print("Sensors            : Current Location sensor, Dirt Detector sensor.")
    print("=" * 65)


if __name__ == "__main__":
    print_peas()
    # 1. Run Simple Reflex Agent demonstration
    run_simulation(agent_type="simple", start_location='A', initial_dirt={'A': 'Dirty', 'B': 'Dirty'}, max_steps=5)
    # 2. Run Model-Based Agent demonstration
    run_simulation(agent_type="model", start_location='A', initial_dirt={'A': 'Dirty', 'B': 'Dirty'}, max_steps=5)
    # 3. Permutations benchmark
    evaluate_all_permutations()

