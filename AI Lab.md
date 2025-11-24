# 1) 8 puzzle
The best way to optimize the 8-puzzle is to use the **A\* (A-Star) Search Algorithm**.

## 🚀 A\* Search Algorithm: The Optimization

A\* is an improvement over BFS and DFS (Depth-First Search) because it uses a **heuristic function** to guess which path is most likely to lead to the goal. Instead of exploring all states equally (like BFS), A\* prioritizes states that look "closer" to the goal.

A\* doesn't use a simple queue; it uses a **Priority Queue** (implemented as a min-heap) to sort the next states based on a cost function, $f(n)$.

The cost function $f(n)$ for any state $n$ is calculated as:

$$f(n) = g(n) + h(n)$$

-----

### Understanding the Cost Components

| Component | Calculation ($g(n)$) | Role |
| :--- | :--- | :--- |
| **Path Cost** | The number of moves already taken to reach the current state $n$. | This is the "B" in BFS. It ensures we favor shorter paths. |
| **Heuristic Cost** | An estimate of the number of moves required to get from $n$ to the goal. | This is the "smarts" of A\*. It guides the search toward the goal. |
| **Total Cost** | $g(n) + h(n)$ | The estimated total cost (moves) of the solution if it goes through state $n$. |

-----

### The Best Heuristic for the 8-Puzzle

A good heuristic must be **admissible** (it must never overestimate the actual cost to reach the goal). The two most common admissible heuristics for the 8-puzzle are:

1.  **Misplaced Tiles Heuristic:** $h(n)$ is the count of tiles that are not in their correct final position.
2.  **Manhattan Distance Heuristic (Better):** $h(n)$ is the sum of the horizontal and vertical distances of every tile from its correct position.

We will use the **Manhattan Distance** since it provides a more accurate estimate and makes the search much faster.

## 🛠️ A\* Implementation (Python Code)

To implement A\*, we need to use a `heapq` (a min-heap implementation of a priority queue) and define the two cost functions.

```python
import heapq

def get_manhattan_distance(state, goal_state=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
    """Calculates the Manhattan distance heuristic (h(n))."""
    distance = 0
    for i in range(9):
        # Skip the blank tile (0)
        if state[i] == 0:
            continue
        
        # Find the tile's current position (current_row, current_col)
        current_row, current_col = divmod(i, 3)
        
        # Find the tile's goal position (goal_row, goal_col)
        # Note: We assume state[i] is the tile value (1 to 8)
        # We find the index of this tile's value in the goal_state tuple
        target_index = goal_state.index(state[i])
        goal_row, goal_col = divmod(target_index, 3)
        
        # Calculate Manhattan distance (abs(dx) + abs(dy))
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def solve_8_puzzle_astar(initial_state):
    """
    Solves the 8-puzzle using A* Search with Manhattan Distance.
    """
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    # Priority Queue stores: (f_cost, g_cost, state, path)
    # f_cost = g_cost + h_cost
    h_initial = get_manhattan_distance(initial_state)
    priority_queue = [(h_initial, 0, initial_state, [])] 
    
    # Store g_cost for visited states to check for shorter paths
    visited_g_costs = {initial_state: 0}

    while priority_queue:
        # Get the state with the lowest f_cost
        f_cost, g_cost, current_state, path = heapq.heappop(priority_queue)

        if current_state == goal_state:
            return path + [current_state]

        # Find the blank tile's position (0)
        blank_index = current_state.index(0)
        row, col = divmod(blank_index, 3)

        # Possible moves (Up, Down, Left, Right)
        moves = []
        if row > 0: moves.append((-1, 0))
        if row < 2: moves.append((1, 0))
        if col > 0: moves.append((0, -1))
        if col < 2: moves.append((0, 1))

        # Generate neighbors (next states)
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            new_blank_index = new_row * 3 + new_col
            
            # Create the new state by swapping tiles
            new_state_list = list(current_state)
            new_state_list[blank_index], new_state_list[new_blank_index] = \
                new_state_list[new_blank_index], new_state_list[blank_index]
            new_state = tuple(new_state_list)

            # g(n): Cost to reach the new state is 1 step more than current
            new_g_cost = g_cost + 1
            
            # Check if this new state has been reached with a lower or equal g_cost already
            if new_state not in visited_g_costs or new_g_cost < visited_g_costs[new_state]:
                
                # h(n): Heuristic cost for the new state
                h_cost = get_manhattan_distance(new_state)
                
                # f(n): Total estimated cost
                f_cost = new_g_cost + h_cost
                
                # Update records and add to priority queue
                visited_g_costs[new_state] = new_g_cost
                new_path = path + [current_state]
                heapq.heappush(priority_queue, (f_cost, new_g_cost, new_state, new_path))

    return None

def print_puzzle(state):
    """Prints the 8-puzzle state in a 3x3 grid format."""
    print("-------------")
    for i in range(0, 9, 3):
        row = state[i:i+3]
        # Format 0 as an empty space ' '
        print(f"| {' ' if row[0]==0 else row[0]} | {' ' if row[1]==0 else row[1]} | {' ' if row[2]==0 else row[2]} |")
    print("-------------")

# --- Example Usage (Same initial state as before) ---
if __name__ == "__main__":
    # Solvable state (requires 4 moves with BFS/A*)
    initial_puzzle = (1, 2, 3, 4, 5, 6, 0, 7, 8) 
    
    # A harder state (requires 22 moves) - A* is essential here!
    # initial_puzzle = (8, 6, 7, 2, 5, 4, 3, 0, 1)

    print("Searching for solution using A* Search...")
    solution_path = solve_8_puzzle_astar(initial_puzzle)

    print("\n" + "="*30)
    print("A* Solution Output")
    print("="*30)
    
    if solution_path:
        # Subtract 1 because the initial state is counted as the first step
        print(f"Solution Found in {len(solution_path) - 1} moves!")
        for i, state in enumerate(solution_path):
            print(f"\nStep {i}:")
            print_puzzle(state)
    else:
        print("No solution found for the given initial state.")
```

-----

### Sample Output

```
Searching for solution using A* Search...

==============================
A* Solution Output
==============================
Solution Found in 4 moves!

Step 0:
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
|   | 7 | 8 |
-------------

Step 1:
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 |   | 8 |
-------------

Step 2:
-------------
| 1 | 2 | 3 |
| 4 | 5 |   |
| 7 | 8 | 6 |
-------------

Step 3:
-------------
| 1 | 2 | 3 |
| 4 |   | 5 |
| 7 | 8 | 6 |
-------------

Step 4:
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 |   |
-------------
```

-----

Would you like to try running this A\* code on the challenging 22-move puzzle I mentioned: `(8, 6, 7, 2, 5, 4, 3, 0, 1)`?
