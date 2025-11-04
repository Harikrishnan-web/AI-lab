AIM (simple)

To solve the 8-Puzzle problem using Breadth-First Search (BFS).
We start from a given initial puzzle state and search level by level to reach the goal state.
Finally, we display the correct sequence of moves and the minimum number of steps.

ALGORITHM (simple)

Take the initial puzzle state and push it into a queue (state + path).

Define goal state → (1, 2, 3, 4, 5, 6, 7, 8, 0)

Create a visited set to avoid repeating states.

Repeat while queue not empty:

pop the front element (current state + path)

if current state == goal → return path + current state

find index of blank (0)

calculate which moves are possible (Up / Down / Left / Right)

for each possible move → swap blank → generate new state

if new state not visited → add it to visited and push to queue

If queue ends without reaching goal → no solution.

PROGRAM (simple but same logic)
from collections import deque

def solve_8_puzzle_bfs(initial_state):
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path + [current_state]

        blank = current_state.index(0)
        row, col = divmod(blank, 3)

        moves = []
        if row > 0: moves.append((-1, 0))  # up
        if row < 2: moves.append((1, 0))   # down
        if col > 0: moves.append((0, -1))  # left
        if col < 2: moves.append((0, 1))   # right

        for r, c in moves:
            nr, nc = row + r, col + c
            new_blank = nr*3 + nc

            new_state = list(current_state)
            new_state[blank], new_state[new_blank] = new_state[new_blank], new_state[blank]
            new_state = tuple(new_state)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [current_state]))

    return None

def print_puzzle(state):
    for i in range(0,9,3):
        print(state[i:i+3])


# Example run
if __name__ == "__main__":
    start = (1,2,3,0,4,6,7,5,8)
    result = solve_8_puzzle_bfs(start)

    if result:
        print("Solution Found!")
        for i, st in enumerate(result):
            print(f"\nStep {i}:")
            print_puzzle(st)
    else:
        print("No solution found")