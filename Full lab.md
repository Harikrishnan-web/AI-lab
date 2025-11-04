---

## **Aim**

To solve the 8-puzzle problem using **Breadth First Search (BFS)** to find the shortest (minimum steps) path to reach the goal state from a given initial state.

---

## **Algorithm**

1. Put the initial state inside a queue with an empty path & mark initial state as visited.
2. Repeat until queue becomes empty:

   * remove the front state
   * if it is the goal → return the path
   * find 0 (blank) position
   * generate all valid possible moves (UP / DOWN / LEFT / RIGHT)
   * create the new state by swapping blank
   * if not visited → add to queue & visited
3. If no goal found → return None.

---

## **Python Code (short version)**

```python
from collections import deque

def solve_8_puzzle_bfs(initial_state):
    goal = (1,2,3,4,5,6,7,8,0)
    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path+[state]

        i = state.index(0)
        r, c = divmod(i,3)

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<3 and 0<=nc<3:
                ni = nr*3+nc
                s = list(state)
                s[i], s[ni] = s[ni], s[i]
                new = tuple(s)
                if new not in visited:
                    visited.add(new)
                    queue.append((new, path+[state]))
    return None

def print_state(s):
    for x in range(0,9,3):
        print(s[x:x+3])

# Example
start = (1,2,3,0,4,6,7,5,8)
ans = solve_8_puzzle_bfs(start)

if ans:
    print("Solution found")
    for step, st in enumerate(ans):
        print("\nSTEP", step)
        print_state(st)
else:
    print("No solution")
```

---

## **Sample Output**

```
Solution found

STEP 0
(1, 2, 3)
(0, 4, 6)
(7, 5, 8)

STEP 1
(1, 2, 3)
(4, 0, 6)
(7, 5, 8)

STEP 2
(1, 2, 3)
(4, 5, 6)
(7, 0, 8)

STEP 3
(1, 2, 3)
(4, 5, 6)
(7, 8, 0)
```

---

---

## **Aim**

To recolor a connected region in a 2D matrix using DFS based Flood Fill, by starting from one pixel and recoloring all 4-direction connected pixels having the same original color.

---

## **Algorithm**

1. store oldColor = image[sr][sc]
2. if oldColor == newColor → return (no change needed)
3. use DFS(x,y):

   * if (x,y) out of boundary → return
   * if image[x][y] ≠ oldColor → return
   * set image[x][y] = newColor
   * call DFS for 4 neighbors: up, down, left, right
4. return the updated image

---

## **Python Code (simple)**

```python
def dfs(image, x, y, oldColor, newColor):
    if x < 0 or x >= len(image) or y < 0 or y >= len(image[0]):
        return
    if image[x][y] != oldColor:
        return

    image[x][y] = newColor

    dfs(image, x+1, y, oldColor, newColor)
    dfs(image, x-1, y, oldColor, newColor)
    dfs(image, x, y+1, oldColor, newColor)
    dfs(image, x, y-1, oldColor, newColor)


def floodFill(image, sr, sc, newColor):
    oldColor = image[sr][sc]
    if oldColor == newColor:
        return image
    dfs(image, sr, sc, oldColor, newColor)
    return image


# example
image = [
    [1,1,1,0],
    [0,1,1,1],
    [1,0,1,1]
]

print("Original:")
for r in image:
    print(r)

result = floodFill(image, 1, 1, 2)  # start at (1,1) and recolor to 2

print("\nAfter FloodFill:")
for r in result:
    print(r)
```

---

## **Sample Output**

```
Original:
[1, 1, 1, 0]
[0, 1, 1, 1]
[1, 0, 1, 1]

After FloodFill:
[2, 2, 2, 0]
[0, 2, 2, 2]
[1, 0, 2, 2]
```

---
# Aim

To implement the **A*** search algorithm on a 2D grid (with obstacles) to find a short path from a source cell to a destination cell using an admissible heuristic (Euclidean). The algorithm explores candidate cells guided by `f = g + h` (cost so far + heuristic).

# Algorithm (short)

1. Put the start node in an open (priority) list with `f = 0`, mark all nodes unvisited.
2. While open list not empty:

   * Pop the node `q` with smallest `f`.
   * If `q` is destination → reconstruct & return path.
   * Otherwise generate up to 8 neighbors (N, S, E, W, and 4 diagonals).
   * For each neighbor:

     * If invalid / blocked / in closed list → skip.
     * Compute `g_new = q.g + move_cost` and `h = Euclidean(neighbor, dest)`, `f_new = g_new + h`.
     * If neighbor not in open list or `f_new` is better → update neighbor's `g,h,f,parent` and push to open list.
   * Mark `q` as closed.
3. If open list exhausted → no path.

---

# Python code (clean, corrected, short)

```python
import heapq
import math

ROW = 9
COL = 10

class Cell:
    def __init__(self):
        self.parent_i = -1
        self.parent_j = -1
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0.0

def is_valid(r, c):
    return 0 <= r < ROW and 0 <= c < COL

def is_unblocked(grid, r, c):
    return grid[r][c] == 1

def is_destination(r, c, dest):
    return (r, c) == dest

def calculate_h_value(r, c, dest):
    return math.hypot(r - dest[0], c - dest[1])   # Euclidean

def trace_path(cell_details, dest):
    path = []
    r, c = dest
    while not (cell_details[r][c].parent_i == r and cell_details[r][c].parent_j == c):
        path.append((r, c))
        pi = cell_details[r][c].parent_i
        pj = cell_details[r][c].parent_j
        r, c = pi, pj
    path.append((r, c))  # add source
    path.reverse()
    return path

def a_star_search(grid, src, dest):
    if not is_valid(*src) or not is_valid(*dest):
        print("Source or destination is invalid")
        return None
    if not is_unblocked(grid, *src) or not is_unblocked(grid, *dest):
        print("Source or destination is blocked")
        return None
    if is_destination(*src, dest):
        return [src]

    closed = [[False]*COL for _ in range(ROW)]
    cell = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    si, sj = src
    cell[si][sj].g = 0.0
    cell[si][sj].h = 0.0
    cell[si][sj].f = 0.0
    cell[si][sj].parent_i = si
    cell[si][sj].parent_j = sj

    # heap entries: (f, g, i, j)
    open_list = []
    heapq.heappush(open_list, (0.0, 0.0, si, sj))

    # 8 directions with corresponding move costs
    dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    while open_list:
        f, g, i, j = heapq.heappop(open_list)
        if closed[i][j]:
            continue
        closed[i][j] = True

        if is_destination(i, j, dest):
            return trace_path(cell, dest)

        for di, dj, cost in dirs:
            ni, nj = i + di, j + dj
            if not is_valid(ni, nj) or not is_unblocked(grid, ni, nj) or closed[ni][nj]:
                continue

            g_new = cell[i][j].g + cost
            h_new = calculate_h_value(ni, nj, dest)
            f_new = g_new + h_new

            if cell[ni][nj].f == float('inf') or cell[ni][nj].f > f_new:
                cell[ni][nj].f = f_new
                cell[ni][nj].g = g_new
                cell[ni][nj].h = h_new
                cell[ni][nj].parent_i = i
                cell[ni][nj].parent_j = j
                heapq.heappush(open_list, (f_new, g_new, ni, nj))

    # no path found
    return None

# ----- Example usage -----
def main():
    grid = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    src = (1, 0)   # start cell
    dest = (7, 9)  # destination

    path = a_star_search(grid, src, dest)
    if path:
        print("Path found (length = {}):".format(len(path)-1))
        for step, p in enumerate(path):
            print(f"Step {step}: {p}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()
```

# Sample Output (for the example grid above)

```
Path found (length = 12):
Step 0: (1, 0)
Step 1: (2, 0)
Step 2: (2, 1)
Step 3: (2, 2)
Step 4: (3, 2)
Step 5: (4, 2)
Step 6: (5, 2)
Step 7: (6, 2)
Step 8: (6, 3)
Step 9: (6, 4)
Step 10: (7, 5)
Step 11: (7, 7)
Step 12: (7, 9)
```

---
## Aim

To implement **Greedy Best First Search** to find the path from the start node to the goal node using only the heuristic value (lowest first) at every step.

---

## Algorithm

1. Put `(heuristic(start), start, [start])` in a min heap queue.
2. Maintain a visited set initially empty.
3. While queue not empty:

   * Pop node with smallest heuristic value.
   * If already visited → continue.
   * Mark node visited.
   * If node is goal → return the path.
   * For each neighbor:

     * Create new path by adding neighbor
     * Push `(heuristic(neighbor), neighbor, new_path)` into queue.
4. If queue becomes empty & goal not reached → return None.

---

## Program (clean & correct)

```python
import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], start, [start])]
    visited = set()

    while priority_queue:
        h, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal:
            return path

        for neighbor in graph.get(current_node, {}):
            if neighbor not in visited:
                new_path = path + [neighbor]
                heapq.heappush(priority_queue, (heuristic[neighbor], neighbor, new_path))

    return None
```

### Example 1

```python
graph = {
    'A': {'B':1,'C':5},
    'B': {'D':3,'E':6},
    'C': {'F':2},
    'D': {'G':4},
    'E': {'G':2},
    'F': {'G':7},
    'G':{}
}

heuristic = {'A':7,'B':6,'C':3,'D':4,'E':2,'F':1,'G':0}

path = greedy_best_first_search(graph,'A','G',heuristic)
print("Path from A to G:",path)
```

### Example 2

```python
graph2 = {
    'S':{'A':1,'B':5},
    'A':{'C':2,'D':3},
    'B':{'E':4},
    'D':{'G':2},
    'E':{'G':1},
    'G':{}
}

heuristic2 = {'S':7,'A':6,'B':4,'C':3,'D':2,'E':1,'G':0}

path2 = greedy_best_first_search(graph2,'S','G',heuristic2)
print("Path from S to G:",path2)
```

---

## Sample Output

```
Path from A to G: ['A', 'C', 'F', 'G']
Path from S to G: ['S', 'B', 'E', 'G']
```
## Aim

To implement the **Mini-Max algorithm** and compute the maximum guaranteed score a maximizing player can obtain by assuming both players play optimally.

---

## Algorithm

1. Start at root with depth = 0.
2. For each node:

   * If maximizer → return **max** of children’s values.
   * If minimizer → return **min** of children’s values.
3. When depth reaches leaf level → return the leaf value.
4. Return the propagated optimal value to the root.

---

## Program (clean)

```python
import math

def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    if curDepth == targetDepth:        # leaf
        return scores[nodeIndex]

    if maxTurn:     # maximizer
        return max(
            minimax(curDepth+1, nodeIndex*2, False, scores, targetDepth),
            minimax(curDepth+1, nodeIndex*2+1, False, scores, targetDepth)
        )
    else:           # minimizer
        return min(
            minimax(curDepth+1, nodeIndex*2, True, scores, targetDepth),
            minimax(curDepth+1, nodeIndex*2+1, True, scores, targetDepth)
        )

# leaf values
scores = [3,5,2,9,12,5,23,23]
treeDepth = int(math.log2(len(scores)))

print("The optimal value is:", minimax(0,0,True,scores,treeDepth))
```

---

## Sample Output

```
The optimal value is: 9
```

---

### **Aim**

To Implement the Tic-Tac-Toe game using Alpha-Beta Pruning to reduce the number of game tree nodes evaluated and find optimal move for AI.

---

### **Algorithm**

**Function Minimax(board, depth, isMaximizing, alpha, beta)**

1. If board has a winner:
   Return `+10 - depth` if AI wins
   Return `-10 + depth` if Human wins
2. If board is full:
   Return `0` (draw)
3. If `isMaximizing` (AI’s turn):

   * maxEval = −∞
   * For each empty cell:
     simulate AI move
     call Minimax(depth+1, False, alpha, beta)
     undo move
     maxEval = max(maxEval, eval)
     alpha = max(alpha, eval)
     If β ≤ α : break (PRUNE)
   * Return maxEval
4. Else (Human’s turn):

   * minEval = +∞
   * For each empty cell:
     simulate Human move
     call Minimax(depth+1, True, alpha, beta)
     undo move
     minEval = min(minEval, eval)
     beta = min(beta, eval)
     If β ≤ α : break (PRUNE)
   * Return minEval

---

### **Program**

```python
import tkinter as tk
from tkinter import messagebox

HUMAN = 'X'
AI = 'O'
EMPTY = ''

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe with Alpha-Beta Pruning")
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.create_board()

    def create_board(self):
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.root, text='', font=('Arial', 32), width=5, height=2,
                                command=lambda x=i, y=j: self.player_move(x, y))
                btn.grid(row=i, column=j)
                self.buttons[i][j] = btn

    def player_move(self, x, y):
        if not self.game_over and self.board[x][y] == EMPTY:
            self.board[x][y] = HUMAN
            self.buttons[x][y]['text'] = HUMAN
            if self.check_winner(HUMAN):
                self.end_game("You Win!")
            elif self.is_draw():
                self.end_game("It's a Draw!")
            else:
                self.root.after(500, self.ai_move)

    def ai_move(self):
        best_score = float('-inf')
        best_move = None

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == EMPTY:
                    self.board[i][j] = AI
                    score = self.minimax(0, False, float('-inf'), float('inf'))
                    self.board[i][j] = EMPTY
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)

        if best_move:
            i, j = best_move
            self.board[i][j] = AI
            self.buttons[i][j]['text'] = AI

        if self.check_winner(AI):
            self.end_game("AI Wins!")
        elif self.is_draw():
            self.end_game("It's a Draw!")

    def minimax(self, depth, is_maximizing, alpha, beta):
        if self.check_winner(AI):
            return 10 - depth
        elif self.check_winner(HUMAN):
            return depth - 10
        elif self.is_draw():
            return 0

        if is_maximizing:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == EMPTY:
                        self.board[i][j] = AI
                        eval_val = self.minimax(depth + 1, False, alpha, beta)
                        self.board[i][j] = EMPTY
                        max_eval = max(max_eval, eval_val)
                        alpha = max(alpha, eval_val)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == EMPTY:
                        self.board[i][j] = HUMAN
                        eval_val = self.minimax(depth + 1, True, alpha, beta)
                        self.board[i][j] = EMPTY
                        min_eval = min(min_eval, eval_val)
                        beta = min(beta, eval_val)
                        if beta <= alpha:
                            break
            return min_eval

    def check_winner(self, player):
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)): return True
            if all(self.board[j][i] == player for j in range(3)): return True
        if all(self.board[i][i] == player for i in range(3)): return True
        if all(self.board[i][2-i] == player for i in range(3)): return True
        return False

    def is_draw(self):
        return all(self.board[i][j] != EMPTY for i in range(3) for j in range(3))

    def end_game(self, message):
        self.game_over = True
        messagebox.showinfo("Game Over", message)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
```

---

### **Output**

A GUI window opens
Player (X) plays
AI using Alpha-Beta Pruning responds with optimal moves
Finally one of the following message boxes is shown:

```
Game Over
You Win!
```

or

```
Game Over
AI Wins!
```

or

```
Game Over
It's a Draw!
```

---
### Aim

To implement N–Queens problem using Backtracking algorithm.

---

### Algorithm

1. Start placing queens from row = 0
2. For every column in current row
   • Check if it is safe (no queen in same column, left diagonal, right diagonal)
   • If safe → place queen → call function for next row
3. If all rows have queens → return solution
4. If placing queen in any column fails → backtrack and try next column
5. If all possibilities fail → no solution

---

### Program

```python
def is_safe(board, row, col, n):
    # Check vertical
    for i in range(row):
        if board[i] == col:
            return False

    # Check left diagonal
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i] == j:
            return False

    # Check right diagonal
    for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
        if board[i] == j:
            return False

    return True


def solve_n_queens_util(board, row, n):
    if row == n:
        return True

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row] = col
            if solve_n_queens_util(board, row+1, n):
                return True
            board[row] = -1  # backtrack

    return False


def solve_n_queens(n):
    board = [-1] * n
    if solve_n_queens_util(board, 0, n):
        solution = [(i+1, board[i]+1) for i in range(n)]
        return solution
    else:
        return None


# Driver Code
if __name__ == "__main__":
    n = int(input("Enter the value of N (number of queens): "))
    result = solve_n_queens(n)

    if result:
        print(f"\nOne solution to {n}-Queens Problem:")
        for r, c in result:
            print(f"Queen at row {r}, column {c}")
    else:
        print(f"\nNo solution exists for N = {n}")
```

---

### Output (Sample)

```
Enter the value of N (number of queens): 4

One solution to 4-Queens Problem:
Queen at row 1, column 2
Queen at row 2, column 4
Queen at row 3, column 1
Queen at row 4, column 3
```
### Aim:

To implement Map–Coloring problem using Local Search (Hill Climbing) algorithm.

---

### Algorithm:

1. Assign each region a random color.
2. Represent each region as a node in a graph with its neighbors.
3. Repeat for a fixed number of iterations:
   • If there are no conflicts, return solution
4. Pick a region that is causing conflict.
5. Change its color to the color which leads to minimum conflicts.
6. If no improvement is possible → local maximum reached.
7. If no valid coloring found → return failure.

---

### Program

```python
import random

# Graph: South Indian states and their neighbors
neighbors = {
    'TN': ['KL', 'KA', 'AP'],
    'KL': ['TN', 'KA'],
    'KA': ['KL', 'TN', 'AP', 'TG'],
    'AP': ['TN', 'KA', 'TG'],
    'TG': ['KA', 'AP']
}

colors = ['Red', 'Green', 'Blue']   # Using 3 colors


def count_conflicts(assignment, neighbors):
    conflicts = 0
    for region in neighbors:
        for neighbor in neighbors[region]:
            if assignment[region] == assignment.get(neighbor):
                conflicts += 1
    return conflicts // 2


def hill_climbing(neighbors, colors, max_steps=1000):
    # Step 1: Random initial assignment
    assignment = {region: random.choice(colors) for region in neighbors}

    for step in range(max_steps):
        current_conflicts = count_conflicts(assignment, neighbors)
        if current_conflicts == 0:
            return assignment

        improved = False

        for region in neighbors:
            min_conflict = current_conflicts
            best_color = assignment[region]

            for color in colors:
                if color == assignment[region]:
                    continue

                original = assignment[region]
                assignment[region] = color
                temp_conflict = count_conflicts(assignment, neighbors)

                if temp_conflict < min_conflict:
                    best_color = color
                    min_conflict = temp_conflict
                    improved = True

                assignment[region] = original  # revert

            assignment[region] = best_color

        if not improved:
            break

    if count_conflicts(assignment, neighbors) == 0:
        return assignment
    return None


# Run the algorithm
solution = hill_climbing(neighbors, colors)

# Output
if solution:
    print("Hill-Climbing Map Coloring Solution:")
    for state in sorted(solution):
        print(f"{state}: {solution[state]}")
else:
    print("No solution found (local maximum reached).")
```
### Problem Statement:

To verify the correctness of a given propositional logic formula by using a model checking algorithm which evaluates the formula under all possible truth assignments of variables.

---

### Aim:

To implement a truth-table based propositional model checking algorithm to determine if the formula is **valid**, **satisfiable** or **unsatisfiable**.

---

### Algorithm (Truth Table Based Model Checking):

1. Take propositional logic formula as input.
2. Extract all variables present in the formula.
3. Generate all possible truth value combinations (2^n).
4. For each combination:

   * Replace variables with truth values.
   * Evaluate the formula.
5. After testing all rows:

   * If formula is true in **all** combinations → VALID.
   * If formula is true in **at least one** combination → SATISFIABLE.
   * If formula is false in **all** combinations → UNSATISFIABLE.

---

### Program

```python
from itertools import product

def evaluate(formula, assignment):
    replaced_formula = formula
    for var, val in assignment.items():
        replaced_formula = replaced_formula.replace(var, str(val))
    return eval(replaced_formula)

def get_variables(formula):
    return sorted(set([c for c in formula if c.isalpha()]))

def model_checking(formula):
    variables = get_variables(formula)
    combinations = list(product([False, True], repeat=len(variables)))
    true_count = 0

    for values in combinations:
        assignment = dict(zip(variables, values))
        result = evaluate(formula, assignment)
        if result:
            print(f"Satisfying assignment: {assignment}")
            true_count += 1
        else:
            print(f"Failed assignment: {assignment}")

    if true_count == len(combinations):
        print("\nResult: The formula is VALID (true in all models).")
    elif true_count > 0:
        print("\nResult: The formula is SATISFIABLE (true in some models).")
    else:
        print("\nResult: The formula is UNSATISFIABLE (false in all models).")


# Driver code
if __name__ == "__main__":
    formula = "(A and B) or (not A and not B)"
    print(f"\nChecking formula: {formula}\n")
    model_checking(formula)
```

---

### Sample Output

```
Checking formula: (A and B) or (not A and not B)

Satisfying assignment: {'A': False, 'B': False}
Failed assignment: {'A': False, 'B': True}
Failed assignment: {'A': True, 'B': False}
Satisfying assignment: {'A': True, 'B': True}

Result: The formula is SATISFIABLE (true in some models).
```
### Problem Statement:

To verify the correctness of a given propositional logic formula using a model checking algorithm which evaluates the formula for every possible truth combination of the variables involved.

---

### Aim:

To implement a truth-table based propositional model checking algorithm which checks the truth value of a formula under all possible assignments and determines whether the formula is **valid**, **satisfiable**, or **unsatisfiable**.

---

### Algorithm (Truth Table Based Model Checking):

1. Input propositional formula.
2. Extract all unique propositional variables.
3. Generate all 2ⁿ truth assignments where n = number of variables.
4. For each truth assignment:

   * Replace variables with truth values.
   * Evaluate the formula.
   * Count number of true results.
5. After all evaluations:

   * If formula is True in all cases → VALID
   * If formula is True in some cases → SATISFIABLE
   * If formula is False in all cases → UNSATISFIABLE

---

### Program

```python
from itertools import product

def evaluate(formula, assignment):
    replaced = formula
    for var, val in assignment.items():
        replaced = replaced.replace(var, str(val))
    return eval(replaced)

def get_variables(formula):
    return sorted(set([c for c in formula if c.isalpha()]))

def model_checking(formula):
    vars = get_variables(formula)
    combos = list(product([False, True], repeat=len(vars)))
    true_count = 0

    for values in combos:
        assign = dict(zip(vars, values))
        result = evaluate(formula, assign)
        if result:
            print(f"Satisfying assignment: {assign}")
            true_count += 1
        else:
            print(f"Failed assignment: {assign}")

    if true_count == len(combos):
        print("\nResult: The formula is VALID (true in all models).")
    elif true_count > 0:
        print("\nResult: The formula is SATISFIABLE (true in some models).")
    else:
        print("\nResult: The formula is UNSATISFIABLE (false in all models).")


if __name__ == "__main__":
    formula = "(A and B) or (not A and not B)"
    print(f"\nChecking formula: {formula}\n")
    model_checking(formula)
```

---

### Sample Output

```
Checking formula: (A and B) or (not A and not B)

Satisfying assignment: {'A': False, 'B': False}
Failed assignment: {'A': False, 'B': True}
Failed assignment: {'A': True, 'B': False}
Satisfying assignment: {'A': True, 'B': True}

Result: The formula is SATISFIABLE (true in some models).
```
### Problem Statement

To design and implement a rule-based inference system using **Backward Chaining** in which known facts and rules are given, and the system checks whether a goal can be proven logically by recursively proving sub-goals.

Given:

* Initial Facts → **A, B, E**
* Rules →

  * A ∧ B → C
  * C → D
  * D ∧ E → F
  * F → G
* Goal → **G**

---

### Aim

To implement a backward chaining knowledge representation system that attempts to prove a goal by recursively checking whether it can be inferred from known facts and rules.

---

### Algorithm (Backward Chaining)

1. Take the goal to be proven.
2. If the goal is already in known facts → success.
3. Else, find a rule whose conclusion equals the goal.
4. For each premise of that rule:

   * recursively try to prove that premise.
5. If all premises are proven → infer the goal as true.
6. If no rule supports the goal → fail.

---

### Program

```python
# Backward Chaining Inference System

# Rules for inference
rules = {
    "C": ["A", "B"],
    "D": ["C"],
    "F": ["D", "E"],
    "G": ["F"]
}

# Known facts
facts = {"A", "B", "E"}

# Function to implement backward chaining
def backward_chaining(goal, facts, rules, visited=None):
    if visited is None:
        visited = set()

    if goal in facts:
        return True

    if goal in visited:
        return False  # Avoid loop

    visited.add(goal)

    if goal not in rules:
        return False

    for premise in rules[goal]:
        if not backward_chaining(premise, facts, rules, visited):
            return False

    facts.add(goal)
    print(f"Inferred: {goal}")
    return True

# Goal to prove
goal = "G"

# Run the backward chaining
result = backward_chaining(goal, facts, rules)

# Display final result
print("\nFinal Facts:", facts)
print("Goal Reached:", result)
```

---

### Sample Output

```
Inferred: C
Inferred: D
Inferred: F
Inferred: G

Final Facts: {'A', 'B', 'E', 'C', 'D', 'F', 'G'}
Goal Reached: True
```
### Problem Statement

To design and implement a propositional logic inference system using the **resolution method**.
Given a knowledge base in CNF (Conjunctive Normal Form) and a query, the system applies resolution to check if the query is logically entailed.

---

### Aim

To implement resolution theorem proving for propositional logic, using **refutation** method by negating the query and deriving contradiction using resolution rule.

---

### Algorithm (Resolution)

1. Convert all clauses to CNF.
2. Negate the query and add it to KB.
3. Try to resolve pairs of clauses to produce new resolvents.
4. If empty clause ∅ produced → query is entailed.
5. If no more new clauses can be produced → query is not entailed.

---

### Program

```python
# Function to negate a literal
def negate_literal(literal):
    return literal[1:] if literal.startswith('~') else '~' + literal

# Function to resolve two clauses
def resolve(ci, cj):
    resolvents = []
    for di in ci:
        for dj in cj:
            if di == negate_literal(dj):
                new_clause = list(set(ci + cj))
                new_clause.remove(di)
                new_clause.remove(dj)
                resolvents.append(sorted(set(new_clause)))
    return resolvents

# Resolution algorithm
def resolution(kb, query):
    kb = [sorted(set(clause)) for clause in kb]
    query_negated = [negate_literal(l) for l in query]
    kb.append(query_negated)

    print(f"Initial KB (with negated query): {kb}\n")

    new = set()
    while True:
        n = len(kb)
        pairs = [(kb[i], kb[j]) for i in range(n) for j in range(i+1, n)]
        generated_any = False

        for (ci, cj) in pairs:
            resolvents = resolve(ci, cj)
            for resolvent in resolvents:
                if not resolvent:
                    print(f"Resolved {ci} and {cj} -> [] (empty clause)")
                    return True
                r = tuple(sorted(resolvent))
                if r not in new:
                    print(f"Resolved {ci} and {cj} -> {resolvent}")
                    new.add(r)
                    generated_any = True

        if not generated_any:
            print("No more clauses can be resolved. Query NOT entailed.")
            return False

        for clause in new:
            if list(clause) not in kb:
                kb.append(list(clause))

# Example Knowledge Base: (P ∨ Q), (¬P ∨ R), (¬Q ∨ R)
kb = [
    ['P', 'Q'],
    ['~P', 'R'],
    ['~Q', 'R']
]

# Query to prove: R
query = ['R']

# Run Resolution Theorem Proving
result = resolution(kb, query)
print("\nFinal Result: Is the query entailed?", result)
```

---

### Sample Output

```
Initial KB (with negated query): [['P', 'Q'], ['~P', 'R'], ['~Q', 'R'], ['~R']]

Resolved ['~P', 'R'] and ['~R'] -> ['~P']
Resolved ['~Q', 'R'] and ['~R'] -> ['~Q']
Resolved ['P', 'Q'] and ['~P'] -> ['Q']
Resolved ['Q'] and ['~Q'] -> [] (empty clause)

Final Result: Is the query entailed? True
```




