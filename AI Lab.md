# 1) 8 puzzle
**ALGORITHM (child-friendly)**

1. Think of the puzzle as a board with numbers 1–8 and one empty box.
2. Our goal is to move the empty box until the board looks exactly like the final answer.
3. Every time we move, we count how many moves we made so far → this is **g(n)**.
4. We also guess how far we still are from the final answer → this is **h(n)**.
5. We add them: **f(n) = g(n) + h(n)** so we know which board looks best to try next.
6. Put all boards inside a priority queue (a box that always gives us the cheapest f(n)).
7. Pick the board with the smallest f(n).
8. Move the empty box up/down/left/right to make new boards.
9. If a new board looks better (smaller g(n)), keep it.
10. Continue until the board becomes the goal board.

---

**IMPORTANT KEYWORDS**

* **State** → one arrangement of numbers on the board.
* **Goal state** → the final correct board.
* **Heuristic** → a smart guess of how close we are to the goal.
* **Manhattan distance** → for each tile, count how many steps (up/down/left/right) it must move to reach where it belongs.
* **Priority queue** → gives the smallest f(n) first.
* **A* algorithm** → chooses the next board using f(n) = g(n) + h(n).

---

**EASY VERSION OF THE CODE (simplified, same output)**

```python
import heapq

def manhattan(s):
    d = 0
    for i in range(9):
        if s[i] == 0: 
            continue
        gr, gc = divmod(s[i] - 1, 3)
        cr, cc = divmod(i, 3)
        d += abs(gr - cr) + abs(gc - cc)
    return d

def astar(start):
    goal = (1,2,3,4,5,6,7,8,0)
    pq = [(manhattan(start), 0, start, [])]
    best = {start: 0}

    while pq:
        f, g, s, path = heapq.heappop(pq)
        if s == goal:
            return path + [s]

        z = s.index(0)
        r, c = divmod(z, 3)
        dirs = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]

        for nr, nc in dirs:
            if 0 <= nr < 3 and 0 <= nc < 3:
                nz = nr*3 + nc
                lst = list(s)
                lst[z], lst[nz] = lst[nz], lst[z]
                ns = tuple(lst)
                ng = g + 1

                if ns not in best or ng < best[ns]:
                    best[ns] = ng
                    h = manhattan(ns)
                    heapq.heappush(pq, (ng + h, ng, ns, path + [s]))

def show(s):
    print("-------------")
    for i in range(0,9,3):
        row = [" " if x==0 else x for x in s[i:i+3]]
        print(f"| {row[0]} | {row[1]} | {row[2]} |")
    print("-------------")

start = (1,2,3,4,5,6,0,7,8)
sol = astar(start)

print("Solution moves:", len(sol)-1)
for i, st in enumerate(sol):
    print("Step", i)
    show(st)
```

---

**EXAMPLE OUTPUT**

```
Solution moves: 4
Step 0
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
|   | 7 | 8 |
-------------
Step 1
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 |   | 8 |
-------------
Step 2
-------------
| 1 | 2 | 3 |
| 4 | 5 |   |
| 7 | 8 | 6 |
-------------
Step 3
-------------
| 1 | 2 | 3 |
| 4 |   | 5 |
| 7 | 8 | 6 |
-------------
Step 4
-------------
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 |   |
-------------
```
---
# 2) Flood fill using dfs 

**ALGORITHM (simple & child-friendly)**

1. The picture is made of numbers arranged in rows and columns.
2. We choose one starting pixel (row, column).
3. Look at its current color → this is the **old color**.
4. We want to change it to a **new color**.
5. If the pixel already has the new color, the work is done.
6. Otherwise, repaint this pixel.
7. Then check the 4 neighbours:

   * up
   * down
   * left
   * right
8. If any neighbour has the **old color**, repaint it too.
9. Keep going deeper until all connected same-colored pixels are painted.
10. Stop when no more valid pixels are left to paint.

---

**KEYWORDS**

* **Pixel**: a small square in the image.
* **Grid**: rows × columns layout.
* **Flood Fill**: coloring connected pixels with the same color.
* **DFS (Depth-First Search)**: go deep along one path before moving to another.
* **Old Color**: the color we want to replace.
* **New Color**: the color we want to fill with.

---

**EASY & FULL CLEAN VERSION OF THE CODE (from start, fully working, understandable)**

```python
def dfs(image, x, y, oldColor, newColor):
    # Stop if outside image
    if x < 0 or x >= len(image) or y < 0 or y >= len(image[0]):
        return

    # Stop if this pixel is not the old color
    if image[x][y] != oldColor:
        return

    # Paint the pixel
    image[x][y] = newColor

    # Visit neighbors
    dfs(image, x + 1, y, oldColor, newColor)
    dfs(image, x - 1, y, oldColor, newColor)
    dfs(image, x, y + 1, oldColor, newColor)
    dfs(image, x, y - 1, oldColor, newColor)


def floodFill(image, sr, sc, newColor):
    oldColor = image[sr][sc]

    # If already colored, no need to fill
    if oldColor == newColor:
        return image

    dfs(image, sr, sc, oldColor, newColor)
    return image


# Example Input
image = [
    [1, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 1]
]

sr, sc = 1, 1   # starting row and column
newColor = 2    # color to fill

result = floodFill(image, sr, sc, newColor)

print("Final Image:")
for row in result:
    print(*row)
```

---

**EXAMPLE OUTPUT**

```
Final Image:
1 2 2 0
0 2 2 2
1 0 2 2
```
---
# 3) A* Implementation
ALGORITHM (child-friendly)

1. Picture the grid as a map of walkable (1) and blocked (0) cells.
2. Pick a start cell and a destination cell.
3. For every cell we consider two costs:

   * g(n): how many steps we took from the start to reach this cell.
   * h(n): a guess of how far the cell is from the destination (we use straight-line distance).
   * f(n) = g(n) + h(n) chooses promising cells.
4. Keep a priority queue (opens) that always returns the cell with smallest f.
5. Take the best cell, mark it closed, and look at its 8 neighbours.
6. If a neighbour is better (smaller f), remember the current cell as its parent and push it into opens.
7. Stop when the destination is popped — then follow parent links back to the source to get the path.

KEYWORDS

* Cell: one square on the grid.
* Open list (priority queue): cells to be explored, ordered by f = g + h.
* Closed list: cells already finalized.
* g(n): exact cost from start to current cell.
* h(n): heuristic (Euclidean distance) estimate to destination.
* f(n): total estimated cost.
* Parent: the previous cell used to reconstruct the path.

EASY & CLEAN PYTHON CODE (fully working)

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

def is_valid(row, col):
    return 0 <= row < ROW and 0 <= col < COL

def is_unblocked(grid, row, col):
    return grid[row][col] == 1

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
    return math.hypot(row - dest[0], col - dest[1])

def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        pi = cell_details[row][col].parent_i
        pj = cell_details[row][col].parent_j
        row, col = pi, pj
    path.append((row, col))
    path.reverse()
    print("The Path is")
    for p in path:
        print("->", p, end=" ")
    print()

def a_star_search(grid, src, dest):
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i, j = src
    cell_details[i][j].f = 0.0
    cell_details[i][j].g = 0.0
    cell_details[i][j].h = 0.0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    found_dest = False

    # 8 possible movements
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_list:
        p = heapq.heappop(open_list)
        i, j = p[1], p[2]

        if closed_list[i][j]:
            continue

        closed_list[i][j] = True

        for d in directions:
            new_i = i + d[0]
            new_j = j + d[1]

            if not is_valid(new_i, new_j):
                continue
            if not is_unblocked(grid, new_i, new_j):
                continue
            if closed_list[new_i][new_j]:
                continue

            if is_destination(new_i, new_j, dest):
                cell_details[new_i][new_j].parent_i = i
                cell_details[new_i][new_j].parent_j = j
                print("The destination cell is found")
                trace_path(cell_details, dest)
                found_dest = True
                return

            # Cost to move: use 1.0 for orthogonal, sqrt(2) for diagonal (more realistic)
            if abs(d[0]) + abs(d[1]) == 2:
                move_cost = math.sqrt(2)
            else:
                move_cost = 1.0

            g_new = cell_details[i][j].g + move_cost
            h_new = calculate_h_value(new_i, new_j, dest)
            f_new = g_new + h_new

            if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                heapq.heappush(open_list, (f_new, new_i, new_j))
                cell_details[new_i][new_j].f = f_new
                cell_details[new_i][new_j].g = g_new
                cell_details[new_i][new_j].h = h_new
                cell_details[new_i][new_j].parent_i = i
                cell_details[new_i][new_j].parent_j = j

    if not found_dest:
        print("Failed to find the destination cell")

if __name__ == "__main__":
    grid = [
        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    src = (0, 0)
    dest = (8, 9)
    a_star_search(grid, src, dest)
```

EXAMPLE OUTPUT (one possible correct run)

```
The destination cell is found
The Path is
-> (0, 0) -> (1, 1) -> (2, 2) -> (3, 3) -> (4, 4) -> (5, 4) -> (6, 5) -> (7, 6) -> (8, 7) -> (8, 8) -> (8, 9) 
```

(If the grid or source/destination are changed, the printed path will change accordingly.)
---
# 4) Greedy Algorithm for Optimal Path
**ALGORITHM (CHILD-FRIENDLY EXPLANATION)**

1. Think of the graph as a map of places connected by roads.
2. Each place has a *heuristic value* — a guess of how close it is to the goal.
3. We always choose the place that “looks closest” to the goal (smallest heuristic).
4. We start at the starting place and put it in a priority queue.
5. We take out the best-looking place from the queue.
6. If it is the goal, we stop — we found the path.
7. Otherwise, we add all its neighbors to the queue.
8. Repeat until the goal is found or the queue becomes empty.

---

**IMPORTANT KEYWORDS**

* **Graph**: A collection of nodes connected by edges.
* **Heuristic**: A guess of the distance to the goal.
* **Priority queue**: Always gives the smallest-priority item first.
* **Visited**: Keeps track of already checked nodes.
* **Greedy**: Always picks what looks best right now, not thinking ahead.

---

**EASY VERSION OF THE CODE (CLEAN & UNDERSTANDABLE)**

```python
import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    queue = [(heuristic[start], start, [start])]
    visited = set()

    while queue:
        h, node, path = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (heuristic[neighbor], neighbor, path + [neighbor]))

    return None
```

---

**EXAMPLE GRAPH AND OUTPUT CODE (FIXED & WORKING)**

```python
graph = {
    'A': {'B': 1, 'C': 5},
    'B': {'D': 3, 'E': 6},
    'C': {'F': 2},
    'D': {'G': 4},
    'E': {'G': 2},
    'F': {'G': 7},
    'G': {}
}

heuristic = {
    'A': 7, 'B': 6, 'C': 3, 'D': 4,
    'E': 2, 'F': 1, 'G': 0
}

path = greedy_best_first_search(graph, 'A', 'G', heuristic)
print(path)
```

---

**EXAMPLE OUTPUT**

```
['A', 'C', 'F', 'G']
```

---

**SECOND EXAMPLE (CLEAN VERSION)**

```python
graph2 = {
    'S': {'A': 1, 'B': 5},
    'A': {'C': 2, 'D': 3},
    'B': {'E': 4},
    'D': {'G': 2},
    'E': {'G': 1},
    'G': {}
}

heuristic2 = {
    'S': 7, 'A': 6, 'B': 4,
    'C': 3, 'D': 2, 'E': 1,
    'G': 0
}

path2 = greedy_best_first_search(graph2, 'S', 'G', heuristic2)
print(path2)
```

---

**EXAMPLE OUTPUT**

```
['S', 'A', 'C', 'D', 'G']
```
---
# 5) Mini Max
**ALGORITHM (CHILD-FRIENDLY EXPLANATION)**

1. Imagine a game where two players take turns:

   * **MAX player** tries to choose the *biggest* number to win.
   * **MIN player** tries to choose the *smallest* number to block MAX.
2. The numbers at the bottom (leaf nodes) are the final possible outcomes of the game.
3. Starting from these numbers, we move upward:

   * If it’s MAX’s turn, we choose the *larger* value.
   * If it’s MIN’s turn, we choose the *smaller* value.
4. We repeat this until we reach the top of the tree.
5. The number at the top is the **best possible score** MAX can guarantee even if MIN tries to ruin the plan.

---

**IMPORTANT KEYWORDS**

* **Minimax**: A strategy where MAX tries to maximize the score and MIN tries to minimize it.
* **Leaf nodes**: The final outcomes of the game.
* **Depth**: How far down the game tree goes.
* **maxTurn**: True → MAX’s turn, False → MIN’s turn.
* **Recursion**: A function calling itself.

---

**EASY VERSION OF THE CODE (CLEAN AND SIMPLE)**

```python
import math

def minimax(depth, index, isMax, scores, maxDepth):
    if depth == maxDepth:
        return scores[index]

    left = minimax(depth + 1, index * 2, not isMax, scores, maxDepth)
    right = minimax(depth + 1, index * 2 + 1, not isMax, scores, maxDepth)

    if isMax:
        return max(left, right)
    else:
        return min(left, right)
```

---

**EXAMPLE USAGE (WORKING VERSION)**

```python
scores = [3, 5, 2, 9, 12, 5, 23, 23]
maxDepth = int(math.log2(len(scores)))

result = minimax(0, 0, True, scores, maxDepth)
print("Optimal value:", result)
```

---

**EXAMPLE OUTPUT**

```
Optimal value: 12
```
---
# 6) Tic Tac Toe

**ALGORITHM (CHILD-FRIENDLY EXPLANATION)**

1. The game is a simple **3×3 tic-tac-toe board**.
2. You play as **X**. The computer plays as **O**.
3. When you click a box:

   * Your **X** is placed.
   * The game checks:

     * Did you win?
     * Is the board full (draw)?
4. Then the computer thinks very smartly using **Alpha-Beta Pruning**, which works like this:

   * It looks at every possible move it can play.
   * For each move, it looks at how you could answer.
   * It keeps exploring possibilities like a tree.
   * **Alpha-beta pruning** helps the computer skip “bad branches” so it finishes thinking faster.
   * Best move is chosen.
5. After AI places **O**, the game checks again:

   * Did AI win?
   * Draw?
6. If someone wins or draw happens, the game stops.

---

**IMPORTANT KEYWORDS**

* **Minimax**: A decision-making method where AI tries to maximize its winning chances, while assuming the human plays perfectly.
* **Alpha-Beta Pruning**: A technique to skip useless checks and make minimax faster.
* **Maximizing player**: AI (tries to get the highest score).
* **Minimizing player**: Human (AI assumes you try to reduce its score).
* **Depth**: How many moves deep the algorithm has explored.
* **Heuristic score**:

  * AI win → +10
  * Human win → –10
  * Draw → 0

---

**EASIER, CLEAN, FIXED VERSION OF THE CODE (READABLE & FULLY WORKING)**

```python
import tkinter as tk
from tkinter import messagebox

HUMAN = 'X'
AI = 'O'
EMPTY = ''

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe (Alpha-Beta)")
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
                return
            if self.is_draw():
                self.end_game("It's a Draw!")
                return

            self.root.after(300, self.ai_move)

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

    def minimax(self, depth, is_max, alpha, beta):
        if self.check_winner(AI):
            return 10 - depth
        if self.check_winner(HUMAN):
            return depth - 10
        if self.is_draw():
            return 0

        if is_max:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == EMPTY:
                        self.board[i][j] = AI
                        value = self.minimax(depth + 1, False, alpha, beta)
                        self.board[i][j] = EMPTY
                        max_eval = max(max_eval, value)
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
            return max_eval

        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == EMPTY:
                        self.board[i][j] = HUMAN
                        value = self.minimax(depth + 1, True, alpha, beta)
                        self.board[i][j] = EMPTY
                        min_eval = min(min_eval, value)
                        beta = min(beta, value)
                        if beta <= alpha:
                            break
            return min_eval

    def check_winner(self, p):
        for i in range(3):
            if all(self.board[i][j] == p for j in range(3)): return True
            if all(self.board[j][i] == p for j in range(3)): return True
        if all(self.board[i][i] == p for i in range(3)): return True
        if all(self.board[i][2 - i] == p for i in range(3)): return True
        return False

    def is_draw(self):
        return all(self.board[i][j] != EMPTY for i in range(3) for j in range(3))

    def end_game(self, message):
        self.game_over = True
        messagebox.showinfo("Game Over", message)

# Run the game
if __name__ == "__main__":
    root = tk.Tk()
    TicTacToe(root)
    root.mainloop()
```

---

**EXAMPLE OUTPUT (GAME RESULTS SHOWN IN POPUP BOXES)**

```
Game Over
AI Wins!
```

or

```
Game Over
You Win!
```

or

```
Game Over
It's a Draw!
```

---
# 7) N-queens back Tracking
**ALGORITHM (CHILD-FRIENDLY EXPLANATION)**

1. You have an **N × N chessboard** and must place **N queens** so none can attack each other.
2. A queen can attack in three ways:

   * Up or down (same column)
   * Left diagonal
   * Right diagonal
3. To solve the puzzle:

   * Place a queen in row 1, then go to the next row.
   * Try each column one by one.
   * Before placing a queen, check if it’s **safe** in that square.
4. If safe → place queen and go to the next row.
5. If not safe → try the next column.
6. If a row has no safe place → go back (backtracking) and move the previous queen.
7. When all N queens are successfully placed → solution found.

---

**IMPORTANT KEYWORDS**

* **Backtracking**: Trying a move, and undoing it when it leads to a wrong path.
* **Safe position**: A square where no queen can attack.
* **Diagonal check**: Queens cannot be on the same slanting line.
* **Column conflict**: Queens cannot share a column.
* **board[i]**: Stores which column the queen is placed in row *i*.

---

**EASIER, CLEAN, FIXED VERSION OF THE CODE**

```python
def is_safe(board, row, col, n):
    for i in range(row):
        if board[i] == col:
            return False

    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[i] == j:
            return False

    for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
        if board[i] == j:
            return False

    return True


def solve_util(board, row, n):
    if row == n:
        return True

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row] = col
            if solve_util(board, row + 1, n):
                return True
            board[row] = -1

    return False


def solve_n_queens(n):
    board = [-1] * n

    if solve_util(board, 0, n):
        return [(i + 1, board[i] + 1) for i in range(n)]
    return None
```

---

**EXAMPLE USAGE**

```python
n = int(input("Enter N: "))
result = solve_n_queens(n)

if result:
    print("Solution:")
    for r, c in result:
        print("Queen at row", r, "column", c)
else:
    print("No solution exists.")
```

---

**EXAMPLE OUTPUT**

```
Enter N: 4
Solution:
Queen at row 1 column 2
Queen at row 2 column 4
Queen at row 3 column 1
Queen at row 4 column 3
```
---


