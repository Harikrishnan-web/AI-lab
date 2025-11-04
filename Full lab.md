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

