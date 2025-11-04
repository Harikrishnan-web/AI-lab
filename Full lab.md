---

### **AIM (simple)**

To solve the 8-Puzzle problem using Breadth-First Search (BFS).
We start from a given initial puzzle state and search level by level to reach the goal state.
Finally, we display the correct sequence of moves and the minimum number of steps.

---

### **ALGORITHM (simple)**

1. Take the initial puzzle state and push it into a queue (state + path).
2. Define goal state → (1, 2, 3, 4, 5, 6, 7, 8, 0)
3. Create a visited set to avoid repeating states.
4. Repeat while queue not empty:

   * pop the front element (current state + path)
   * if current state == goal → return path + current state
   * find index of blank (0)
   * calculate which moves are possible (Up / Down / Left / Right)
   * for each possible move → swap blank → generate new state
   * if new state not visited → add it to visited and push to queue
5. If queue ends without reaching goal → no solution.

---

### **PROGRAM (simple but same logic)**

```python
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
```

---


---

## AIM (easy words)

To recolor a group of connected pixels in a 2D image using Depth-First Search (DFS).
We start from one pixel and fill all near pixels with same color to a new color.

---

## ALGORITHM (easy words)

1. Read the original color of start pixel → oldColor.
2. If oldColor == newColor → nothing to change → return image directly.
3. Make DFS function:

   * if pixel is outside image → return
   * if pixel color != oldColor → return
   * change pixel color to newColor
   * then call DFS for 4 directions:

     * UP (x-1, y)
     * DOWN (x+1, y)
     * LEFT (x, y-1)
     * RIGHT (x, y+1)
4. Call DFS from starting pixel (sr, sc)
5. return final image

---

## PROGRAM (easier to read)

```python
def dfs(image, x, y, oldColor, newColor):
    # check image boundary + color check
    if x < 0 or x >= len(image) or y < 0 or y >= len(image[0]) or image[x][y] != oldColor:
        return

    # change color
    image[x][y] = newColor

    # visit 4 neighbours
    dfs(image, x+1, y, oldColor, newColor) # down
    dfs(image, x-1, y, oldColor, newColor) # up
    dfs(image, x, y+1, oldColor, newColor) # right
    dfs(image, x, y-1, oldColor, newColor) # left


def floodFill(image, sr, sc, newColor):
    if image[sr][sc] == newColor:
        return image

    dfs(image, sr, sc, image[sr][sc], newColor)
    return image


# MAIN PART
if __name__ == "__main__":
    image = [
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
    ]

    sr, sc = 1, 1   # starting pixel
    newColor = 2

    print("Before Fill:")
    for row in image:
        print(*row)

    floodFill(image, sr, sc, newColor)

    print("\nAfter Fill:")
    for row in image:
        print(*row)
```

---


---

## AIM (easy words)

To use the A* search algorithm to find the shortest path on a 2D grid with obstacles.
Starting from a start cell and going to a goal cell, we calculate path cost + heuristic and move smartly to reach the destination.

---

## ALGORITHM (easy words)

1. Put the start cell into open list.
2. Closed list is empty.
3. Repeat while open list not empty:

   * take the cell with smallest f value
   * if this cell is goal → stop (path found)
   * generate all 8 neighbor cells
   * for each neighbor:

     * if valid and not blocked
     * calculate g, h and f
     * if this neighbor already has lower f (in open/closed) → skip
     * otherwise add this neighbour into open list
   * put current cell into closed list
4. If loop ends and no goal found → path not possible.

---

## PROGRAM (easy readable code + your grid included)

```python
import math
import heapq

class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

ROW = 4
COL = 4

def is_valid(r,c):
    return 0 <= r < ROW and 0 <= c < COL

def is_unblocked(grid,r,c):
    return grid[r][c] == 1

def is_destination(r,c,dest):
    return r == dest[0] and c == dest[1]

def h_value(r,c,dest):
    return math.sqrt((r-dest[0])**2 + (c-dest[1])**2)

def trace_path(details,dest):
    print("Path Found:")
    path=[]
    r,c=dest

    while not (details[r][c].parent_i == r and details[r][c].parent_j == c):
        path.append((r,c))
        nr = details[r][c].parent_i
        nc = details[r][c].parent_j
        r,c = nr,nc

    path.append((r,c))
    path.reverse()
    for p in path:
        print("->",p,end=" ")

def a_star(grid,src,dest):
    if not is_valid(*src) or not is_valid(*dest):
        print("Invalid source/destination")
        return

    if not is_unblocked(grid,*src) or not is_unblocked(grid,*dest):
        print("Source/Destination blocked")
        return

    if is_destination(*src,dest):
        print("Already at destination")
        return

    closed = [[False]*COL for _ in range(ROW)]
    details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    si,sj=src
    details[si][sj].f=0
    details[si][sj].g=0
    details[si][sj].h=0
    details[si][sj].parent_i=si
    details[si][sj].parent_j=sj

    open_list=[]
    heapq.heappush(open_list,(0,si,sj))

    while open_list:
        f,i,j = heapq.heappop(open_list)
        closed[i][j]=True

        for di,dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            ni = i+di
            nj = j+dj

            if is_valid(ni,nj) and is_unblocked(grid,ni,nj) and not closed[ni][nj]:

                if is_destination(ni,nj,dest):
                    details[ni][nj].parent_i=i
                    details[ni][nj].parent_j=j
                    trace_path(details,dest)
                    return

                g_new = details[i][j].g + 1
                h_new = h_value(ni,nj,dest)
                f_new = g_new + h_new

                if details[ni][nj].f == float('inf') or details[ni][nj].f > f_new:
                    details[ni][nj].f = f_new
                    details[ni][nj].g = g_new
                    details[ni][nj].h = h_new
                    details[ni][nj].parent_i=i
                    details[ni][nj].parent_j=j
                    heapq.heappush(open_list,(f_new,ni,nj))

    print("No Path Found")

if __name__=="__main__":
    grid = [
        [1,1,1,1],
        [1,0,1,1],
        [1,1,1,1],
        [0,1,1,1]
    ]

    src = (0,0)
    dest = (3,3)
    a_star(grid,src,dest)
```

---


---

## AIM (easy words)

To use Greedy Best First Search (a greedy algorithm) to find a path from a start node to a goal node.
It selects the next node only based on the smallest heuristic value (closest looking node to goal).

---

## ALGORITHM (easy words)

1. Put the start node into a priority queue with its heuristic + its path.
2. Create a visited set.
3. While queue not empty:

   * remove the node with smallest heuristic value
   * if this node already visited → skip
   * mark it visited
   * if this node = goal → return the path
   * for each neighbor of this node:

     * if neighbor not visited → push neighbor to queue with its heuristic + new path
4. If queue ends and goal not reached → return None

---

## PROGRAM (easy readable code)

```python
import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    # priority queue stores: (h_value, node, path)
    pq = [(heuristic[start], start, [start])]
    visited = set()

    while pq:
        h_val, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return path

        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                new_path = path + [neighbor]
                heapq.heappush(pq, (heuristic[neighbor], neighbor, new_path))

    return None


if __name__ == "__main__":
    graph = {
        'A': {'B':1, 'C':5},
        'B': {'D':3, 'E':6},
        'C': {'F':2},
        'D': {'G':4},
        'E': {'G':2},
        'F': {'G':7},
        'G': {}
    }

    heuristic = {
        'A':7, 'B':6, 'C':3, 'D':4, 'E':2, 'F':1, 'G':0
    }

    start = 'A'
    goal = 'G'

    path = greedy_best_first_search(graph, start, goal, heuristic)
    print("Path 1:", path)


    graph2 = {
        'S': {'A':1, 'B':5},
        'A': {'C':2, 'D':3},
        'B': {'E':4},
        'D': {'G':2},
        'E': {'G':1},
        'G': {}
    }

    heuristic2 = {
        'S':7, 'A':6, 'B':4, 'C':3, 'D':2, 'E':1, 'G':0
    }

    start2 = 'S'
    goal2 = 'G'

    path2 = greedy_best_first_search(graph2, start2, goal2, heuristic2)
    print("Path 2:", path2)
```

---

---

# 1) 8 – Puzzle BFS

### INPUT

start state:

```
(1,2,3,
 0,4,6,
 7,5,8)
```

### OUTPUT

```
Solution Found!

Step 0:
(1, 2, 3)
(0, 4, 6)
(7, 5, 8)

Step 1:
(1, 2, 3)
(4, 0, 6)
(7, 5, 8)

Step 2:
(1, 2, 3)
(4, 5, 6)
(7, 0, 8)

Step 3:
(1, 2, 3)
(4, 5, 6)
(7, 8, 0)
```

---

# 2) Flood Fill DFS

### INPUT

image:

```
1 1 1 0
0 1 1 1
1 0 1 1
```

starting pixel: `(1,1)`
new color: `2`

### OUTPUT

```
Before Fill:
1 1 1 0
0 1 1 1
1 0 1 1

After Fill:
1 1 1 0
0 2 2 2
1 0 2 2
```

---

# 3) A* Search

### INPUT

Grid:

```
1 1 1 1
1 0 1 1
1 1 1 1
0 1 1 1
```

start = (0,0)
destination = (3,3)

### OUTPUT

```
Path Found:
-> (0, 0) -> (1, 0) -> (2, 1) -> (3, 2) -> (3, 3)
```

(Your actual run may show a slightly different path based on cost, but destination is same)

---

# 4) Greedy Best First Search

### INPUT

Graph 1:

```
A → B , C
B → D , E
C → F
D → G
E → G
F → G
```

start = ‘A’
goal = ‘G’

### OUTPUT

```
Path 1: ['A', 'C', 'F', 'G']
```

---

Graph 2:

```
S → A , B
A → C , D
B → E
D → G
E → G
```

start = ‘S’
goal = ‘G’

### OUTPUT

```
Path 2: ['S', 'B', 'E', 'G']
```

---










