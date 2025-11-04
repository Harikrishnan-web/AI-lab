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
