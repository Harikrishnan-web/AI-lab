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
