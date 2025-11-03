import tkinter as tk
import random

GRID_SIZE = 4
CELL_SIZE = 100

class WumpusWorld:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.pack()
        self.agent_pos = [0, 0]
        self.gold_pos = [random.randint(0, 3), random.randint(0, 3)]
        self.wumpus_pos = [random.randint(1, 3), random.randint(1, 3)]
        self.pits = [[random.randint(1, 3), random.randint(1, 3)] for _ in range(3)]

        # Avoid overlap with agent
        if self.gold_pos == self.agent_pos:
            self.gold_pos = [3, 3]
        if self.wumpus_pos == self.agent_pos:
            self.wumpus_pos = [2, 2]
        self.pits = [pit for pit in self.pits if pit != self.agent_pos]

        self.draw_grid()
        self.master.bind("<Key>", self.move_agent)

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")

                if [i, j] == self.agent_pos:
                    self.canvas.create_text(x0 + 50, y0 + 50, text="A", font=("Arial", 30))
                elif [i, j] == self.gold_pos:
                    self.canvas.create_text(x0 + 50, y0 + 50, text="G", font=("Arial", 30))
                elif [i, j] == self.wumpus_pos:
                    self.canvas.create_text(x0 + 50, y0 + 50, text="W", font=("Arial", 30))
                elif [i, j] in self.pits:
                    self.canvas.create_text(x0 + 50, y0 + 50, text="P", font=("Arial", 25))

    def move_agent(self, event):
        key = event.keysym
        i, j = self.agent_pos
        if key == "Up" and i > 0:
            self.agent_pos[0] -= 1
        elif key == "Down" and i < GRID_SIZE - 1:
            self.agent_pos[0] += 1
        elif key == "Left" and j > 0:
            self.agent_pos[1] -= 1
        elif key == "Right" and j < GRID_SIZE - 1:
            self.agent_pos[1] += 1
        self.draw_grid()
        self.check_status()

    def check_status(self):
        if self.agent_pos == self.gold_pos:
            self.show_message("You found the gold! You win!")
        elif self.agent_pos == self.wumpus_pos:
            self.show_message("You were eaten by the Wumpus!")
        elif self.agent_pos in self.pits:
            self.show_message("You fell into a pit!")

    def show_message(self, message):
        self.canvas.create_text(
            GRID_SIZE * CELL_SIZE / 2,
            GRID_SIZE * CELL_SIZE / 2,
            text=message,
            font=("Arial", 24),
            fill="red"
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Wumpus World Visualization")
    game = WumpusWorld(root)
    root.mainloop()
