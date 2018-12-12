import numpy as np
import tkinter as tk
import heapq
import time

# Has A* pathfinding algorithm with GUI in Visualizer.

class GameBoard(tk.Frame):
    def __init__(self, parent, rows=10, columns=10, threshold=0.2,
                 grid_val=[1, 2, 3, 4], search_dict=None, search_dict_start=None):
        '''size is the size of a square, in pixels'''

        self.rows = rows
        self.columns = columns

        self.canvas_width  = 750
        self.canvas_height = 750
        # so that modifications can be made on GUI
        self.parent = parent
        # so that animations with pauses can be created
        self.after_MBS = None
        self.mbss_i = 0

        # parameters for A* algorithm
        self.searchLoc = [rows-1, columns-1]
        self.search_dict = search_dict
        self.search_dict_start = search_dict_start
        self.color_val_best_path     = - 3
        self.color_val_searched_path = -2

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.canvas_width, height=self.canvas_height,
                                background="black")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        #self.canvas.bind("<1>", lambda event: self.canvas.focus_set())
        self.canvas.focus_set()
        self.canvas.bind("<Key>", self.takeUserInput)

        self.game = Grid_Game(width=rows, height=columns, threshold=threshold,
                              grid_val=grid_val)  # initiate Grid
        self.color_vals = [self.game.unblocked, self.game.blocked,
                           self.game.tail, self.game.current]
        self.grid_colorvals = self.game.grid.copy()
        self.show()
        # print("\n\nUse 'w,a,s,d' to play.\n"
        #       "Use 'n' to see optimum path.\n"
        #       "Use 'm' to see searched cells.\n")

    def show(self):
        # stop taking keyboard inputs from user if game is over.
        # if self.grid.IsOver:
        #     self.canvas.unbind("<Key>")

        xsize = int(self.canvas_width  / self.columns)
        ysize = int(self.canvas_height  / self.rows)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        #self.data = self.grid.grid
        color_dict = {self.color_vals[0]: 'white',
                      self.color_vals[1]: 'grey',
                      self.color_vals[2]: 'red',
                      self.color_vals[3]: 'blue',
                      self.color_val_best_path: 'gold',
                      self.color_val_searched_path: 'lightsteelblue1',
                      }
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                color = color_dict[self.grid_colorvals[row][col]]
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black",
                                             fill=color, tags="square")
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

    def takeUserInput(self, event):
        char = event.char
        if char == 'd':
            self.game.play('right')
            self.grid_colorvals = self.game.grid.copy()
        elif char == 'a':
            self.game.play('left')
            self.grid_colorvals = self.game.grid.copy()
        elif char == 'w':
            self.game.play('up')
            self.grid_colorvals = self.game.grid.copy()
        elif char == 's':
            self.game.play('down')
            self.grid_colorvals = self.game.grid.copy()
        elif char == 'm':
            self.showSearchedCells()
        elif char == 'n':
            self.mbss_i = 0
            self.moveBySearch_start()
        self.show()

    def showSearchedCells(self):
        for cell in self.search_dict:
            x,y = cell
            #if x < self.grid.w and y < self.grid.h:
            if self.grid_colorvals[x][y] != self.color_val_best_path:
                self.grid_colorvals[x][y] = self.color_val_searched_path
            # paint current cell in case its painted over
            x, y = self.game.currentLoc
            self.grid_colorvals[x][y] = self.color_vals[3]
            self.show()

        # k = tuple(self.searchLoc)
        # self.grid.grid[self.searchLoc[0]][self.searchLoc[1]] = self.colorVal_best_path # it paints last point once more
        # # so that first point is painted.
        # nextLoc = self.search_dict[k]
        # if nextLoc is None:
        #     print('Thats it, best path!')
        #     for cell in self.search_dict:
        #         x,y = cell
        #         if x < self.grid.w and y < self.grid.h:
        #             if self.grid.grid[x][y] != self.colorVal_best_path:
        #                 self.grid.grid[x][y] = self.colorVal_searched_path
        #     self.parent.after_cancel(self.after_MBS)
        #     self.after_MBS = None
        #     self.show()
        # else:
        #     self.searchLoc = nextLoc
        #     self.grid.grid[nextLoc[0]][nextLoc[1]] = self.colorVal_best_path
        #     # self.after_MBS = self.parent.after(100, self.moveBySearch)
        #     self.after_MBS = self.parent.after(5, self.moveBySearch)
        # # self.show()

    def moveBySearch_start(self, pause_time=70):
        if self.mbss_i != len(self.search_dict_start):
            x, y = self.search_dict_start[self.mbss_i]
            self.grid_colorvals[x][y] = self.color_val_best_path
            # update ~ recall this function again;
            self.after_MBS = self.parent.after(pause_time, self.moveBySearch_start)
            self.mbss_i = self.mbss_i + 1
        else:
            # paint the last ~ target cell;
            self.grid_colorvals[self.rows - 1][self.columns - 1] = self.color_val_best_path
            # stop recalling this func;
            self.parent.after_cancel(self.after_MBS)
            self.after_MBS = None
            # paint current cell in case its painted over
            x, y = self.game.currentLoc
            self.grid_colorvals[x][y] = self.color_vals[3]
        self.show()

class Grid_Game:
    def __init__(self, width=10, height=10, threshold=0.25, grid_val=[1, 2, 3, 4],
                 badmove_limit=15):
        self.w = width
        self.h = height
        self.threshold = threshold  # percentage amount of blocks

        self.grid = np.array([])
        self.grid_padded = []
        self.currentLoc = [0, 0]
        self.hasWon = False
        self.IsOver = False
        self.badmoveLimit = badmove_limit

        # grid values
        self.unblocked = grid_val[0]
        self.blocked = grid_val[1]
        self.tail = grid_val[2]
        self.current = grid_val[3]

        # performance metrics
        self.no_recurring_badmove = 0
        self.no_total_badmove = 0
        self.no_move = 0
        self.final_distance = 0
        self.grid_size = self.w + self.h - 2

        # A_start parameters

        self.createBinaryGrid()

    def createBinaryGrid(self):
        generated = np.random.randint(0, 100, [self.w, self.h])
        # apply threshold
        np.place(generated, generated < self.threshold * 100, self.blocked)
        np.place(generated, generated >= self.threshold * 100, self.unblocked)
        self.grid = generated

        # give it some additional room: clearance around start/end region
        self.grid[0][1] = self.unblocked
        self.grid[1][0] = self.unblocked
        self.grid[self.w - 2][self.h - 1] = self.unblocked
        self.grid[self.w - 1][self.h - 2] = self.unblocked
        # start
        self.grid[0][0] = self.current
        self.grid[self.w - 1][self.h - 1] = self.unblocked
        # use a small trick to make checks easier
        self.grid_padded = np.pad(self.grid, 1, 'constant', constant_values=self.blocked)

    def checkGameState(self):
        x, y = self.currentLoc
        # check win
        if self.grid[self.w - 1][self.h - 1] == self.current:
            self.hasWon = True
            self.IsOver = True

        # check if stuck
        check1 = 1 if self.grid_padded[x + 1][y + 2] in (self.blocked, self.tail) else 0
        check2 = 1 if self.grid_padded[x + 1][y] in (self.blocked, self.tail) else 0
        check3 = 1 if self.grid_padded[x + 2][y + 1] in (self.blocked, self.tail) else 0
        check4 = 1 if self.grid_padded[x][y + 1] in (self.blocked, self.tail) else 0
        neighbours = check1 * check2 * check3 * check4
        if neighbours != 0:  # cant move anymore
            self.IsOver = True
            print('You got stuck!')

        # check stupidity
        if self.no_recurring_badmove >= self.badmoveLimit:
            self.IsOver = True
            print('\nTOO much bad moves...Cant stand anymore!')

        if self.IsOver is True:
            self.displayResults()

    def displayResults(self):
        print('\n\nGame is over.')
        self.final_distance = (self.w - self.currentLoc[0]) + (self.h - self.currentLoc[1]) - 2

        if self.hasWon is True:
            print('You have WON!')
        else:
            print('You have LOST!')
        print('number of moves:', self.no_move)
        print('number of bad moves: ', self.no_total_badmove)
        print('distance: ', self.final_distance)
        print('grid size: ', self.grid_size)
        print('\nScore(lower the better): ', self.score())

    def updateGrid(self, move):
        if move is None:
            return
        self.no_move += 1
        move2del_x = {
            'right': 1,
            'left': -1,
            'up': 0,
            'down': 0
        }
        move2del_y = {
            'right': 0,
            'left': 0,
            'up': -1,
            'down': 1
        }
        del_x = move2del_x[move]
        del_y = move2del_y[move]
        x = self.currentLoc[1] + del_x
        y = self.currentLoc[0] + del_y
        # check if the move is valid
        if self.grid_padded[y + 1][x + 1] == self.blocked:
            self.no_total_badmove += 1
            self.no_recurring_badmove += 1
            print('Hit wall!')
        elif self.grid_padded[y + 1][x + 1] == self.tail:
            self.no_total_badmove += 1
            self.no_recurring_badmove += 1
            print('Hit your tail!')
        else:
            self.no_recurring_badmove = 0
            self.grid[self.currentLoc[0]][self.currentLoc[1]] = self.tail
            self.grid_padded[self.currentLoc[0] + 1][self.currentLoc[1] + 1] = self.tail
            self.currentLoc[1] = x
            self.currentLoc[0] = y
            self.grid[self.currentLoc[0]][self.currentLoc[1]] = self.current
            self.grid_padded[self.currentLoc[0] + 1][self.currentLoc[1] + 1] = self.current
        self.checkGameState()

    def play(self, move):
        self.checkGameState()
        if self.IsOver is False:
            self.updateGrid(move)

    def score(self):
        return abs(self.no_move - self.grid_size + 1) * (self.final_distance + 1) ** 2

    def passable(self, cell):
        x, y = cell
        return self.grid_padded[x + 1, y + 1] not in [self.tail, self.blocked]

    def neighbors(self, curr):
        # with reference to grid_padded
        x, y = curr
        neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neighbours = filter(self.passable, neighbours)

        return neighbours

# ====

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class GridWithWeights(Grid_Game):
    def __init__(self, width=10, height=10, threshold=0.25, grid_val=[1, 2, 3, 4]):
        super().__init__(width=width, height=height, threshold=threshold, grid_val=grid_val)
        self.weights = {}
        self.createWeights()

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

    def createWeights(self):
        # get grid values
        r, c = self.grid.shape
        for row in range(0, r):
            for col in range(0, c):
                self.weights[(row, col)] = 1
                # self.weights[(row, col)] = np.random.normal(1, 0.001, 1)

# ====

class Visualizer:
    def __init__(self):
        self.GWW = None

        self.came_from = None
        self.path = None
        self.GB = None

    def heuristic(self, target, curr):
        (x1, y1) = target
        (x2, y2) = curr
        h = abs(x1 - x2) + abs(y1 - y2)
        dx1 = abs(x1 - x2)
        dy1 = abs(y1 - y2)
        cross = abs(dx1 * y1 - x1 * dy1)
        h += cross * 0.001
        return h

    def a_star_search(self, grid_ww, start, goal):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next in grid_ww.neighbors(current):
                new_cost = cost_so_far[current] + grid_ww.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        return came_from, cost_so_far

    def getBestPath(self, grid, search_dict, target):
        # A* has a soln, however it starts at end and travels back to start.
        # To get a nice path visual we need to go from start to target.
        # So invert the order of best path.
        best = []
        searched = []
        searchLoc = target
        nextLoc = [0, 0]
        while nextLoc is not None:
            k = tuple(searchLoc)
            # best.append(grid.grid[searchLoc[0]][searchLoc[1]])
            nextLoc = search_dict[k]
            if nextLoc is None:
                for cell in search_dict:
                    x, y = cell
                    #if x < grid.w and y < grid.h:
                    if [[x], [y]] not in best:
                        searched.append([[x], [y]])
            else:
                searchLoc = nextLoc
                best.append([nextLoc[0], nextLoc[1]])

            best_in_order = best[::-1] # reverses the order of elements

        # noinspection PyUnboundLocalVariable
        return best_in_order, searched

    def generateGrid_w_path(self, rows=10, cols=10, thresh=0.1):
        print('Creating a grid.')
        target = (rows - 1, cols - 1)
        # make sure generated grid has a connection between start and target.
        grid_connected = False
        while grid_connected is False:
            self.GWW = GridWithWeights(width=rows, height=cols, threshold=thresh)
            start_time = time.time()
            self.came_from, cost_so_far = self.a_star_search(self.GWW, (0, 0), target)
            self.a_star_elapsedTime = time.time() - start_time
            grid_connected = True if target in self.came_from else False
            print('...')


        self.path, _ = self.getBestPath(self.GWW, self.came_from, target)
        print('A* performance:')
        print('number of searched cells:    ', len(self.came_from))
        print('number of steps in best path:    ', len(self.path))
        print('solution took:  ', self.a_star_elapsedTime, ' seconds.')

    def buildGUI(self):
        root = tk.Tk()
        root.title('Grid Pathfinding Learning ~ Visualized')
        self.GB = GameBoard(root, rows=self.GWW.w, columns=self.GWW.h,
                            search_dict=self.came_from, search_dict_start=self.path)
        self.GB.game = self.GWW
        self.GB.grid_colorvals = self.GWW.grid.copy()
        self.GB.pack(side="top", fill="both", expand="true", padx=4, pady=4)
        self.GB.show()

    def startGUI(self):
        # call startGUI after all modifications
        self.GB.parent.mainloop()


