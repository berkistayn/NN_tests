# Use the Trainer object to train a supervised NN such that NN's moves are checked by the
# pathfinding algorithm, A*. The NN structure is a FC one, nn_w_range.
# Trainer class essentially brings two classes together: Visualizer and NN. By doing so,
# it is possible to visualize the whole learning process.
# Note that visualizer class is a rough mix of functions which generates a proper Grid object
# and accompanying GUI.

from gridShortestPath.supervised.nn_w_range import NN_w_range
from gridShortestPath.supervised.visualizer import *
from tkinter import X, Y, BOTH, Button, Label
from tkinter.filedialog import askopenfilename

# NOTE: fix labels_dict -> we should get left, right type outputs with it showing the right
# direction.( came_from is in reverse but just inverting is not enough?)
class Trainer(Visualizer):
    def __init__(self, range, rows=20, cols=20, thresh=0.2, allowed_off_steps=4):
        super().__init__()
        self.range = range
        self.NN     = None

        self.rows = rows
        self.cols = cols
        self.thresh = thresh

        self.after_Training = None
        self.after_NN = None

        self.allowed_off_the_path = allowed_off_steps
        self.LSOTP_i = 0

        # generate grid, solve with A*
        self.generateGrid_w_path(rows=rows, cols=cols, thresh=thresh)
        # build standard GUI
        self.buildGUI()
        # add modifications
        self.addButtons()
        # Create NN
        self.create_NN()


    def getLabel(self):
        # It is lengthy; easy to debug, easy to read.
        # We will get a V2 with 1 dictionary for speed...
        if self.GB.game.IsOver is True:
            print('Game is over, starting a new one.')
            self.restart()
        cur = self.GB.game.currentLoc
        if cur in self.path:
            target_index = self.path.index(cur) + 1

            # final step doesnt exist in the self.path
            if target_index != len(self.path):
                change = np.array(self.path[target_index]) - np.array(cur)
                tuple_change = tuple(change.tolist())

                move_dict = {(1, 0): 'down',
                             (-1, 0): 'up',
                             (0, 1): 'right',
                             (0, -1): 'left'}
                move = move_dict[tuple_change]
            else:
                # depending on current location decide on last move;
                if cur[0] == self.rows - 1:
                    move = 'right'
                else:
                    move = 'down'

            label_dict = {'up'    : [1, 0, 0, 0],
                          'down'  : [0, 1, 0, 0],
                          'right' : [0, 0, 1, 0],
                          'left'  : [0, 0, 0, 1]}
            label = np.array(label_dict[move]).reshape(1, 4)
            print('Labeled Move:   ', move)

            return label
        else:
            print('Out of keys ~ labels, creating new grid.')
            return None

    def create_NN(self):
        self.NN = NN_w_range(range=self.range, act_func='elu')

    def NN_move(self):
        if self.after_NN is not None:
            self.after_NN = self.GB.parent.after(100, self.NN_move)
            if self.GB.game.IsOver is True:
                print('NN stopped playing.')
                self.b1.config(state='normal')
                self.b2.config(state='normal')
                self.b4.config(state='normal')
                self.b5.config(state='normal')
                self.b6.config(state='normal')
                self.b7.config(bg='burlywood3')
                self.b7.config(fg='black')
                self.b7.config(text='Let NN play!')

                self.GB.parent.after_cancel(self.after_NN)
                self.after_NN = None

        input_vector = self.NN.generate_input(self.GWW)
        move = self.NN.makeMove(input_vector)
        # visualize the move
        print('NN makes the following move: ', move)
        self.GB.game.play(move)
        self.GB.grid_colorvals = self.GB.game.grid.copy()
        self.GB.show()

    def addButtons(self):
        label = tk.Label(self.GB.parent, text="HOW TO USE?\n\n"
                                              "AIM:\n"
                                              "Reach right-bottom most corner\n\n"
                                              "TO PLAY:\n"
                                              "Press 'w,a,s,d' to move\n\n"
                                              "TO SEE A* ALGORITHM RESULTS:\n"
                                              "Press 'n' to see the best path\n"
                                              "Press 'm' to see searched cells",
                                              bg="black", fg='white')

        self.b1 = Button(self.GB.parent, text="Load weights", bg='burlywood3',
                    command=self.loadWeights)
        self.b2 = Button(self.GB.parent, text="Train", bg='burlywood3',
                    command=self.train_cont)
        self.b4 = Button(self.GB.parent, text="Make a move with NN", bg='burlywood3',
                    command=self.NN_move)
        self.b5 = Button(self.GB.parent, text="Restart with a new grid", bg='burlywood3',
                    command=self.restart)
        self.b6 = Button(self.GB.parent, text="Save weights", bg='burlywood3',
                    command=self.saveWeights)
        self.b7 = Button(self.GB.parent, text="Let NN play!", bg='burlywood3',
                    command=self.NN_move_cont)

        label.pack(side='left', padx=4, pady=4, fill=BOTH)
        label.config(font=("courier"))
        self.b7.pack(fill=X, padx=2, pady=2)
        self.b2.pack(fill=X, padx=2, pady=2)
        self.b4.pack(fill=X, padx=2, pady=2)
        self.b5.pack(fill=X, padx=2, pady=2)
        self.b6.pack(fill=X, padx=2, pady=2)
        self.b1.pack(fill=X, padx=2, pady=2)

    def train_once(self):
        # or continuously depending on following
        if self.after_Training is not None:
            self.after_Training = self.GB.parent.after(100, self.train_once)

        self.NN_move()
        input_vector = self.NN.generate_input(self.GWW)
        label = self.getLabel()
        # train if a label exists, allow some unlabeled moves as well
        if label is None:
            self.LSOTP_i = self.LSOTP_i + 1
        else:
            # epochs is an important hyperparameter
            self.NN.trainOnce(input_vector, label, epochs=3)
        if self.LSOTP_i == self.allowed_off_the_path:
            # if it starts to wonder around;
            self.LSOTP_i = 0
            return self.restart()


    def train_cont(self):
        if self.after_Training is None:
            print('Training till stopped...')
            self.b1.config(state='disabled')
            self.b4.config(state='disabled')
            self.b5.config(state='disabled')
            self.b6.config(state='disabled')
            self.b7.config(state='disabled')
            self.b2.config(bg='green')
            self.b2.config(fg='white')
            self.b2.config(text='PRESS TO STOP')

            self.after_Training = self.GB.parent.after(1, self.train_once)
        else:
            print('Stopped training.')
            self.b1.config(state='normal')
            self.b4.config(state='normal')
            self.b5.config(state='normal')
            self.b6.config(state='normal')
            self.b7.config(state='normal')
            self.b2.config(bg='burlywood3')
            self.b2.config(fg='black')
            self.b2.config(text='Train')

            self.GB.parent.after_cancel(self.after_Training)
            self.after_Training = None

    def NN_move_cont(self):
        if self.after_NN is None:
            print('NN is playing.')
            self.b1.config(state='disabled')
            self.b2.config(state='disabled')
            self.b4.config(state='disabled')
            self.b5.config(state='disabled')
            self.b6.config(state='disabled')
            self.b7.config(bg='green')
            self.b7.config(fg='white')
            self.b7.config(text='PRESS TO STOP')
            self.NN_move()
            self.after_NN = self.GB.parent.after(1, self.NN_move)
        else:
            print('NN stopped playing.')
            self.b1.config(state='normal')
            self.b2.config(state='normal')
            self.b4.config(state='normal')
            self.b5.config(state='normal')
            self.b6.config(state='normal')
            self.b7.config(bg='burlywood3')
            self.b7.config(fg='black')
            self.b7.config(text='Let NN play!')

            self.GB.parent.after_cancel(self.after_NN)
            self.after_NN = None

    def restart(self, random_index_coeff=1.1):
        # following does not work
        self.generateGrid_w_path(self.rows, self.cols, self.thresh)
        if self.after_Training is not None: # if training is on
            random_curLoc_index = np.random.random_integers(0, int(len(self.path) / random_index_coeff), 1)[0]
            self.GWW.currentLoc = self.path[random_curLoc_index].copy()
            self.GWW.grid[0][0] = self.GWW.blocked
            self.GWW.grid[self.GWW.currentLoc[0], self.GWW.currentLoc[1]] = self.GWW.current
        self.GB.game = self.GWW
        self.GB.grid_colorvals = self.GWW.grid.copy()
        self.GB.search_dict_start = self.path
        self.GB.search_dict = self.came_from
        self.GB.show()

    def saveWeights(self):
        self.NN.saveWeights(self.range, self.allowed_off_the_path)

    def loadWeights(self):
        self.NN_move()  # need to initialize tensorflow
        file_name = askopenfilename()
        self.NN.loadWeights(file_name)


# Create Trainer
T = Trainer(range=4, rows=50, cols=50, thresh=0.3, allowed_off_steps=3)
# final step;
T.startGUI()


