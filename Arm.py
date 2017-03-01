import numpy as np
import matplotlib.pyplot as plt
import time

import theano
from theano import tensor as tns


class Arm(object):
    
    def __init__(self, lengths):
        
        self.lengths = np.array(lengths) # lengths of the arm segments
        self.n = len(self.lengths) # number of movable joints
        
        self.lastaction = np.zeros_like(lengths)
        
        self.friction = 0.10 # resistance: determines how fast speed decays
        self.inertia = 0.01 # unresponsiveness: the effect of actions on speed
        
        self.reset()
        self.compile_dynamics()
    
    def reset(self):
        """ Reset the angles and the angle velocities of the arm. """
        
        angles = (2*np.pi) * np.random.random(size=self.n)
        velocities = np.zeros(self.n)
        placeholder = np.nan * np.zeros(2) # dummy tip position
        
        # first set the angles and velocities to their correct values:
        self.state = np.concatenate([angles, velocities, placeholder])
        
        # now that the angles have been set, recompute the tip position:
        self.state[-2:] = self.x[-1], self.y[-1]
    
    def DYNAMICS(self, STATE, ACTION):
        
        OLD_ANGLES = STATE[0 : self.n]
        OLD_VELOCITY = STATE[self.n : -2]
    
        FRICTIONLESS = self.inertia*OLD_VELOCITY + (1 - self.inertia)*ACTION
        NEW_VELOCITY = (1 - self.friction) * FRICTIONLESS

        # NEW_ANGLES = OLD_ANGLES + NEW_VELOCITY
        NEW_ANGLES = OLD_ANGLES + OLD_VELOCITY
        
        ABSOLUTE_ANGLES = tns.cumsum(NEW_ANGLES)
        
        X = tns.sum(self.lengths * np.cos(ABSOLUTE_ANGLES))
        Y = tns.sum(self.lengths * np.sin(ABSOLUTE_ANGLES))

        return tns.concatenate([NEW_ANGLES, NEW_VELOCITY, [X, Y]])

    def compile_dynamics(self):

        S = tns.dvector("S")
        U = tns.dvector("U")
    
        F = self.DYNAMICS(S, U)
    
        Fs = theano.gradient.jacobian(F, wrt=S)
        Fu = theano.gradient.jacobian(F, wrt=U)
    
        F_PARAMS = [F, Fs, Fu]
    
        print("Compiling dynamics . . .")
        self.dynamics = theano.function(inputs=[S, U], outputs=F)
        self.dynamics_params = theano.function(inputs=[S, U], outputs=F_PARAMS)
        print("Done.\n")

    @property
    def angles(self):
        
        return self.state[0 : self.n]

    @property
    def x(self):
        
        absolute_angles = np.cumsum(self.angles)
        relative_positions = self.lengths * np.cos(absolute_angles)
        absolute_positions = np.cumsum(relative_positions)

        return np.concatenate([[0.0], absolute_positions])

    @property
    def y(self):
        
        absolute_angles = np.cumsum(self.angles)
        relative_positions = self.lengths * np.sin(absolute_angles)
        absolute_positions = np.cumsum(relative_positions)

        return np.concatenate([[0.0], absolute_positions])

    @property
    def tipx(self):
        
        return self.state[-2]

    @property
    def tipy(self):
        
        return self.state[-1]

    def move(self, action):
        
        self.lastaction = action
        self.state = self.dynamics(self.state, action)


class Box(object):
    
    def __init__(self, low, high):
        
        self.low = low
        self.high = high
        
    @property
    def shape(self):
        
        return self.low.shape
    
    def sample(self):
        
        return self.low + (self.high - self.low)*np.random.random(size=self.shape)


class ReachingGame(object):
    
    def __init__(self, n=3, lengths=None, figsize=(10, 10)):
        
        if n is None and lengths is None:
            n = 3
        
        if lengths is None:
            lengths = np.ones(n)
        
        if n is None:
            n = len(lengths)

        # parameter initialization
        self.arm = Arm(lengths)
        self.reset_goal()
        
        stateones = np.ones(2*self.arm.n + 4)
        actionones = np.ones(self.arm.n)
        
        self.observation_space = Box(-np.inf*stateones, np.inf*stateones)
        self.action_space = Box(-0.5*actionones, 0.5*actionones)
        
        # loss function parameters
        self.threshold = 0.1
        self.sharpness = 5.0
        self.regweight = 0.9
        self.offset = self.threshold**2 * (self.sharpness - 1.0)

        # create and compile loss expression in Theano
        self.compile_loss()

        # some plotting-relevant parameters
        self.figsize = figsize
        self.lastreward = 0
        self.isvisible = False

    def reset(self):
        
        self.arm.reset()
        self.reset_goal()
        
        return self.observe()
    
    def reset_goal(self):
        
        angle = (2*np.pi) * np.random.random()
        radius = np.random.random()

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        self.set_goal(x, y)
    
    def step(self, action):
        
        self.arm.move(action)
        
        state = self.observe()
        self.lastreward = -self.loss(self.goal, self.arm.state, action)
        
        return state, self.lastreward, False, {}
    
    def compile_loss(self):
        
        S = tns.dvector("STATE")
        U = tns.dvector("ACTION")
        GOAL = tns.dvector("[X*, Y*]")
        
        # loss, part 1: distance-based loss
    
        TIP = S[-2:]
        DIST2 = tns.sum((GOAL - TIP)**2)
        
        SHARP = self.sharpness*DIST2
        BLUNT = DIST2 + self.offset
        SMALL = (DIST2 < self.threshold**2)
        
        PROPER_LOSS = theano.ifelse.ifelse(SMALL, SHARP, BLUNT)
    
        # loss, part 2: action-based loss
    
        REGULARIZER = tns.sum(U**2)
    
        # part 1 + part 2
    
        L = PROPER_LOSS + self.regweight*REGULARIZER

        Ls = theano.grad(L, wrt=S)
        Lu = theano.grad(L, wrt=U)

        Lss = theano.gradient.jacobian(Ls, wrt=S)
        Lus = theano.gradient.jacobian(Lu, wrt=S, disconnected_inputs='ignore')
        Luu = theano.gradient.jacobian(Lu, wrt=U, disconnected_inputs='ignore')
    
        LOSS_PARAMS = [L, Ls, Lu, Lss, Lus, Luu]
    
        print("Compiling loss . . .")
        self.loss = theano.function(inputs=[GOAL, S, U], outputs=L)
        self.loss_params = theano.function(inputs=[GOAL, S, U], outputs=LOSS_PARAMS)
        print("Done.\n")

    def makeplot(self):
        """ Initialize the plot of the arm and goal. """
        
        plt.ion() # don't stop and wait after plotting

        # Plotting
        self.figure, self.axes = plt.subplots(figsize=self.figsize)
        
        armlength = np.sum(self.arm.lengths)
        windowsize = [-1.1*armlength, 1.1*armlength]

        plt.xlim(windowsize)
        plt.ylim(windowsize)

        self.armlines, = self.axes.plot(self.arm.x, self.arm.y, 'bo-', ms=10, lw=5, alpha=0.5)
        self.dot, = self.axes.plot([self.goalx], [self.goaly], 'ro', ms=15, alpha=0.5)
        self.losstext = self.axes.text(-armlength, 0.95*armlength, "", fontsize=18)

        self.axes.set_aspect('equal')
        
        self.isvisible = True
        plt.show()
    
    def close(self):
        """ Close the plot of the arm and goal. """
        
        self.reset()
        
        if self.isvisible:
            plt.close(self.figure)

    @property
    def goal(self):
        
        return np.array([self.goalx, self.goaly])
    
    def set_goal(self, x, y):
        
        self.goalx = x
        self.goaly = y
        
    def update(self):

        self.armlines.set_xdata(self.arm.x)
        self.armlines.set_ydata(self.arm.y)

        self.dot.set_xdata([self.goalx])
        self.dot.set_ydata([self.goaly])
        
        self.losstext.set_text("reward = %.2f" % self.lastreward)
        
        if self.isvisible:
            self.figure.canvas.draw_idle()
            plt.pause(1e-8) # matplotlib witchcraft
    
    def render(self):
        
        if not self.isvisible:
            self.makeplot()
        
        self.update()
    
    def observe(self):
        
        return np.concatenate([self.arm.state, self.goal])


if __name__ == '__main__':

    game = ReachingGame(lengths=np.ones(7))
    
    import time

    for i in range(320):

        forces = game.action_space.sample()

        game.step(forces)
        game.render()
        time.sleep(1./25)

