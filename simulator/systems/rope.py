"""Code for rope system
Credit to https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import os
import cv2
import pymunk
import matplotlib.pyplot as plt

from .base import *
from pymunk.vec2d import Vec2d
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


class RopeEngine(Engine):

    def __init__(self, dt, state_dim, action_dim, param_dim,
                 num_mass_range=[4, 8], k_range=[500., 1500.], gravity_range=[-2., -8.],
                 position_range=[-0.6, 0.6], bihop=True):

        # state_dim = 4
        # action_dim = 1
        # param_dim = 5
        # param [n_ball, init_x, k, damping, gravity]

        self.radius = 0.06
        self.mass = 1.

        self.num_mass_range = num_mass_range
        self.k_range = k_range
        self.gravity_range = gravity_range
        self.position_range = position_range

        self.bihop = bihop

        super(RopeEngine, self).__init__(dt, state_dim, action_dim, param_dim)

    def init(self, param=None):
        if param is None:
            self.n_ball, self.init_x, self.k, self.damping, self.gravity = [None] * 5
        else:
            self.n_ball, self.init_x, self.k, self.damping, self.gravity = param
            self.n_ball = int(self.n_ball)

        num_mass_range = self.num_mass_range
        position_range = self.position_range
        if self.n_ball is None:
            self.n_ball = rand_int(num_mass_range[0], num_mass_range[1])
        if self.init_x is None:
            self.init_x = np.random.rand() * (position_range[1] - position_range[0]) + position_range[0]
        if self.k is None:
            self.k = rand_float(self.k_range[0], self.k_range[1])
        if self.damping is None:
            self.damping = self.k / 20.
        if self.gravity is None:
            self.gravity = rand_float(self.gravity_range[0], self.gravity_range[1])
        self.param = np.array([self.n_ball, self.init_x, self.k, self.damping, self.gravity])

        # print('Env Rope param: n_ball=%d, init_x=%.4f, k=%.4f, damping=%.4f, gravity=%.4f' % (
        #     self.n_ball, self.init_x, self.k, self.damping, self.gravity))

        self.space = pymunk.Space()
        self.space.gravity = (0., self.gravity)

        self.height = 1.0
        self.rest_len = 0.3

        self.add_masses()
        self.add_rels()

        self.state_prv = None

    @property
    def num_obj(self):
        return self.n_ball

    def add_masses(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        x = self.init_x
        y = self.height
        self.balls = []

        for i in range(self.n_ball):
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d(x, y)
            shape = pymunk.Circle(body, self.radius, (0, 0))

            if i == 0:
                # fix the first mass to a specific height
                move_joint = pymunk.GrooveJoint(self.space.static_body, body, (-2, y), (2, y), (0, 0))
                self.space.add(body, shape, move_joint)
            else:
                self.space.add(body, shape)

            self.balls.append(body)
            y -= self.rest_len

    def add_rels(self):
        give = 1. + 0.075
        # add springs over adjacent balls
        for i in range(self.n_ball - 1):
            c = pymunk.DampedSpring(
                self.balls[i], self.balls[i + 1], (0, 0), (0, 0),
                rest_length=self.rest_len * give, stiffness=self.k, damping=self.damping)
            self.space.add(c)

        # add bihop springs
        if self.bihop:
            for i in range(self.n_ball - 2):
                c = pymunk.DampedSpring(
                    self.balls[i], self.balls[i + 2], (0, 0), (0, 0),
                    rest_length=self.rest_len * give * 2, stiffness=self.k * 0.5, damping=self.damping)
                self.space.add(c)

    def add_impulse(self):
        impulse = (self.action[0], 0)
        self.balls[0].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def get_param(self):
        return self.n_ball, self.init_x, self.k, self.damping, self.gravity

    def get_state(self):
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    def step(self):
        self.add_impulse()
        self.state_prv = self.get_state()
        self.space.step(self.dt)

    def render(
        self, states, actions=None, video=True, image=False, path=None,
        draw_edge=True, lim=(-2.5, 2.5, -2.5, 2.5), states_gt=None, count_down=False
    ):
        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (640, 480))

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        if actions is not None and actions.ndim == 3:
            '''get the first ball'''
            actions = actions[:, 0, :]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            if video:
                canvas = FigureCanvasAgg(fig)

            if draw_edge:
                for x in range(n_ball - 1):
                    plt.plot([states[i, x, 0], states[i, x + 1, 0]],
                             [states[i, x, 1], states[i, x + 1, 1]],
                             '-', color=c[1], lw=2, alpha=0.5)

            circles = []
            circles_color = []
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius * 5 / 4)
                circles.append(circle)
                circles_color.append(c[0])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
            ax.add_collection(pc)

            if states_gt is not None:
                circles = []
                circles_color = []
                for j in range(n_ball):
                    circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=self.radius * 5 / 4)
                    circles.append(circle)
                    circles_color.append('orangered')
                pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
                ax.add_collection(pc)

            if actions is not None:
                F = actions[i, 0] / 4
                normF = norm(F)
                if normF < 1e-10:
                    pass
                else:
                    ax.arrow(states[i, 0, 0] + F / normF * 0.1, states[i, 0, 1],
                             F, 0., fc='Orange', ec='Orange', width=0.04, head_width=0.2, head_length=0.2)

            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-2.5, 1.5, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                canvas.draw()
                # frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                # frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = np.asarray(canvas.buffer_rgba())[..., :-1]
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(5):
                        out.write(frame)
            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')
            plt.close()

        if video:
            out.release()


