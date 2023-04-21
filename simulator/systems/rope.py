"""Physics engine for rope system

Credit to https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import os
import cv2
import pymunk
import matplotlib.pyplot as plt

from .base import *
from tqdm import tqdm
from pymunk.vec2d import Vec2d
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


class RopeEngine(Engine):

    def __init__(self, params):
        self.params = params
        self.radius = params.radius
        self.mass = params.mass
        self.n_ball = params.n_bodies
        self.k = params.spring_constant
        self.gravity = params.gravity
        self.damping = self.k*params.damping_factor
        self.position_range = params.position_range
        self.compression_range = params.compression_range

        super(RopeEngine, self).__init__(params.dt, params.state_dim, params.param_dim)

    def init(self):
        self.init_x = 0
        self.param = np.array([self.n_ball, self.init_x, self.k, self.damping, self.gravity])

        self.space = pymunk.Space()
        self.space.gravity = (0., self.gravity)

        self.height = self.params.init_height
        self.rest_len = self.params.rest_length

        self.add_masses()
        self.add_rels()

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
                move_joint = pymunk.PinJoint(self.space.static_body, body, (x, y), (0, 0))
                self.space.add(body, shape, move_joint)
            else:
                self.space.add(body, shape)

            self.balls.append(body)

            # sample angle btwn -60 and 60 degrees
            # factor = np.random.uniform(*self.compression_range)
            # factor = 2-factor if np.random.rand() > 0.5 else factor
            theta = np.random.beta(a=0.5, b=0.5)*(np.pi/2)
            x -= self.rest_len*np.sin(theta)
            y -= self.rest_len*np.cos(theta)

    def add_rels(self):
        # add springs over adjacent balls
        for i in range(self.n_ball-1):
            c = pymunk.DampedSpring(
                self.balls[i], self.balls[i + 1], (0, 0), (0, 0),
                rest_length=self.rest_len, stiffness=self.k, damping=self.damping)
            self.space.add(c)

    def get_param(self):
        return self.n_ball, self.init_x, self.k, self.damping, self.gravity

    def get_state(self):
        state = np.zeros((self.n_ball, self.state_dim))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([
                ball.position[0], ball.position[1], 
                ball.velocity[0], ball.velocity[1],
                0, 0  # will be computed later    
            ])
        return state

    def step(self):
        self.space.step(self.dt)

    def render(
        self, states, video=True, image=False, path=None,
        draw_edge=True, lim=(-2.5, 2.5, -2.5, 2.5), states_gt=None, count_down=False
    ):
        if video:
            video_size = (1280, 960)
            video_path = f"{path}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, video_size)

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']
        time_step, n_ball, _ = states.shape
        for i in tqdm(range(time_step), desc="Rendering trajectory..."):
            fig, ax = plt.subplots(1, dpi=100)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            if draw_edge:
                for x in range(n_ball-1):
                    plt.plot([states[i, x, 0], states[i, x + 1, 0]],
                             [states[i, x, 1], states[i, x + 1, 1]],
                             '-', color=c[1], lw=2, alpha=0.5)

            if states_gt is not None:
                circles = []
                circles_color = []
                for j in range(n_ball):
                    circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=self.radius * 5 / 4)
                    circles.append(circle)
                    circles_color.append(c[1])
                pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=0.25)
                ax.add_collection(pc)

            circles = []
            circles_color = []
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius * 5 / 4)
                circles.append(circle)
                circles_color.append(c[0])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
            ax.add_collection(pc)
            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-2.5, 1.5, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
                out.write(frame)
                if i == time_step-1:
                    for _ in range(5):
                        out.write(frame)
            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')
            plt.close()

        if video:
            out.release()


