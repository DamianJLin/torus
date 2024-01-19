import itertools
import matplotlib.pyplot as plt
import numpy as np

# Constants
n_tiles = 4  # Number of times to apply developing map.


# Mathematical helper funtions.

def angle(p, q):
    return np.arctan2(q[1][0] - p[1][0], q[0][0] - p[0][0])


def eucl_length(p, q):
    diff = p - q
    return np.sqrt(
        np.dot(diff.T, diff)
    )


class Quadrilteral:
    """
    Quadrilteral class with methods that apply the transition map.
    """

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def transition(self, perm):
        """
        Compute the quadrangle to the right of the quadrangle
        c---d
        |   |
        a---b.

        If perm is the identity permutation, it does this by feeding a, b, c, d into the variables
        x, y, z, w. A nontrivial permutation can be applied to have this function compute trandition
        maps in other directions. This must of course be inverted at the end for the new
        quadrilateral to have a consistent vertex labelling.
        """

        verts = self.a, self.b, self.c, self.d
        x = verts[perm[0]]
        y = verts[perm[1]]
        z = verts[perm[2]]
        w = verts[perm[3]]

        theta_0 = angle(x, z)
        theta_1 = angle(y, w)
        theta = theta_1 - theta_0
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        rho_0 = eucl_length(x, z)
        rho_1 = eucl_length(y, w)
        rho = rho_1 / rho_0
        scaling = np.eye(2) * rho

        t = y - x

        linear = rotation @ scaling
        affine = t

        verts = list(map(
            lambda v: linear @ (v - x) + x + affine,
            [x, y, z, w]
        ))

        pinv = list(sorted(range(len(perm)), key=perm.__getitem__))
        a = verts[pinv[0]]
        b = verts[pinv[1]]
        c = verts[pinv[2]]
        d = verts[pinv[3]]

        return Quadrilteral(a, b, c, d)

    def transition_right(self):
        return self.transition(perm=(0, 1, 2, 3))

    def transition_up(self):
        return self.transition(perm=(0, 2, 1, 3))

    def transition_left(self):
        return self.transition(perm=(1, 0, 3, 2))

    def transition_down(self):
        return self.transition(perm=(2, 0, 3, 1))

    def points(self):
        return self.a, self.b, self.c, self.d


# Create the graphics with matplotlib.

# Quadrileteral points must be in the orientation:
# c---d
# |   |
# a---b

print(
    "Input points of quadrilateral in the format: x y.\n"
    "Quadrileteral points must be in the orientation:\n"
    "c---d\n"
    "|   |\n"
    "a---b"
)


def get_pos(display):
    pos = None
    try:
        print(f'{display}: ', end='')
        pos = tuple(map(lambda x: float(x), input().split(' ')))
        if len(pos) != 2:
            raise ValueError('More than two space-separated strings detected.')
    except ValueError:
        print("Error: Could not parse input. Please enter two space-separated floats.")
        exit(0)
    if pos is not None:
        return pos
    else:
        print("An error occured.")
        exit(1)


a = get_pos('a')
b = get_pos('b')
c = get_pos('c')
d = get_pos('d')

# Make into 2D ndarrays.
a, b, c, d = list(map(
    lambda x: np.array([[x[0]], [x[1]]]),
    (a, b, c, d)
))

fig, ax = plt.subplots(figsize=(24, 10))
ax.set_aspect('equal')
ax.set_xlim(-20, 20)
ax.set_ylim(-10, 10)


def plot_line(p, q, **kwargs):
    ax.plot(
        (p[0][0], q[0][0]),
        (p[1][0], q[1][0]),
        **kwargs
    )


def plot_quad(quad, **kwargs):
    a, b, c, d = quad.points()
    plot_line(a, b, **kwargs)
    plot_line(b, d, **kwargs)
    plot_line(d, c, **kwargs)
    plot_line(c, a, **kwargs)


def print_points(quad):
    a, b, c, d = quad.points()
    print(
        f'({a[0][0]}, {a[1][0]}), ({b[0][0]}, {b[1][0]}), '
        f'({c[0][0]}, {c[1][0]}), ({d[0][0]}, {d[1][0]})'
    )


def concat(n, func, arg, *args, **kwargs):
    result = arg
    for i in range(n):
        result = func(result, *args, **kwargs)
    return result


def transition_by(quad, i, j):
    if i > 0:
        transition_x = Quadrilteral.transition_right
    elif i < 0:
        transition_x = Quadrilteral.transition_left
    elif i == 0:
        def transition_x(_): return _

    if j > 0:
        transition_y = Quadrilteral.transition_up
    elif j < 0:
        transition_y = Quadrilteral.transition_down
    elif j == 0:
        def transition_y(_): return _

    return concat(
        abs(j),  transition_y, concat(
            abs(i), transition_x, quad
        )
    )


base = Quadrilteral(a, b, c, d)

# As they say, premature optimisation is the root of all evil.
for ij in itertools.product(range(-n_tiles, n_tiles + 1), repeat=2):
    if ij != (0, 0):
        plot_quad(
            transition_by(base, *ij),
            color='blue'
        )

plot_quad(base, color='r', linewidth=2)

fig.savefig('torus.png')
print('Image saved as ./torus.png')
plt.show()
