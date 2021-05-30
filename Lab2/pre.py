
import math
import numpy as np
import matplotlib.pyplot as plt

class point:
    x, z = 0, 0

    def __init__(self, x: float, z: float):
        self.x, self.z = x, z

    def __add__(self, other):
        return point(self.x + other.x, self.z + other.z)

    def __sub__(self, other):
        return point(self.x - other.x, self.z - other.z)

    def __len__(self):
        return 2

    def __str__(self):
        return str(self.x) + "," + str(self.z)

    def dis(self):
        return math.sqrt(self.x ** 2 + self.z ** 2)


class example:
    rho = 1.0

    def __init__(self, rho: float):
        self.rho = rho

    def F(self, p: point):
        return (p.x - 1) ** 2 + (p.z - 2) ** 2

    def subject_to(self, p: point):
        return 2 * p.x + 3 * p.z - 5

    def L(self, p: point, y_t: float):
        return self.F(p) + y_t * self.subject_to(p) + 0.5 * self.rho * (self.subject_to(p) ** 2)


def advance_retreat_method(loss_function: example, lambda_: float, start: point, direction: list, step=0,
                           delta=0.1) -> tuple:
    alpha0, point0 = step, start

    alpha1 = alpha0 + delta
    point1 = point0 + point(direction[0] * delta, direction[1] * delta)
    if loss_function.L(point0, lambda_) < loss_function.L(point1, lambda_):
        while True:
            delta *= 2
            alpha2 = alpha0 - delta
            point2 = point0 - point(direction[0] * delta, direction[1] * delta)
            if loss_function.L(point2, lambda_) < loss_function.L(point0, lambda_):
                alpha1, alpha0 = alpha0, alpha2
                point1, point0 = point0, point2
            else:
                return alpha2, alpha1
    else:
        while True:
            delta *= 2
            alpha2 = alpha1 + delta
            point2 = point1 + point(direction[0] * delta, direction[1] * delta)
            if loss_function.L(point2, lambda_) < loss_function.L(point1, lambda_):
                alpha0, alpha1 = alpha1, alpha2
                point0, point1 = point1, point2
            else:
                return alpha0, alpha2


def golden_search(loss_function: example, lambda_: float, start: point, direction: list, epsilon=0.1) -> float:
    a, b = advance_retreat_method(loss_function, lambda_, start, direction)

    # find the minimum
    golden_num = (math.sqrt(5) - 1) / 2
    p, q = a + (1 - golden_num) * (b - a), a + golden_num * (b - a)
    while abs(a - b) > epsilon:
        f_p = loss_function.L(start + point(direction[0] * p, direction[1] * p), lambda_)
        f_q = loss_function.L(start + point(direction[0] * q, direction[1] * q), lambda_)
        if f_p < f_q:
            b, q = q, p
            p = a + (1 - golden_num) * (b - a)
        else:
            a, p = p, q
            q = a + golden_num * (b - a)

    return (a + b) / 2


def drawResult(loss_function: example, points: list, label: str, epsilon: float, other_label=''):
    plt.figure()
    plt.title(
        label + '(rho=' + str(loss_function.rho) + other_label + ',epsilon=' + str(epsilon) + ',iteration=' + str(
            len(points)) + ')')

    # draw the function and condition
    X = np.arange(-2, 4.5 + 0.05, 0.05)
    Y = np.arange(-2, 3 + 0.05, 0.05)
    Y2 = (5 - 2 * X) / 3
    X, Y = np.meshgrid(X, Y)
    Z1 = loss_function.F(point(X, Y))
    contour2 = plt.contour(X, Y, Z1, colors='k')
    plt.clabel(contour2, fontsize=8, colors='k')

    # draw the result
    x, z = [], []
    for p in points:
        x.append(p.x)
        z.append(p.z)
    plt.plot(x, z, 'b*-')
    contour1 = plt.contour(X, Y, Z1, [loss_function.F(points[-1])], colors='blue')
    plt.clabel(contour1, inline=True, fontsize=8, colors='blue')

    # draw the start point
    plt.scatter(points[0].x, points[0].z, color='blue')
    plt.text(points[0].x, points[0].z, 'start(%.3g,%.3g,%.3g)' % (points[0].x, points[0].z, loss_function.F(points[0])),
             color='blue', verticalalignment='top')
    # draw the end point
    plt.scatter(points[-1].x, points[-1].z, color='blue')
    plt.text(points[-1].x, points[-1].z,
             'end(%.3g,%.3g,%.3g)' % (points[-1].x, points[-1].z, loss_function.F(points[-1])), color='blue',
             verticalalignment='bottom')
    plt.show()
