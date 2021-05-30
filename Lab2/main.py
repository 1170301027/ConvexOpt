
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from pre import point, example, drawResult
from constrainedOptimization import ALM, ADMM

if __name__ == '__main__':
    epsilon = 0.01
    loss_function, start = example(rho=1.0), point(-2, -2)
    y_t = 0

    def testALM():
        points = ALM(loss_function, start, y_t=y_t, epsilon=epsilon)
        for i in points:
            print(str(i))
        drawResult(loss_function, points, 'ALM', epsilon)

    def testADMM():
        points = ADMM(loss_function, start, y_t=y_t, epsilon=epsilon)
        for i in points:
            print(str(i))
        drawResult(loss_function, points, 'ADMM', epsilon)

    testALM()
    # testADMM()