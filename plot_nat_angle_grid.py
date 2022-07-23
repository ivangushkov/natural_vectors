
import numpy as np
from natural_vectors import NaturalVector
import matplotlib.pyplot as plt
import time

def main():
    """
    Create a NxN grid visualising the natural angles between the first N natural numbers
    """
    N = 10

    start_time = time.time()

    grid = np.empty((N, N))
    for i in range(N):

        n = i + 1
        print(f"bruh {n}")
        for j in range(N):

            m = j + 1

            nat_vec1 = NaturalVector(n)
            nat_vec2 = NaturalVector(m)

            grid[i, j] = nat_vec1.get_natural_angle(nat_vec=nat_vec2)

    print(time.time() - start_time)
    plt.matshow(grid)
    plt.show()


if __name__ == "__main__":

    main()