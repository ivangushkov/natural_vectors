import primefac
import time
import numpy as np
from numba import int64
from numba.experimental import jitclass # TODO: compile for fast aff booiii


class NaturalVector():
    """
    A class implementing the Natural Vectors concept, which is just an elaborate (and rigorous) shitpost by 
    my main mans Lasse "Inverse Cowgirl" Tomsson

    Just for fun playa, don't hate :)

    """
    
    def __init__(self, n):
        # Natural number
        self.n = n

        # Get factors
        self.factors = np.array(list(primefac.primefac(n)))
        # a list of unique primes sorted in ascending order
        self.unique_primes = list(set(self.factors))
        self.unique_primes.sort()


        self.vector_scalars = [(self.factors == unique_prime).sum() for unique_prime in self.unique_primes]
    
    def unit_vector(self, vec):
        """
        scale vector to unit length
        """
        # cursedcuresedcursedcursedcursed
        if np.linalg.norm(vec) == 0:
            return vec

        return vec / np.linalg.norm(vec)
    
    def angle_between(self, vec1, vec2):
        """
        Calculates the angle between two vectors. If the vectors are the same, returns 0
        """
        vec1_u = self.unit_vector(vec=vec1)
        vec2_u = self.unit_vector(vec=vec2)

        if len(vec1_u) == 1:              # Scalar case
            if vec1_u == vec2:
                return 0
        elif (vec1_u == vec2_u).all():    # Vector case
            return 0

        return np.arccos(np.clip(np.dot(vec1_u, vec2_u), -1.0, 1.0))

    def get_natural_vector(self):
        
        if self.n == 1:
            return [0]

        # Get biggest prime and gen a lut
        biggest_prime   = np.max(self.factors)
        primes_lut      = np.array(list(primefac.primegen(limit=biggest_prime)))


        # Get the positions of the unique primes
        prime_factor_indices = np.where(np.in1d(primes_lut, self.unique_primes))[0]
        prime_factor_indices = np.append(prime_factor_indices, len(primes_lut)) # The biggest prime is actually not part of the LUT


        basis_vecs = np.zeros((max(prime_factor_indices)+1, len(self.vector_scalars)))

        for i,prime_ind in enumerate(prime_factor_indices):
            basis_vecs[prime_ind,i] = 1

        natural_vec = basis_vecs @ self.vector_scalars

        return natural_vec
    
    def get_natural_angle(self, nat_vec):
        """
        Given another natural vector nat_vec, get the natural angle between the two vectors
        """
        u = self.get_natural_vector()
        v = nat_vec.get_natural_vector()

        len_diff = len(u) - len(v)

        if len_diff > 0:
            v = np.append(v,np.zeros((len_diff,)))
        elif len_diff < 0:
            u = np.append(u, np.zeros((abs(len_diff),)))
        else:
            pass

        angle = self.angle_between(vec1=u, vec2=v)

        return angle
        
        


if __name__ == "__main__":
    """
    Example of use:
    """
    natvec1 = NaturalVector(n=420)
    natvec2 = NaturalVector(n=69)


    angle = natvec1.get_natural_angle(nat_vec=natvec2) * 180 / np.pi
    print(f"The angle is: {angle}")