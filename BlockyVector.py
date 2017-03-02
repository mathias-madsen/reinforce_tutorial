import numpy as np


class BlockyVector(list):
    
    def __init__(self, elements):
        """ Create a list of arrays on which various operations can be done. """

        self.extend(elements)
    
    def __repr__(self):
        
        return "BlockyVector(%s)" % list(self)

    def __add__(self, other):
        
        return BlockyVector([a + b for a, b in zip(self, other)])

    def __iadd__(self, other):
        
        blocky_sum = self + other # note: not list concatenation
        
        # assert self.shape == other.shape == blocky_sum.shape
        
        return blocky_sum

    def __sub__(self, other):
        
        return BlockyVector([a - b for a, b in zip(self, other)])

    def __sum__(self, other):

        return BlockyVector([a + b for a, b in zip(self, other)])
    
    def __mul__(self, other):
        
        if type(other) == BlockyVector:
            return BlockyVector([a * b for a, b in zip(self, other)])
        
        else:
            return BlockyVector([other * block for block in self])
    
    def __rmul__(self, scalar):
        
        return self.__mul__(scalar)

    def __pow__(self, exponent):
        
        return BlockyVector([block ** exponent for block in self])

    def sum(self):
        
        return sum(np.sum(block) for block in self)
    
    @property
    def shape(self):
        
        return [block.shape for block in self]
    
    def __eq__(self, other):
        
        return BlockyVector([a == b for a, b in zip(self, other)])
    
    def all(self):
        
        return np.all([np.all(block) for block in self])

    def any(self):
        
        return np.any([np.any(block) for block in self])


if __name__ == '__main__':
    
    one22 = np.ones((2, 2))
    one13 = np.ones((1, 3))
    
    ones = BlockyVector([one22, one13])
    twos = BlockyVector([2*one22, 2*one13])

    assert ones.shape == [(2, 2), (1, 3)]
    assert twos.shape == [(2, 2), (1, 3)]
    assert 2*ones == twos

    ones += ones
    
    assert ones.shape == [(2, 2), (1, 3)] # blocky addition != concatenation
    assert twos.shape == [(2, 2), (1, 3)] # no reason this should fail
    assert ones == twos # true if __iadd__ worked
    
    Ax = np.array([2., 3., 4., 6.])
    Ay = np.array([0., 0., 1., 1.])

    Bx = BlockyVector(Ax)
    By = BlockyVector(Ay)
    
    As = np.array(Ax) + np.array(Ay)
    Ad = np.array(Ax) - np.array(Ay)
    Bm = np.mean([Bx, By], axis=0)
    
    assert np.allclose(0.5*As, Bm)
    assert np.allclose(As, Bx + By)
    assert np.allclose(Ad, Bx - By)
    
    Ax = np.array([2., 6.])
    Ay = np.array([0., 0., 1., 0.])

    b = BlockyVector([Ax, Ay])
    
    bb = b * b
    b2 = b ** 2
    
    blocky_equality = (bb == b2)
    
    assert blocky_equality.all()
    assert blocky_equality.any()
    
    list_of_blocky_vectors = []
    
    for i in range(7):
        
        ones23 = np.ones((2, 3))
        ones14 = np.ones((1, 4))
        ones14 = np.ones((5, 7))
        
        new_blocky_vector = BlockyVector([ones23, ones14])
        list_of_blocky_vectors.append(new_blocky_vector)
    
    mean_vector = np.mean(list_of_blocky_vectors, axis=0)
    mean_vector = BlockyVector(mean_vector)
    
    assert mean_vector.shape == new_blocky_vector.shape
    assert mean_vector == new_blocky_vector # since they're all identical
