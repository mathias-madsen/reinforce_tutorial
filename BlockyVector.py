import numpy as np


class BlockyVector(list):
    
    def __init__(self, blocks):
        """ Create a list of arrays on which various operations can be done. """

        for block in blocks:
            self.append(block)
    
    def __repr__(self):
        
        return "BlockyVector(%s)" % list(self)

    def __add__(self, other):
        
        if type(other) is list:
            raise Warning("You probably didn't mean to add a list to a BlockyVector")
        
        return BlockyVector([a + b for a, b in zip(self, other)])

    def __iadd__(self, other):
    
        return self.__add__(other)
    
    def __sum__(self, other):

        return BlockyVector([a + b for a, b in zip(self, other)])
    
    def __mul__(self, other):
        
        if np.isscalar(other):
            return BlockyVector([other * block for block in self])

        else:
            return BlockyVector([a * b for a, b in zip(self, other)])
    
    def __rmul__(self, other):
        
        return self.__mul__(other)
        
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
    
    Ax = np.array([2., 3., 4., 6.])
    Ay = np.array([0., 0., 1., 1.])

    Bx = BlockyVector(Ax)
    By = BlockyVector(Ay)
    
    As = np.array(Ax) + np.array(Ay)
    Bm = np.mean([Bx, By], axis=0)
    
    assert np.allclose(0.5*As, Bm)
    assert np.allclose(As, Bx + By)
    
    Ax = np.array([2., 6.])
    Ay = np.array([0., 0., 1., 0.])

    b = BlockyVector([Ax, Ay])
    
    bb = b * b
    b2 = b ** 2
    
    blocky_equality = (bb == b2)
    
    assert blocky_equality.all()
    assert blocky_equality.any()
    
    several_blocky_vectors = []
    
    for i in range(1000):
        one_block = np.random.normal(size=(2, 3))**2
        another_block = np.random.normal(size=(1, 4))**2
        new_blocky_vector = BlockyVector([one_block, another_block])
        several_blocky_vectors.append(new_blocky_vector)
    
    gradient = np.mean(several_blocky_vectors, axis=0)
    location = new_blocky_vector
    
    print(location)
    print()
    
    location += gradient

    print(location)
    print()
    
    print(type(location))
    print(location.shape)
    