def test():
    import numpy
    array = numpy.random.rand(1024 * 1024, 1)
    numpy.exp(array)


if __name__ == '__main__':
    import timeit
    print(timeit.timeit(
        stmt="test()", 
        setup='from __main__ import test',
        number=100))
