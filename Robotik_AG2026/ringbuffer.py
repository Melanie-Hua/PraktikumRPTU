measure_samples = 3
class RingBuffer:
    """ Class that implements a not-yet-full buffer. """
    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.data = []

    class __Full:
        """ Class that implements a full buffer. """
        def add(self, x):
            """ Add an element overwriting the oldest one. """
            self.data[self.currpos] = x
            self.currpos = (self.currpos+1) % self.bufsize
        def get(self):
            """ Return list of elements in correct order. """
            return self.data[self.currpos:]+self.data[:self.currpos]

    def add(self,x):
        """ Add an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.bufsize:
            # Initializing current position attribute
            self.currpos = 0
            # Permanently change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data


# Sample usage to recreate example figure values
import numpy as np
if __name__ == '__main__':

    # Creating ring buffer
    x = RingBuffer(measure_samples)

    print(x.__class__, x.get())

    # Adding elements until buffer is full
    for i in range(measure_samples):
        x.add(i)
    # Displaying class info and buffer data
    print(x.__class__, x.get())

    # Adding data simulating a data acquisition scenario
    print('')
    print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())
    while self.measusre.is_set():
        x.add(i)
        print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())
        test = '{:0.1f}'.format(np.mean(x.get()))
        #print(test)
        