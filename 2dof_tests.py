import collections

b = 4
class test:
    arm = collections.namedtuple('arm','L1,m1')
   
    def __init__(self,L1,m1):
        self.par = self.arm(L1=L1,m1=m1)
        
dave = test(10,2)

print(dave.par.m1)