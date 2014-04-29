#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# by zhangzhi @2014-04-29 11:51:54 
# Copyright 2014 NONE rights reserved.

import util 




if __name__ == '__main__':
    DG = util.genRandomGraph(2200, 18333, 1.0, 10.0)
    f = open("graph1.txt", 'w')
    for a, b, w in DG.edges(data=True):
        w = w['weight']
        f.write("%s %s %.2f\n" % (a, b, w))
    f.close()

