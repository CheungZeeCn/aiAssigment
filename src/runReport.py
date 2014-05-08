#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by zhangzhi @2014-04-26 15:59:31
# Copyright 2014 NONE rights reserved.

import random
import time
import sys
import os

outputPath = './output'
reportPath = None

taskDict = {
    'facebook_weighted.txt': [(0,500), (0,1028), (395,2433), (1985,3254), (985,3086)],
    'hand-made_G1.txt': [(0,2061), (120,2761), (120,11900), (1580,20000), (3580,4600)], 
    'hand-made_G2.txt': [(120, 500), (0, 726), (28, 72), (1028, 2064), (20, 2064)], 
}
#taskDict = {
#    'facebook_weighted.txt': [(0,500), (0,1028)],
#}

progList = ['gasp.py', 'gaspDp.py', 'gaspDpWheel.py', ]
#progList = ['gasp.py' ]

def getTimeStr():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

if __name__ == '__main__':
    reportIdStr = getTimeStr()
    reportPath = os.path.join(outputPath, reportIdStr)
    os.mkdir(reportPath)
    print 'created dir:', reportPath
    times = 5
    NGEN = 5000

    for graph in taskDict: 
        for start, end in taskDict[graph]:  
            for prog in progList:
                for t in range(times):   
                    t = t + 1
                    outputName = "%s_%s_%s_%s_%s_%s.report.txt" % \
                                    (graph, start, end, prog, t, NGEN)
                    outputName = os.path.join(reportPath, outputName)
                    cmd = "python %s %s %s %s %s >> %s" %('./'+prog, graph, start, end, NGEN, outputName)  
                    print "run -> ", cmd
                    os.system(cmd)

