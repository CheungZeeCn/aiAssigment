#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# by zhangzhi @2014-04-28 13:18:02 
# Copyright 2014 NONE rights reserved.

import random
import util
from util import *
from deap import base
from deap import creator
from deap import tools
import networkx as nx

def readGraph(fname):
    DG=nx.DiGraph()        
    with open(fname) as f:
        for l in f:
            edge = l.strip().split(' ')
            #print edge
            if edge == []:
                continue
            DG.add_edge(edge[0], edge[1], weight=float(edge[2]), color='black')  
    return DG

def showInd(ind):
    print "ind:", ind, "nodeSet:", ind.nodeSet, "fitNess", ind.fitness

def showPop(pop):
    print '--'*40
    for ind in pop:
        showInd(ind)
    print '--'*40

def genRandomGraph(nN, nE, wMin, wMax):
    wMin = float(wMin)
    wMax = float(wMax)
    DG = nx.DiGraph()
    maxNE = nN * (nN -1)        
    if nE > maxNE:
        nE = maxNE
    elif nE < nN * 2:
        nE = nN - 1

    edgeSet = set()
    nodeSet = set()

    #connected  
    eCount = 0
    nList  = range(nN)
    random.shuffle(nList) 
    for a, b in zip(nList[:-1], nList[1:]):
        w = random.uniform(wMin, wMax)
        DG.add_edge(a, b, weight=w, color='black')  
        w = random.uniform(wMin, wMax)
        DG.add_edge(b, a, weight=w, color='black')  
        edgeSet.add((a,b))
        edgeSet.add((b,a))
        eCount += 2

    while True:
        if eCount == nE:
            break
        a = random.randint(0, nN-1) 
        b = random.randint(0, nN-1) 
        w = random.uniform(wMin, wMax)  
        if a == b or ((a, b) in edgeSet):
            continue
        DG.add_edge(a, b, weight=w, color='black')
        edgeSet.add((a, b))
        eCount += 1
    return DG
    

if __name__ == '__main__':
    print readGraph("graph.txt")
