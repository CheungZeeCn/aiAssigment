#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# by zhangzhi @2014-04-26 15:59:31 
# Copyright 2014 NONE rights reserved.

import random
import util
from util import *
from deap import base
from deap import creator
from deap import tools
from operator import attrgetter
import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import pylab
from collections import Counter
import time

lInitCounter = Counter()
gInitCounter = Counter()


print "load data"
DG = util.readGraph('graph.txt')
print "load data done"
src = '1'
dest = '10'
LCXPB = 0.8
LMUTPB = 0.05
GCXPB = 0.8
GMUTPB = 0.8
POPU = 60
LPOPU = 30
GPOPU = 30

"""
def randomPickEdge(DG, n, exclude=[]):
    "find out an edge for node n"        
    exclude = set(exclude)
    nodes = set(DG[n].keys()) - exclude  
    #print nodes
    if len(nodes) == 0:
        return None
    n2 = random.sample(nodes, 1)[0]
    return n2
"""

def randomPickEdgeWithCounterOrWeight(DG, n, counter, exclude=[]):
    """find out an edge for node n. 
        Among all the neighbours of n, 
        we pick up the node with minimum value 
        in counter
    """
    exclude = set(exclude)
    nodes = set(DG[n].keys()) - exclude  
    #print nodes
    if len(nodes) == 0:
        return None
    if random.random() >= 0.5:
        n3 = min(nodes, key=lambda x: DG[n][x]['weight'])
        return n3
    else:
        n2 = min(nodes, key=lambda x: counter[x])
        return n2

def randomPickEdgeWithCounter(DG, n, counter, exclude=[]):
    """find out an edge for node n. 
        Among all the neighbours of n, 
        we pick up the node with minimum value 
        in counter
    """
    exclude = set(exclude)
    nodes = set(DG[n].keys()) - exclude  
    #print nodes
    if len(nodes) == 0:
        return None
    n2 = min(nodes, key=lambda x: counter[x])
    #n2 = random.sample(nodes, 1)[0]
    return n2

def initIndividualL(DG, s, d):
    "Use lInitCounter for counting"
    idi = creator.Individual()    
    idi.append(s)
    n = s
    while True:
        nextNode = randomPickEdgeWithCounterOrWeight(DG, n, lInitCounter, exclude=idi)
        if nextNode == None: #dead, reset
            idi = creator.Individual()
            idi.append(s)
            n = s
            idi.nodeSet = set([])
        elif nextNode == d: # connected
            idi.append(nextNode)     
            idi.nodeSet = set(idi[1:-1])
            return idi
        else: #growing
            idi.append(nextNode)     
            lInitCounter[nextNode] += 1
            n = nextNode

def initIndividualG(DG, s, d):
    idi = creator.Individual()    
    idi.append(s)
    n = s
    while True:
        nextNode = randomPickEdgeWithCounter(DG, n, gInitCounter, exclude=idi)
        if nextNode == None: #dead, reset
            idi = creator.Individual()
            idi.append(s)
            n = s
            idi.nodeSet = set([])
        elif nextNode == d: # connected
            idi.append(nextNode)     
            idi.nodeSet = set(idi[1:-1])
            gInitCounter[nextNode] += 1
            return idi
        else: #growing
            idi.append(nextNode)     
            gInitCounter[nextNode] += 1
            n = nextNode

def evalOneMin(individual):
    fit = 0
    for i in range(len(individual)-1):
        s = individual[i]
        d = individual[i+1]
        fit += DG[s][d]['weight'] 
    return [fit]

def selBestOrRandom(individuals, k):
    chosen = []
    if random.random() <= 0.2:
        return tools.selRandom(individuals, k)
    else:
        return tools.selBest(individuals, k)

def selTournamentWithoutReplacement(individuals, k):
    chosen = []
    random.shuffle(individuals)
    #print individuals
    for i in range(0, len(individuals), k):
        #print '..'*40
        #print "chosen:", chosen
        chosen.append(max(individuals[i:i+k], key=attrgetter("fitness")))
        #print individuals[i:i+k], [j.fitness for j in  individuals[i:i+k]]
        #print "chosen:", chosen
        #print '..'*40
    return chosen

def getIndex(n, ind):
    for i in range(len(ind)):
        if ind[i] == n:
            return i
    return -1

def mateL(a, b):
    "cross over"
    interList = list(a.nodeSet & b.nodeSet)
    aa = copy.deepcopy(a)
    bb = copy.deepcopy(b)
    if len(interList) == 0:
        # can do nothing
        #print >>sys.stderr, "cross error, it's hard to gen love with different genders."
        pass
    else:
        node = random.choice(interList)
        #print >>sys.stderr, "OK!", node
        ia = getIndex(node, a)  
        ib = getIndex(node, b)  
        aa = copy.deepcopy(a)
        bb = copy.deepcopy(b)
        aaTail = aa[ia:]  
        bbTail = bb[ib:]  
        aa[ia:] = bbTail
        bb[ib:] = aaTail
        #repair function
        repair(aa, ia)
        repair(bb, ib)
        aa.nodeSet = set(aa[1:-1])
        bb.nodeSet = set(bb[1:-1])
        aa.fitness.values = evalOneMin(aa)
        bb.fitness.values = evalOneMin(bb)
        #return best two
        minList = sorted([a, b, aa, bb], key=lambda x:x.fitness.values[0])
        #print "before"
        #for ini in [a, b, aa, bb]:
        #    showInd(ini)
        #print "after"
        #for ini in minList:
        #    showInd(ini)
        aa, bb = minList[0], minList[1]
    return aa, bb

def mateG(a, b):
    "cross over"
    interList = list(a.nodeSet & b.nodeSet)
    aa = copy.deepcopy(a)
    bb = copy.deepcopy(b)
    if len(interList) == 0:
        # can do nothing
        #print >>sys.stderr, "cross error, it's hard to gen love with different genders."
        pass
    else:
        node = random.choice(interList)
        #print >>sys.stderr, "OK!", node
        ia = getIndex(node, a)  
        ib = getIndex(node, b)  
        aa = copy.deepcopy(a)
        bb = copy.deepcopy(b)
        aaTail = aa[ia:]  
        bbTail = bb[ib:]  
        aa[ia:] = bbTail
        bb[ib:] = aaTail
        #repair function
        repair(aa, ia)
        repair(bb, ib)
        aa.nodeSet = set(aa[1:-1])
        bb.nodeSet = set(bb[1:-1])
        aa.fitness.values = evalOneMin(aa)
        bb.fitness.values = evalOneMin(bb)

    return aa, bb

def repair(ind, cxSite):
    setB = set(ind[cxSite:])
    begin = 0
    end = len(ind)
    for i in range(cxSite):
        n = ind[i]
        if n in setB:#repair
            begin = i
            for j in range(len(ind)-1, cxSite, -1): 
                if ind[j] == n:
                    end = j
                    break
            #print "repair !"
            #raw_input("xx")
            #showInd(ind)
            ind[:] = ind[:begin] + ind[j:]
            #showInd(ind)
            return True
    return False
    
def canMate(a, b):
    if len(a.nodeSet & b.nodeSet) == 0:
        # can do nothing
        return False
    else:
        return True

def updateFitness(pop, updateAll=True):
    if updateAll == True:
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    else:
        invalid_ind = [ind for ind in pop if not ind.fitness.valid] 
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

def mutateL(ind, d):
    mutSite = random.randint(0, len(ind) - 2)
    newInd = copy.deepcopy(ind)
    newInd[:] = newInd[:mutSite+1]
    n = newInd[-1]
    while True:
        nextNode = randomPickEdgeWithCounterOrWeight(DG, n, lInitCounter, exclude=newInd)
        if nextNode == None: #dead, reset
            return None
        elif nextNode == d: # connected
            newInd.append(nextNode)     
            newInd.nodeSet = set(newInd[1:-1])
            newInd.fitness.values = evalOneMin(newInd)
            return newInd
        else: #growing
            newInd.append(nextNode)
            lInitCounter[nextNode] += 1
            n = nextNode

def mutateG(ind, d):
    mutSite = random.randint(0, len(ind) - 2)
    newInd = copy.deepcopy(ind)
    newInd[:] = newInd[:mutSite+1]
    n = newInd[-1]
    while True:
        nextNode = randomPickEdgeWithCounter(DG, n, gInitCounter, exclude=newInd)
        if nextNode == None: #dead, reset
            return None
        elif nextNode == d: # connected
            newInd.append(nextNode)     
            newInd.nodeSet = set(newInd[1:-1])
            newInd.fitness.values = evalOneMin(newInd)
            return newInd
        else: #growing
            newInd.append(nextNode)     
            gInitCounter[nextNode] += 1
            n = nextNode


if __name__ == '__main__':
    # types wrapping
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, nodeSet=set())

    # functions wrapping
    toolbox = base.Toolbox()
    tb = toolbox
    toolbox.register("individualL", initIndividualL, DG, src, dest)
    toolbox.register("individualG", initIndividualG, DG, src, dest)
    toolbox.register("populationL", tools.initRepeat, list, toolbox.individualL)
    toolbox.register("populationG", tools.initRepeat, list, toolbox.individualG)
    toolbox.register("evaluate", evalOneMin)
    toolbox.register("mateL", mateL)
    toolbox.register("mateG", mateG)
    toolbox.register("mutateL", mutateL)
    toolbox.register("mutateG", mutateG)
    toolbox.register("selectL", selTournamentWithoutReplacement)
    toolbox.register("selectG", selBestOrRandom)

    #print DG.nodes()
    #print randomPickEdge(DG, '1', ['6'])
    #idi = initIndividual(DG, '1', '4')
    #print idi
    #print evalOneMin(idi)

    # algorithm
    print "Init..." 
    timeInitBegin = time.time()
    lPop = toolbox.populationL(n=LPOPU)
    timeInitLDone = time.time()
    gPop = toolbox.populationG(n=GPOPU)
    timeInitGDone = time.time()
    # Evaluate the entire population
    updateFitness(lPop)
    updateFitness(gPop)
    timeInitDone = time.time()
    print "Initialation Done [%f]" % (timeInitDone-timeInitBegin)
    print "*"*40, "popL", "*"*40
    showPop(lPop)
    print "*"*40, "popG", "*"*40
    showPop(gPop)

    # Begin the evolution
    NGEN = 100
    for g in range(NGEN):
        print("# -- Generation %i --" % g)
        """ G part !!!"""
        offspring = toolbox.selectG(gPop, GPOPU/2) 
        cxOut = []
        for c1, c2 in zip(offspring[::1], offspring[1::1]):
            if random.random() < GCXPB:
                #cc1 cc2 is totally new
                cc1, cc2 = toolbox.mateG(c1, c2)
                cxOut.append(cc1)
                cxOut.append(cc2)

        # mutation
        mutantOut = []
        for mutant in offspring:
            if random.random() < GMUTPB:
                mut = toolbox.mutateG(mutant, dest)
                if mut != None:
                    mutantOut.append(mut)   
        #compose a big group of population
        #bigPop = pop + mutantOut + cxOut
        bigPop = mutantOut + cxOut + lPop
        gPop = tools.selRandom(bigPop, GPOPU)

        """ L part !!!"""
        # select
        #offspring = toolbox.selectL(lPop, 2) 
        offspring = tools.selBest(lPop, LPOPU) 
        # deep copy out
        # offspring = list(map(toolbox.clone, offspring))
        # cross over
        cxOut = []
        for c1 in offspring:
            if random.random() < LCXPB:
                #cc1 cc2 is totally new
                #random select c2 from gPop
                c2 = tools.selRandom(gPop, 1)[0]
                cc1, cc2 = toolbox.mateL(c1, c2)
                cxOut.append(cc1)
                #cxOut.append(cc2)
            else:
                cxOut.append(c1)

        # mutation
        mutantOut = []
        for mutant in cxOut:
            if random.random() < LMUTPB:
                mut = toolbox.mutateL(mutant, dest)
                if mut != None:
                    if mut.fitness.values[0] < mutant.fitness.values[0]:
                        mutantOut.append(mut)   
                    else:
                        mutantOut.append(mutant)   
                else:
                    mutantOut.append(mutant)   
            else:
                mutantOut.append(mutant)   
        #compose a big group of population
        #bigPop = pop + mutantOut + cxOut
        lPop = mutantOut

        print "==" * 40
        lFits = [ind.fitness.values[0] for ind in lPop]
        length = len(lPop)
        lMean = sum(lFits) / length
        lSum2 = sum(x*x for x in lFits)
        lStd = abs(lSum2 / length - lMean**2)**0.5
        gFits = [ind.fitness.values[0] for ind in gPop]
        length = len(gPop)
        gMean = sum(gFits) / length
        gSum2 = sum(x*x for x in gFits)
        gStd = abs(gSum2 / length - gMean**2)**0.5
        
        #showPop(lPop)
        print("#  lMin %s" % min(lFits))
        print("#  lMax %s" % max(lFits))
        print("#  lAvg %s" % lMean)
        print("#  lStd %s" % lStd)
        print "- " * 40
        print("#  gMin %s" % min(gFits))
        print("#  gMax %s" % max(gFits))
        print("#  gAvg %s" % gMean)
        print("#  gStd %s" % gStd)
        print "==" * 40

        # immigrantion
        gBest = tools.selBest(gPop, 1)[0] 
        lBest = tools.selBest(lPop, 1)[0]
        lWorst = tools.selWorst(lPop, 1)[0]

        if gBest.fitness.values[0] < lBest.fitness.values[0]:
            lPop.append(gBest)
        elif gBest.fitness.values[0] < lWorst.fitness.values[0]:
            lPop.append(gBest)
        else:      
            gPop.append(lBest)

    lPopBest = tools.selBest(lPop, 1)[0]
    print "lPopBest"
    showInd(lPopBest)
    gPopBest = tools.selBest(gPop, 1)[0]
    print "gPopBest"
    showInd(gPopBest)
    shortest = nx.dijkstra_path(DG,source=src,target=dest)
    best = creator.Individual()      
    best[:] = shortest
    best.fitness.values = evalOneMin(best)
    print "Best"
    showInd(best)

    #for node in DG:
    #    if node in best and node in popBest:
    #        DG.node[node]['color'] = 'red'
    #    elif node in best:
    #        DG.node[node]['color'] = 'green'
    #    elif node in popBest:
    #        DG.node[node]['color'] = 'blue'
    #    else:
    #        DG.node[node]['color'] = 'cyan'

    #node_color = [DG.node[v]['color'] for v in DG]
    #edge_labels = dict([((u,v,),d['weight']) 
    #                for u,v,d in DG.edges(data=True)])

    #for a, b in zip(popBest[:-1:2], popBest[1::2]):
    #    DG[a][b]['color'] = 'blue'    

    #print zip(best[:-1], best[1::])
    #for a, b in zip(best[:-1], best[1::]):
    #    if DG[a][b]['color'] == 'blue':
    #        DG[a][b]['color'] = 'red'   
    #    else:
    #        DG[a][b]['color'] = 'green'   

    #edge_colors = [ d['color'] for a,b,d in DG.edges(data=True) ] 
    #
    #pos=nx.spring_layout(DG)
    #nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels)
    #nx.draw(DG, pos, node_color=node_color, node_size=300, edge_color=edge_colors, edge_cmap=plt.cm.Reds)

    #plt.show()

