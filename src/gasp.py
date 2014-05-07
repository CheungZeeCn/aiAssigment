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
import time

print "load data"
G = util.readGraph2('graph.txt')
print "load data done"
src = '0'
dest = '500'
CXPB = 0.8
MUTPB = 0.05
POPU = 60


def randomPickEdge(G, n, exclude=[]):
    "find out an edge for node n"        
    exclude = set(exclude)
    nodes = set(G[n].keys()) - exclude  
    #print nodes
    if len(nodes) == 0:
        return None
    n2 = random.sample(nodes, 1)[0]
    return n2

def initIndividual(G, s, d):
    idi = creator.Individual()    
    idi.append(s)
    n = s
    while True:
        nextNode = randomPickEdge(G, n, exclude=idi)
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
            n = nextNode

def evalOneMin(individual):
    fit = 0
    for i in range(len(individual)-1):
        s = individual[i]
        d = individual[i+1]
        fit += G[s][d]['weight'] 
    return [fit]

def selTournamentWithoutReplacement(individuals, k):
    chosen = []
    random.shuffle(individuals)
    for i in range(0, len(individuals), k):
        chosen.append(max(individuals[i:i+k], key=attrgetter("fitness")))
    return chosen

def getIndex(n, ind):
    for i in range(len(ind)):
        if ind[i] == n:
            return i
    return -1

def mate(a, b):
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
            showInd(ind)
            ind[:] = ind[:begin] + ind[j:]
            showInd(ind)
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

def mutate(ind, d):
    mutSite = random.randint(0, len(ind) - 2)
    newInd = copy.deepcopy(ind)
    newInd[:] = newInd[:mutSite+1]
    n = newInd[-1]
    while True:
        nextNode = randomPickEdge(G, n, exclude=newInd)
        if nextNode == None: #dead, reset
            return None
        elif nextNode == d: # connected
            newInd.append(nextNode)     
            newInd.nodeSet = set(newInd[1:-1])
            newInd.fitness.values = evalOneMin(newInd)
            return newInd
        else: #growing
            newInd.append(nextNode)     
            n = nextNode


if __name__ == '__main__':
    # types wrapping
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, nodeSet=set())

    # functions wrapping
    toolbox = base.Toolbox()
    tb = toolbox
    toolbox.register("individual", initIndividual, G, src, dest)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMin)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", selTournamentWithoutReplacement)

    getBestTime = None
    timeBegin = time.time()
    # algorithm
    print "Init..."
    pop = toolbox.population(n=POPU)
    # Evaluate the entire population
    updateFitness(pop)
    timeInitDone = time.time()
    print "Initialation Done"
    showPop(pop)

    #calc dj
    shortest = nx.dijkstra_path(G,source=src,target=dest)
    best = creator.Individual()      
    best[:] = shortest
    best.fitness.values = evalOneMin(best)
    timeDjDone = time.time()
    print "Best"
    showInd(best)

    # Begin the evolution
    NGEN = 500
    for g in range(NGEN):
        print("# -- Generation %i --" % g)
        # select
        offspring = toolbox.select(pop, 2) 
        # deep copy out
        # offspring = list(map(toolbox.clone, offspring))
        # cross over
        cxOut = []
        for c1, c2 in zip(offspring[::1], offspring[1::1]):
            if random.random() < CXPB:
                #cc1 cc2 is totally new
                cc1, cc2 = toolbox.mate(c1, c2)
                cxOut.append(cc1)
                cxOut.append(cc2)

        # mutation
        mutantOut = []
        for mutant in offspring:
            if random.random() < MUTPB:
                mut = toolbox.mutate(mutant, dest)
                if mut != None:
                    mutantOut.append(mut)   

        #compose a big group of population
        #bigPop = pop + mutantOut + cxOut
        bigPop = mutantOut + cxOut
        if len(bigPop) < POPU:
            addNum = POPU - len(bigPop)   
            popAdd = tools.selBest(offspring, addNum) 
            bigPop += popAdd
        #select N
        #print "bigPopLen:%d, popLen: %d, muLen: %d, cxLen: %d" \
        #        % (len(bigPop), len(pop), len(mutantOut), len(cxOut))
        pop = tools.selBest(bigPop, POPU)

        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        Min = min(fits)
        Max = max(fits)
        if Min == best.fitness.values[0] and \
            getBestTime != None:
            getBestTime = time.time()
        
        print "==" * 40
        #showPop(pop)
        print("#  DJBEST %s" % best.fitness.values[0])
        print("#  Min %s" % Min)
        print("#  Max %s" % Max)
        print("#  Avg %s" % mean)
        print("#  Std %s" % std)
        print "==" * 40
        timeNow = time.time()
        print "GAOUT: |" + "GEN[%s] Min[%s] Max[%s] Mean[%s] Std[%.2f] timeUsed[%s]" % \
                (g, Min, Max, mean, std, timeNow-timeDjDone)

    timeGaDone = time.time()
    popBest = tools.selBest(pop, 1)[0]
    print "popBest"
    showInd(popBest)
    print "Best"
    showInd(best)
    #djTime, initTime, getBestTime, allTime
    initTimeP = timeInitDone - timeBegin   
    djTimeP = timeDjDone - timeInitDone
    if getBestTime != None:
        bestTimeP = getBestTime - timeDjDone
    else:
        bestTimeP = None
    allTimeP = timeGaDone - timeDjDone
    print "GAOUT: |ALL" + "djTime[%s] initTime[%s] getBestTime[%s] allTime[%s]" %\
        (djTimeP, initTimeP, bestTimeP, allTimeP) 

