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

print "load data"
DG = util.readGraph('graph.txt')
print "load data done"
src = '1'
dest = '10'
CXPB = 0.8
MUTPB = 0.05
POPU = 30

def randomPickEdge(DG, n, exclude=[]):
    "find out an edge for node n"        
    exclude = set(exclude)
    nodes = set(DG[n].keys()) - exclude  
    #print nodes
    if len(nodes) == 0:
        return None
    n2 = random.sample(nodes, 1)[0]
    return n2

def initIndividual(DG, s, d):
    idi = creator.Individual()    
    idi.append(s)
    n = s
    while True:
        nextNode = randomPickEdge(DG, n, exclude=idi)
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
        fit += DG[s][d]['weight'] 
    return [fit]

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
    reurn -1

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
        #update here
        if aa.fitness < a.fitness:
            aa = a       
        if bb.fitness < b.fitness:
            bb = b       
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
        nextNode = randomPickEdge(DG, n, exclude=newInd)
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
    toolbox.register("individual", initIndividual, DG, src, dest)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMin)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", selTournamentWithoutReplacement)

    #print DG.nodes()
    #print randomPickEdge(DG, '1', ['6'])
    #idi = initIndividual(DG, '1', '4')
    #print idi
    #print evalOneMin(idi)

    # algorithm
    print "Init..."
    pop = toolbox.population(n=POPU)
    # Evaluate the entire population
    updateFitness(pop)
    print "Initialation Done"
    showPop(pop)

    # Begin the evolution
    NGEN = 30
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
                #print "cx ===="
                #showInd(c1)
                #showInd(c2)
                #showInd(cc1)
                #showInd(cc2)
                #print "===="

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
        
        print "==" * 40
        #showPop(pop)
        print("#  Min %s" % min(fits))
        print("#  Max %s" % max(fits))
        print("#  Avg %s" % mean)
        print("#  Std %s" % std)
        print "==" * 40

    popBest = tools.selBest(pop, 1)[0]
    print "popBest"
    showInd(popBest)
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

