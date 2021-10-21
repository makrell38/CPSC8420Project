from argparse import Namespace
import copy
import heapq
import numpy as np
import math
import random
import sys
sys.path.append('bayesian-algorithm-execution')
from bax.alg.algorithms import Algorithm
from bax.util.misc_util import dict_to_namespace
from bax.util.graph import Vertex, backtrack_indices, edges_of_path, jaccard_similarity

class nonDom(Algorithm):
    """Class to find non-dominated points"""
    def __init__(
        self,
        params=None,
        objectives=None,
        verbose=True,
    ):
        super().__init__(params, verbose)

        self.objectives = objectives

    def set_params(self, params):
        """Set parameters."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "nonDom")
        self.params.x_path = getattr(params, "x_path", [])

    def initialize(self):
        """Set execution path."""
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        """
        Return the next x in the algorithm or none if algorithm is done.
        """
        len_path = len(self.exe_path.x)
        x_path = self.params.x_path
        next_x = x_path[len_path] if len_path < len(x_path) else None
        #print("next_x", next_x)
        return next_x

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm traditionally.
        """
        self.initialize()

        nonDomarr = []
        y_path = [f(x) for x in self.params.x_path]
        nonDomarr.append([valY[0] for valY in y_path])
        nonDomarr = np.argsort(nonDomarr[0])
        ret = set()
        ret.add(nonDomarr[len(nonDomarr)-1])
        to_remove = set()
        to_add = set()
        for i in range(len(nonDomarr)-2,-1,-1):
            y = y_path[nonDomarr[i]]
            add = True
            for s in ret:
                val = self.params.x_path[s]
                test_y = f(val)
                if self.dominates(test_y, y):
                    add = False
                    break
                if self.dominates(y, test_y):
                    to_remove.add(s)
            if add:
                to_add.add(nonDomarr[i])
            for r in to_remove:
                ret.remove(r)
            for a in to_add:
                ret.add(a)
        
        output = []
        for s in ret:
            output.append(self.params.x_path[s])
        return self.exe_path, output

    def dominates(self,p1, p2):
        """
        Check if p1 dominates p2.
        """
        objectives = self.objectives
        better = False
        for i in range(objectives):
            if p1[i] < p2[i]:
                return False
            if p1[i] > p2[1]:
                better = True
        return better

    def get_exe_path_nondom(self):
        """Return indexes of non-dominated points."""
        
        nonDomarr = []
        nonDomarr.append([valY[0] for valY in self.exe_path.y])
        nonDomarr = np.argsort(nonDomarr[0])
        ret = set()
        ret.add(nonDomarr[len(nonDomarr)-1])
        to_remove = set()
        to_add = set()
        for i in range(len(nonDomarr)-2,-1,-1):
            y = self.exe_path.y[nonDomarr[i]]
            add = True
            for s in ret:
                test_y = self.exe_path.y[s]
                if self.dominates(test_y, y):
                    add = False
                    break
                if self.dominates(y, test_y):
                    to_remove.add(s)
            if add:
                to_add.add(nonDomarr[i])
            for r in to_remove:
                ret.remove(r)
            for a in to_add:
                ret.add(a)
        return ret


    def get_exe_path_crop(self):
        """
        Return cropped execution path.
        """
        nonDom_ind = self.get_exe_path_nondom()
        exe_path_crop = Namespace()
        exe_path_crop.x = [self.exe_path.x[idx] for idx in nonDom_ind]
        exe_path_crop.y = [self.exe_path.y[idx] for idx in nonDom_ind]
        #print("EXE_PATHCROP", exe_path_crop)
        return exe_path_crop
        #return self.exe_path


    def get_copy(self):
        """Copy the algorithm."""
        return copy.deepcopy(self)
    
    def get_output(self):
        """Return the output of the algorithm."""
        nonDom_ind = self.get_exe_path_nondom()
        ret = Namespace()
        ret.x = [self.exe_path.x[idx] for idx in nonDom_ind]
        ret.y = [self.exe_path.y[idx] for idx in nonDom_ind]
        return ret
