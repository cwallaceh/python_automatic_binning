from __future__ import division

# Debug
# import pdb

# Scientific computing packages
import pandas as pd
import numpy as np

# Local imports
import random
import itertools
from math import isnan

# Remove RuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class GeneticBinning():
    """Genetic Binning
    Parameters
    ----------
    num_ep : int, default: 5
        Number of epochs to run.
    tol : float, default: 0.0001
        Tolerance for stopping criteria.
    in_pop : int, default: 500
        Initial population
    Attributes
    ----------
    bins_ : array
        Array containing the best cuts found by the algorithm.
    iv_ : float
        Max information value found by the algorithm.
    """

    def __init__(self, num_ep=5, tol=0.0001, in_pop=500, min_bin=3, max_bin=9, verbose=True):

        self.num_ep = num_ep
        self.tol = tol
        self.in_pop = in_pop
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.verbose = verbose
        self.pandas = int(pd.__version__.split('.')[1])

    def validate_binsize(self, ls):
        corr_ls = []
        for i in range(len(ls[1:-1])):
            if np.sum(self.var.apply(lambda x: ls[i] < x < ls[i+1])) > (len(self.var) * 0.05):
                corr_ls.append(ls[i+1])
        corr_ls = [-np.inf] + corr_ls + [np.inf]      
        return corr_ls
    
    def pop_gen(self, n):
        """Population generation
        Creates a pd df of length in_pop.
        Each row is a list of n random elements (between min_bin and min(len(var_set), max_bin)) from var_set
        Basically it is a df of random binning cuts of different lengths
        """
        pop = pd.DataFrame(index=np.arange(n), columns=["chromosomes"])
        pop_aux = []
        for i in range(0, len(pop)):
            corr_ls = []
            if len(self.var_set) < self.max_bin:
                rand = random.randint(self.min_bin, len(self.var_set))
            else:
                rand = random.randint(self.min_bin, self.max_bin)
            ls = random.sample(self.var_set, rand)
            ls = [-np.inf] + sorted(ls) + [np.inf]
            pop_aux.append(ls)
        pop["chromosomes"] = pop_aux
        pop['chromosomes'] = pop['chromosomes'].apply(self.validate_binsize)
        pop['tmp'] = pop['chromosomes'].apply(lambda x: ''.join(str(x[1:-1])))
        pop = pop.drop_duplicates('tmp')
        pop['tmp'] = pop['chromosomes'].apply(lambda x: len(x))
        pop = pop.drop(pop[pop['tmp']==2].index)
        return pop

    def crossover(self, pop):
        """Crossover"""
        def remove_adjacent(nums):
            i = 1
            while i < len(nums):
                if nums[i] == nums[i - 1]:
                    nums.pop(i)
                    i -= 1
                i += 1

            return nums

        def combine(l1, l2):
            '''
            Does the actual cross-over
            Takes random n first elements from first and completes with random n last elements from second
            And vice versa
            '''
            limit_1 = random.sample(range(1, len(l1)), 1)[0]
            limit_2 = random.sample(range(1, len(l2)), 1)[0]
            new_l = l1[0:limit_1] + l2[limit_2:]
            new_lp = l2[0:limit_2] + l1[limit_1:]

            return [remove_adjacent(new_l), remove_adjacent(new_lp)]

        pool = pop["chromosomes"].tolist()
        combs = list(itertools.combinations(pool, 2))
        combinations = random.sample(combs, min(len(combs), self.in_pop)) # returns in_pop two-tuple combinations of selected

        children_aux = []
        for combination in combinations:
            child1, child2 = combine(combination[0], combination[1]) # cross-over
            if sorted(child1) == child1 and len(child1) > self.min_bin: # validate order and min size
                children_aux.append(child1)
            if sorted(child2) == child2 and len(child2) > self.min_bin:
                children_aux.append(child2)

        children = pd.DataFrame(index=np.arange(len(children_aux)),
                                columns=["chromosomes"])
        children = pd.DataFrame(index=np.arange(len(children_aux)), columns=['chromosomes'])
        children["chromosomes"] = children_aux

        return children

    def validation(self, pop):
        """
        Validation checks for monoticity in WOE (either decreasing or increasing)
        """
        def is_monotonic(ls):
            a = np.array(ls)
            if np.all(a[1:] >= a[:-1]):
                result = True
            elif np.all(a[1:] <= a[:-1]):
                result = True
            else:
                result = False
            return result

        pop["valid"] = pop["WOE"].map(is_monotonic)
        return pop


    def evaluation(self, pop):
        """Evaluate bins fitness
        First bins var according to each row in pop
        Then builds df with count of good / bad per bin 
        Finally calculates IV of this df and adds WOE / IV per row to pop
        """
        def build_df(bins):
            df = pd.concat([bins, self.metric], axis=1)
            df.columns = ["var", "metric"]
            df = df.groupby(["var", "metric"]).size().unstack().fillna(0)
            df = df.reindex(bins.cat.categories).fillna(0)
            df.columns = ["Good", "Bad"]
            df["Total"] = df["Bad"] + df["Good"]
            return df

        def calculate_iv(df):
            df["Br"] = df["Bad"] / df["Total"]
            df["f(x)Good"] = (df["Good"] / df["Good"].sum())
            df["f(x)Bad"] = (df["Bad"] / df["Bad"].sum())
            df["WOE"] = np.round(np.log(df["f(x)Good"] / df["f(x)Bad"]), 3)
            df["WOE"] = df["WOE"].replace([np.inf, -np.inf], 0)
            df["IV"] = df["WOE"] * (df["f(x)Good"] - df["f(x)Bad"])
            return df

        def get_evaluation(ls):
            a = np.array(ls)
            ls = list(a[np.where(a[1:] >= a[:-1])]) + [np.inf]
            if self.pandas >= 23: #pandas >= 23 has drop duplicates
                bins = pd.cut(self.var, ls, duplicates='drop')
            else:
                bins = pd.cut(self.var, ls, duplicates='drop')
            df = build_df(bins)
            df_calc = calculate_iv(df)
            return df_calc

        df_calc = pop["chromosomes"].map(get_evaluation)
        pop["WOE"] = df_calc.map(lambda x: x["WOE"].tolist())
        pop["IV"] = df_calc.map(lambda x: x["IV"].sum())
        return pop

    def selection(self, pop):
        """
        Selection uses Elitism (10 best) as well as fitness-weighted random selection
        """
        selected = pop[pop["valid"] == 1].sort_values("IV", ascending=False)
        selected['weight'] = selected['IV'] / selected['IV'].sum()
        idx = list(np.random.choice(selected[selected['weight']>0].index, min(len(selected[selected['weight']>0]), 50), replace=False, p=selected[selected['weight']>0]['weight']))
        idx.extend(selected.head(10).index)
        idx = set(idx)
        selected = selected.loc[idx]

        return selected

    def mutation(self, pop):
        """
        Mutation goes over every gene in a sequence and mutates with probability p
        """
        def mutate(ls):
            ls_aux = [min(self.var)] + ls[1:-1] + [max(self.var)]
            ls_result = []
            for i in range(1, len(ls_aux)-1):
                p_mut = random.random()
                if p_mut <= 0.05:
                    ls_result.append(np.round(random.uniform(ls_aux[i-1], ls[i]), self.decimals))
                else:
                    ls_result.append(ls_aux[i])
            return [-np.inf] + ls_result + [np.inf] 
 
        pop["chromosomes"] = pop["chromosomes"].map(mutate)
        return pop

    def fit(self, var, metric, decimals=0):
        self.decimals = decimals
        """Fit the model according to the given data.
        Parameters
        ----------
        var : array
            Variable to optimize
        metric : binary array
            Metric for optimizing the variable (binary)
        decimals: int
            For floats, adjust this parameter to the required # of decimals (2 is a good default value)
        Returns
        -------
        self : object
        """

        def validate_binsize(ls):
            corr_ls = []
            for i in range(len(ls[1:-1])):
                if np.sum(self.var.apply(lambda x: ls[i] < x < ls[i+1])) > (len(self.var) * 0.05):
                    corr_ls.append(ls[i+1])
            corr_ls = [-np.inf] + corr_ls + [np.inf]      
            return corr_ls
    
        self.var = var
        self.var_set = set(var)
        self.metric = metric
        self.bins_ = [-np.inf, 0, np.inf]
        self.iv_ = 0

        iv_aux = -1

        if len(var) != len(metric):
            raise ValueError('var and metric must be the same length')

        pop = self.pop_gen(self.in_pop)
        # Evaluate and validate initial population
        pop = self.evaluation(pop)
        pop = self.validation(pop)
        for i in range(self.num_ep):
            if self.verbose:
                print("Epoch: " + str(i))
            # Selection of chromosomes
            selected = self.selection(pop)
            # Crossover of chromosomes
            children = self.crossover(selected)
            # Mutate chromosomes
            children = self.mutation(children)
            children['chromosomes'] = children['chromosomes'].apply(self.validate_binsize)
            # Evaluate and validate children
            children = self.evaluation(children)
            children = self.validation(children)
            pop = children
            print(f'len pop: {len(pop)}')
            iv = pop[pop["valid"] == True]["IV"].max()
            iv_diff = iv - iv_aux
            if self.verbose:
                print("IV difference: " + str(iv_diff))
            if (abs(iv_diff) >= self.tol) or (isnan(iv_diff)):
                if iv >= iv_aux:
                    self.iv_ = iv
                    self.bins_ = pop.ix[pop[pop["valid"] == True]["IV"].astype(float).idxmax()]["chromosomes"]
                    iv_aux = iv
                    if self.verbose:
                        print("Population max IV: " + str(round(self.iv_, 3)))
                        print("Best cuts: " + str(self.bins_))
                else:
                    pass
        return self