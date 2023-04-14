import time
from random import shuffle
from typing import List

# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool

from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from ..utils.mpqp_utils import gen_cr_from_active_set
from .solver_utils import generate_children_sets


def full_process(program: MPQP_Program, active_set: List[int]):
    """
      This is the function block that is executed in parallel.  This takes a MPQP
    program as well as an active set combination, and checks the feasibility of all
    super sets of cardinality + 1.  This is done without using a pruning list as in
    the other parallel combinatorial algorithm.  This is suited for particularly
    large problems where an exponential number of pruned active sets are stored,
    causing a large memory overhead.

    :param program:
    :param active_set:
    :return:
    """
    # generate all children nodes

    feasible_children = []
    valid_critical_regions = []

    children = generate_children_sets(active_set, program.num_constraints())

    for child in children:

        if program.check_feasibility(child):  # is_feasible(program, child):
            feasible_children.append(child)
        else:
            continue

        if program.check_optimality(child):  # is_optimal(program, child):

            region = gen_cr_from_active_set(program, child)

            if region is not None:
                valid_critical_regions.append(region)

    is_max_depth = len(active_set) + 1 == max(program.num_t(), program.num_x())

    if is_max_depth:
        feasible_children = []

    return [feasible_children, valid_critical_regions]


def solve(program: MPQP_Program, num_cores=-1) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Gupta et al. 2011

    This is the parallel version of the combinatorial.

    url: https://www.sciencedirect.com/science/article/pii/S0005109811003190

    :param num_cores: Sets the number of cores that are allocated to run this
        algorithm.  Default uses all available cores.
    :param program: MPQP to be solved
    :return: the solution of the MPQP
    """

    # thread pool that we will be using
    start = time.time()

    #   make a lambda function which processes an active set on the program (pool
    # map requires a function with one input and one output); consider all active
    # sets produced by adding one constraint and return resulting critical regions
    f = lambda x: full_process(program, x)

    # allocate pool of processor cores
    if num_cores == -1:
        num_cores = num_cpu_cores()
    pool = Pool(num_cores)
    print(f'Spawned threads across {num_cores}')

    to_check = []

    solution = Solution(program, [])

    max_depth = max(program.num_x(), program.num_t()) - len(program.equality_indices)

    if not program.check_feasibility(program.equality_indices):
        return solution

    # initialize the solution
    solution = Solution(program, [])
    # initialize active set to the set of equality constraints
    to_check = list()
    to_check.append(program.equality_indices)
    #   initialize the solution by checking optimality of the set of equality only
    # constraints and adding the critical region if it is optimal
    if program.check_optimality(program.equality_indices):
        region = gen_cr_from_active_set(program, program.equality_indices)
        if region is not None:
            solution.add_region(region)

    # iterate through the layers of the constraint combination tree
    max_depth = (max(program.num_x(), program.num_t())
        - len(program.equality_indices))
    for i in range(max_depth):
        print(f'Time at depth test {i + 1}, {time.time() - start}')
        print(f'Number of active sets to be considered is {len(to_check)}')

        depth_time = time.time()

        f = lambda x: full_process(program, x)

        future_list = []

        shuffle(to_check)

        # queue the list of constraint combinations for parallel processing
        depth_time = time.time()
        outputs = pool.map(f, to_check)
        print(f'Time to run all tasks in parallel {time.time() - depth_time}')

        # save the results of this layer and prepare for the next
        depth_time = time.time()
        to_check = list()
        for output in outputs:
            #   update the constraint combination list to the combinations that were found
            # optimal (the sets adjacent to these will be tested in the next layer)
            if len(output[0]) != 0:
                to_check.extend(output[0])
            # add the newly-optimized critical regions to the solution
            if len(output[1]) != 0:
                for region in output[1]:
                    solution.add_region(region)
        print(f'Time to process all depth outputs {time.time() - depth_time}')

        # break early if there are no more active sets to check
        if len(to_check) == 0:
            break

    # release the pool of processor cores
    pool.clear()

    return solution
