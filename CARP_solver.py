import time
import numpy as np
import copy
import argparse
rules = ['min_depot_distance', 'max_depot_distance', 'min_demand_cost_ratio', 'max_demand_cost_ratio', 'mix']
info = dict()
verbose = False

def floyd_warshall(graph):
    n = len(graph)
    for k in range(1, n):
        for i in range(1, n):
            for j in range(1, n):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    return graph

def path_scanning(tasks, rule, capacity):
    def is_better_task(task, selected_task, load):
        if selected_task is None:
            return True
        elif rule == 'min_depot_distance':
            return info['graph'][task.node_2][info['depot']] < info['graph'][selected_task.node_2][info['depot']]
        elif rule == 'max_depot_distance':
            return info['graph'][task.node_2][info['depot']] > info['graph'][selected_task.node_2][info['depot']]
        elif rule == 'min_demand_cost_ratio':
            return task.demand / task.cost < selected_task.demand / selected_task.cost
        elif rule == 'max_demand_cost_ratio':
            return task.demand / task.cost > selected_task.demand / selected_task.cost
        elif rule == 'mix':
            if load < capacity / 2:
                return info['graph'][task.node_2][info['depot']] > info['graph'][selected_task.node_2][info['depot']]
            else:
                return info['graph'][task.node_2][info['depot']] < info['graph'][selected_task.node_2][info['depot']]
        return False


    def get_next_task(current_node, current_load):
        selected_task = None
        min_distance = np.inf

        for task in free_list.values():
            if current_load + task.demand > capacity:
                continue
            dist = info['graph'][current_node][task.node_1]
            if dist < min_distance or (dist == min_distance and is_better_task(task, selected_task, current_load)):
                selected_task = task
                min_distance = dist
        return selected_task, min_distance
                

    free_list = copy.deepcopy(tasks)
    routes, loads, costs = [], [], 0
    depot_idx = info['depot']
    while free_list:
        current_node = info['depot']
        current_route, current_load, current_cost = [(depot_idx, depot_idx)], 0, 0
        while True:
            next_task, travel_cost = get_next_task(current_node, current_load)
            if next_task is None:
                break
            current_route.append((next_task.node_1, next_task.node_2))
            current_cost += travel_cost + next_task.cost
            current_node = next_task.node_2
            current_load += next_task.demand
            free_list.pop((next_task.node_1, next_task.node_2))
            free_list.pop((next_task.node_2, next_task.node_1))
        current_cost += info['graph'][current_node][info['depot']]
        current_route.append((depot_idx, depot_idx))
        routes.append(current_route)
        loads.append(current_load)
        costs += current_cost
        
    return routes, loads, costs

class Task:
    def __init__(self, node_1, node_2, cost, demand):
        self.node_1 = node_1
        self.node_2 = node_2
        self.cost = cost
        self.demand = demand

class Solution:
    def __init__(self, routes, loads, costs):
        self.routes = routes
        self.loads = loads
        self.costs = costs

    def self_check(self):
        # debug function to check if the cost is calculated correctly
        costs = 0
        for i in range(len(self.routes)):
            route = self.routes[i]
            for j in range(len(route) - 1):
                costs += info['graph'][route[j][1]][route[j + 1][0]]
            for j in range(1, len(route) - 1):
                costs += info['tasks'][(route[j][0], route[j][1])].cost
        if costs != self.costs:
            print('Costs do not match')
            print('Calculated costs:', costs)
            print('Stored costs:', self.costs)
            print('Routes:', self.routes)
            print('Loads:', self.loads)
            print('Graph:')
            print('   ' + ' '.join(map(str, range(1, len(info['graph'])))))
            for i in range(1, len(info['graph'])):
                print(f'{i}:', end=' ')
                for j in range(1, len(info['graph'])):
                    print(int(info['graph'][i][j]), end=' ')
                print()
            exit()
        return

    def print(self):
        if verbose:
            for i in range(len(self.routes)):
                print(f'Route {i + 1}:', end=' ')
                route = self.routes[i][1:-1]
                print(' -> '.join(map(str, route)))
                print(f'    Load: {self.loads[i]}')
                print(f'    Valid: {self.loads[i] <= info["capacity"]}')
            
            print(f'Total cost: {self.costs}')
        else:
            print('s', end=' ')
            route_line = []
            for route in self.routes:
                if len(route) == 2:
                    continue
                route_line.append(0)
                route = route[1:-1]
                route_line.extend(route)
                route_line.append(0)
            print(','.join(map(str, route_line)).replace(" ", ""))
            print('q', end=' ')
            print(int(self.costs))

    def get_random_route(self):
        route_idx = np.random.choice(len(self.routes))
        return route_idx, self.routes[route_idx]
    
    def get_random_task(self, route_idx=None):
        if route_idx is None:
            while True:
                route_idx, route = self.get_random_route()
                if len(route) > 2: # ensure there is at least 1 task in the route
                    break
        else:
            route = self.routes[route_idx]
        task_idx = np.random.choice(len(route) - 2) + 1
        return route_idx, task_idx, route[task_idx]
    
    def get_random_task_index(self, route_idx=None):
        # find a index to insert a task
        # do not need to check if the route is empty
        if route_idx is None:
            route_idx, route = self.get_random_route()
        else:
            route = self.routes[route_idx]

        task_idx = np.random.choice(len(route) - 1) + 1
        return route_idx, task_idx
    
    def remove_task(self, route_idx, task_idx_start, len=None):
        if len is None:
            task = self.routes[route_idx][task_idx_start]
            prev_task = self.routes[route_idx][task_idx_start - 1]
            next_task = self.routes[route_idx][task_idx_start + 1]
            self.costs -= info['graph'][prev_task[1]][task[0]] + info['graph'][task[1]][next_task[0]]
            self.costs += info['graph'][prev_task[1]][next_task[0]]
            self.routes[route_idx].pop(task_idx_start)
            self.loads[route_idx] -= info['demands'][task]
            # if verbose:
            #     print(f'remove edge {prev_task[1]} -> {task[0]} with cost {info["graph"][prev_task[1]][task[0]]}')
            #     print(f'remove edge {task[1]} -> {next_task[0]} with cost {info["graph"][task[1]][next_task[0]]}')
            #     print(f'insert edge {prev_task[1]} -> {next_task[0]} with cost {info["graph"][prev_task[1]][next_task[0]]}')
        else:
            for task_idx in range(task_idx_start, task_idx_start + len):
                task = self.routes[route_idx][task_idx]
                self.loads[route_idx] -= info['demands'][task]
            prev_task = self.routes[route_idx][task_idx_start - 1]
            next_task = self.routes[route_idx][task_idx_start + len]
            self.costs -= info['graph'][prev_task[1]][self.routes[route_idx][task_idx_start][0]] + info['graph'][self.routes[route_idx][task_idx_start + len - 1][1]][next_task[0]]
            self.costs += info['graph'][prev_task[1]][next_task[0]]
            # if verbose:
            #     print(f'remove edge {prev_task[1]} -> {self.routes[route_idx][task_idx_start][0]} with cost {info["graph"][prev_task[1]][self.routes[route_idx][task_idx_start][0]]}')
            #     print(f'remove edge {self.routes[route_idx][task_idx_start + len - 1][1]} -> {next_task[0]} with cost {info["graph"][self.routes[route_idx][task_idx_start + len - 1][1]][next_task[0]]}')
            #     print(f'insert edge {prev_task[1]} -> {next_task[0]} with cost {info["graph"][prev_task[1]][next_task[0]]}')
            for _ in range(len):
                self.routes[route_idx].pop(task_idx_start)


    def insert_task(self, route_idx, task_idx_start, tasks : list, reversed=False):
        if isinstance(tasks, tuple):
            if reversed:
                task = (tasks[1], tasks[0])
            else:
                task = tasks
            prev_task = self.routes[route_idx][task_idx_start - 1]
            next_task = self.routes[route_idx][task_idx_start]
            self.costs += info['graph'][prev_task[1]][task[0]] + info['graph'][task[1]][next_task[0]]
            self.costs -= info['graph'][prev_task[1]][next_task[0]]
            self.routes[route_idx].insert(task_idx_start, task)
            self.loads[route_idx] += info['demands'][task]
            # if verbose:
            #     print(f'insert edge {prev_task[1]} -> {task[0]} with cost {info["graph"][prev_task[1]][task[0]]}')
            #     print(f'insert edge {task[1]} -> {next_task[0]} with cost {info["graph"][task[1]][next_task[0]]}')
            #     print(f'remove edge {prev_task[1]} -> {next_task[0]} with cost {info["graph"][prev_task[1]][next_task[0]]}')
        elif isinstance(tasks, list):
            prev_task = self.routes[route_idx][task_idx_start - 1]
            next_task = self.routes[route_idx][task_idx_start]
            self.costs -= info['graph'][prev_task[1]][next_task[0]]
            if reversed:
                self.costs += info['graph'][prev_task[1]][tasks[-1][1]] + info['graph'][tasks[0][0]][next_task[0]]
                # if verbose:
                #     print(f'remove edge {prev_task[1]} -> {next_task[0]} with cost {info["graph"][prev_task[1]][next_task[0]]}')
                #     print(f'insert edge {prev_task[1]} -> {tasks[-1][1]} with cost {info["graph"][prev_task[1]][tasks[-1][1]]}')
                #     print(f'insert edge {tasks[0][0]} -> {next_task[0]} with cost {info["graph"][tasks[0][0]][next_task[0]]}')
            else:
                self.costs += info['graph'][prev_task[1]][tasks[0][0]] + info['graph'][tasks[-1][1]][next_task[0]]
                # if verbose:
                #     print(f'remove edge {prev_task[1]} -> {next_task[0]} with cost {info["graph"][prev_task[1]][next_task[0]]}')
                #     print(f'insert edge {prev_task[1]} -> {tasks[0][0]} with cost {info["graph"][prev_task[1]][tasks[0][0]]}')
                #     print(f'insert edge {tasks[-1][1]} -> {next_task[0]} with cost {info["graph"][tasks[-1][1]][next_task[0]]}')
            if reversed:
                tasks = [(task[1], task[0]) for task in tasks]
            else:
                tasks = tasks[::-1]
            for task in tasks:
                self.loads[route_idx] += info['demands'][task]
                self.routes[route_idx].insert(task_idx_start, task)

    def mutate(self):
        def flip():
            route_idx, task_idx, task = self.get_random_task()
            self.remove_task(route_idx, task_idx)
            self.insert_task(route_idx, task_idx, task, reversed=True)
            return True

        def single_insertion():
            route_idx, task_idx, task = self.get_random_task()
            self.remove_task(route_idx, task_idx)

            # insert into the same route with 1/2 probability
            if np.random.rand() < 0.5:
                new_route_idx, new_task_idx = self.get_random_task_index(route_idx=route_idx)
                self.insert_task(route_idx, new_task_idx, task)
                return True
            else:
                new_route_idx, new_task_idx = self.get_random_task_index()
                if new_route_idx == route_idx and task_idx != new_task_idx or self.loads[new_route_idx] + info['demands'][task] <= info['capacity']:
                    self.insert_task(new_route_idx, new_task_idx, task)
                    return True
                else:
                    self.insert_task(route_idx, task_idx, task) # restore the task if no valid route is found
                    return False
            
    
        def double_insertion():
            route_idx, task_idx, _ = self.get_random_task()
            if task_idx < len(self.routes[route_idx]) - 2:
                tasks = self.routes[route_idx][task_idx : task_idx + 2]
                self.remove_task(route_idx, task_idx, len=2)
                if np.random.rand() < 0.5:
                    new_route_idx, new_task_idx = self.get_random_task_index(route_idx=route_idx)
                    self.insert_task(route_idx, new_task_idx, tasks)
                    return True
                else:
                    new_route_idx, new_task_idx = self.get_random_task_index()
                    if new_route_idx == route_idx and task_idx != new_task_idx or self.loads[new_route_idx] + sum(info['demands'][task] for task in tasks) <= info['capacity']:
                        self.insert_task(new_route_idx, new_task_idx, tasks)
                        return True
                    else:
                        self.insert_task(route_idx, task_idx, tasks)
                        return False
            
        def swap():
            route_idx_1, task_idx_1, task_1 = self.get_random_task()
            if np.random.rand() < 0.5:
                route_idx_2, task_idx_2, task_2 = self.get_random_task(route_idx=route_idx_1)
            else:
                route_idx_2, task_idx_2, task_2 = self.get_random_task()
            if route_idx_1 == route_idx_2 and task_idx_1 == task_idx_2:
                return False
            if self.loads[route_idx_1] - info['demands'][task_1] + info['demands'][task_2] > info['capacity'] or self.loads[route_idx_2] - info['demands'][task_2] + info['demands'][task_1] > info['capacity']:
                return False
            self.remove_task(route_idx_1, task_idx_1)
            self.insert_task(route_idx_1, task_idx_1, task_2)
            self.remove_task(route_idx_2, task_idx_2)
            self.insert_task(route_idx_2, task_idx_2, task_1)
            return True

        def two_opt_single():
            # 2-opt trick for single route
            route_idx, task_idx_1, _ = self.get_random_task()
            _, task_idx_2, _ = self.get_random_task(route_idx=route_idx)
            starting_task_idx = min(task_idx_1, task_idx_2)
            ending_task_idx = max(task_idx_1, task_idx_2)

            tasks_to_reverse = self.routes[route_idx][starting_task_idx:ending_task_idx + 1] # these tasks will be reversed
            self.remove_task(route_idx, starting_task_idx, len=len(tasks_to_reverse))
            self.insert_task(route_idx, starting_task_idx, tasks_to_reverse, reversed=True)
            return True

        def two_opt_double():
            route_idx_1, route_1_task_idx, _ = self.get_random_task()
            route_idx_2, route_2_task_idx, _ = self.get_random_task()
            if route_idx_1 == route_idx_2:
                return False
            
            if np.random.rand() < 0.5: # split and connect in the same order
                route_1_tasks = self.routes[route_idx_1][route_1_task_idx:-1]
                route_2_tasks = self.routes[route_idx_2][route_2_task_idx:-1]
                route_1_tasks_demand = sum(info['demands'][task] for task in route_1_tasks)
                route_2_tasks_demand = sum(info['demands'][task] for task in route_2_tasks)
                if self.loads[route_idx_1] - route_1_tasks_demand + route_2_tasks_demand > info['capacity'] or self.loads[route_idx_2] - route_2_tasks_demand + route_1_tasks_demand > info['capacity']:
                    return False
                self.remove_task(route_idx_1, route_1_task_idx, len=len(route_1_tasks))
                self.remove_task(route_idx_2, route_2_task_idx, len=len(route_2_tasks))
                self.insert_task(route_idx_2, route_2_task_idx, route_1_tasks)
                self.insert_task(route_idx_1, route_1_task_idx, route_2_tasks)
            else: # split and connect in the reverse order
                route_1_tasks = self.routes[route_idx_1][route_1_task_idx:-1]
                route_2_tasks = self.routes[route_idx_2][1:route_2_task_idx + 1]
                route_1_tasks_demand = sum(info['demands'][task] for task in route_1_tasks)
                route_2_tasks_demand = sum(info['demands'][task] for task in route_2_tasks)
                if self.loads[route_idx_1] - route_1_tasks_demand + route_2_tasks_demand > info['capacity'] or self.loads[route_idx_2] - route_2_tasks_demand + route_1_tasks_demand > info['capacity']:
                    return False
                self.remove_task(route_idx_1, route_1_task_idx, len=len(route_1_tasks))
                self.remove_task(route_idx_2, 1, len=len(route_2_tasks))
                self.insert_task(route_idx_2, 1, route_1_tasks, reversed=True)
                self.insert_task(route_idx_1, route_1_task_idx, route_2_tasks, reversed=True)
            return True
        
        def merge_split():
            # a new operator introduced in MEANS
            # 1. randomly select a number of routes
            # 2. union the tasks in the selected routes as a new task list
            # 3. apply path scanning to the new task list, but generate only 1 route (by setting capacity to the sum of demands)
            # 4. split the new route into multiple routes with Ulusoy's splitting heuristic

            # route_num = np.random.randint(2, len(self.routes) + 1)
            route_num = 2 
            # In theory, the number of routes can be any number between 2 and len(self.routes)
            # The paper suggests that the number of routes should be 2 to avoid deteriorating the solution quality and computational time

            route_indices = np.random.choice(len(self.routes), route_num, replace=False)
            route_indices = sorted(route_indices)[::-1]
            tasks_to_merge = []
            for route_idx in route_indices:
                route = self.routes[route_idx]
                cost = 0
                for j in range(len(route) - 1):
                    cost += info['graph'][route[j][1]][route[j + 1][0]]
                for j in range(1, len(route) - 1):
                    cost += info['tasks'][(route[j][0], route[j][1])].cost
                tasks_to_merge.extend(route[1:-1])
                self.routes.pop(route_idx)
                self.loads.pop(route_idx)
                self.costs -= cost

            free_list = dict()
            sum_demands = 0
            for task in tasks_to_merge:
                free_list[(task[0], task[1])] = info['tasks'][(task[0], task[1])]
                free_list[(task[1], task[0])] = info['tasks'][(task[1], task[0])]
                sum_demands += info['demands'][(task[0], task[1])]
            
            min_cost = np.inf
            new_routes, new_loads, new_costs = [], [], 0
            for rule in rules:
                routes, _, _ = path_scanning(free_list, rule, capacity=sum_demands)
                ordered_tasks = routes[0][1:-1]

                routes, loads, costs = [], [], 0
                current_route, current_load, current_cost = [(info['depot'], info['depot'])], 0, 0
                while ordered_tasks:
                    if current_load + info['demands'][ordered_tasks[0]] <= info['capacity']:
                        current_route.append(ordered_tasks.pop(0))
                        current_cost += info['graph'][current_route[-2][1]][current_route[-1][0]] + info['tasks'][current_route[-1]].cost
                        current_load += info['demands'][current_route[-1]]
                    else:
                        current_cost += info['graph'][current_route[-1][1]][info['depot']]
                        current_route.append((info['depot'], info['depot']))
                        routes.append(current_route)
                        loads.append(current_load)
                        costs += current_cost
                        current_route, current_load, current_cost = [(info['depot'], info['depot'])], 0, 0
                current_cost += info['graph'][current_route[-1][1]][info['depot']]
                current_route.append((info['depot'], info['depot']))
                routes.append(current_route)
                loads.append(current_load)
                costs += current_cost

                if costs < min_cost:
                    min_cost = costs
                    new_routes, new_loads, new_costs = routes, loads, costs
            
            self.routes.extend(new_routes)
            self.loads.extend(new_loads)
            self.costs += new_costs
            return True

        mutation_weights= {
            flip: 0.3,
            single_insertion: 0.05,
            double_insertion: 0.05,
            swap: 0.2,
            two_opt_single: 0.2,
            two_opt_double: 0.2,
            merge_split: 0.1
        }

        mutation_max_trials = {
            flip: 1,
            single_insertion: 10,
            double_insertion: 10,
            swap: 20,
            two_opt_single: 1,
            two_opt_double: 20,
            merge_split: 1
        }

        operators = list(mutation_weights.keys())
        probabilities = np.array(list(mutation_weights.values()))
        probabilities /= probabilities.sum()

        mutation = np.random.choice(operators, p=probabilities)
        # if verbose:
        #     print('Mutation:', mutation.__name__)
        #     self.print()
        while True:
            if mutation():
                break
            else:
                if mutation_max_trials[mutation] == 0:
                    break
                mutation_max_trials[mutation] -= 1
        # if verbose:
        #     print('Mutation done')
        #     self.print()

def read_info(data_path):
    info = dict()
    tasks = dict()
    demands = dict()
    with open(data_path, 'r') as f:
        lines = f.readlines()
    info['name'] = lines[0].split(': ')[-1].rstrip('\n')
    info['vertices'] = int(lines[1].split(': ')[-1])
    info['depot'] = int(lines[2].split(': ')[-1])
    info['required edges'] = int(lines[3].split(': ')[-1])
    info['non-required edges'] = int(lines[4].split(': ')[-1])
    info['vehicles'] = int(lines[5].split(': ')[-1])
    info['capacity'] = int(lines[6].split(': ')[-1])
    info['total cost'] = int(lines[7].split(': ')[-1])
    graph = np.full((info['vertices'] + 1, info['vertices'] + 1), np.inf)
    np.fill_diagonal(graph, 0)
    for task in lines[9:9 + info['required edges'] + info['non-required edges']]:
        task = task.split()
        node_1 = int(task[0])
        node_2 = int(task[1])
        cost = int(task[2])
        demand = int(task[3])

        graph[node_1][node_2] = cost
        graph[node_2][node_1] = cost
        demands[(node_1, node_2)] = demand
        demands[(node_2, node_1)] = demand
        task_instance = Task(node_1, node_2, cost, demand)
        invert_task_instance = Task(node_2, node_1, cost, demand)
        if demand > 0:
            tasks[(node_1, node_2)] = task_instance
            tasks[(node_2, node_1)] = invert_task_instance
    info['graph'] = floyd_warshall(graph)
    info['tasks'] = tasks
    info['demands'] = demands
    return info
        
class Solver:
    def __init__(self, start, time_limit, seed):
        self.start = start
        self.time_limit = time_limit
        self.best_solution = None
        np.random.seed(seed)
    
    def simulated_annealing(self, solutions, initial_temperature = 1, final_temperature = 0.1, cooling_rate = 0.99):
        temperature = initial_temperature
        while temperature > final_temperature:
            for solution_idx in range(len(solutions)):
                new_solution = copy.deepcopy(solutions[solution_idx])
                new_solution.mutate()
                delta = new_solution.costs - solutions[solution_idx].costs
                if delta < 0:
                    solutions[solution_idx] = new_solution
                    if new_solution.costs < self.best_solution.costs:
                        self.best_solution = copy.deepcopy(new_solution)
                elif np.random.rand() < np.exp(-delta / temperature):
                    solutions[solution_idx] = new_solution
            temperature *= cooling_rate
            # if verbose:
            #     print('Temperature:', temperature)
            #     for i in range(len(solutions)):
            #         print(f'Solution {i + 1}:')
            #         solutions[i].print()
        return solutions


    def solve(self):
        solutions = []
        for rule in rules:
            routes, loads, costs = path_scanning(info['tasks'], rule, info['capacity'])
            solutions.append(Solution(routes, loads, costs))
        self.best_solution = min(solutions, key=lambda x: x.costs)
        iter = 1
        if self.best_solution.costs < 3000:
            initial_temperature = 1
        else:
            initial_temperature = 100
        time_begin_SA = time.time()
        while True:
            new_solutions = copy.deepcopy(solutions)
            new_solutions = self.simulated_annealing(new_solutions, initial_temperature=initial_temperature)
            solutions += new_solutions
            solutions.append(self.best_solution)
            solutions = sorted(solutions, key=lambda x: x.costs)[:len(rules)]
            
            # if verbose:
            #     print('Iteration:', iter)
            #     for i in range(len(solutions)):
            #         print(f'Solution {i + 1}:')
            #         solutions[i].print()
            #     print('Current best solutions:')
            #     min(solutions, key=lambda x: x.costs).print()
            #     print('Iter average time:', (time.time() - time_begin_SA) / iter)
                
            time_elapsed = time.time() - self.start
            avg_iter_time = (time.time() - time_begin_SA) / iter
            if time_elapsed + avg_iter_time * 1.05 > self.time_limit:
                break
            iter += 1

if __name__ == '__main__':
    start = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('data_path', type=str)
    argparser.add_argument('-t', '--time_limit', type=int, required=True)
    argparser.add_argument('-s', '--random_seed', type=int, required=True)
    args = argparser.parse_args()
    info = read_info(args.data_path)
    solver = Solver(start, args.time_limit, args.random_seed)
    solver.solve()
    solver.best_solution.print()
    # if verbose:
    #     print('Time elapsed: ', time.time() - start)
