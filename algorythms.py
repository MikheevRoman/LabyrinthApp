from random import choice, random, randrange, shuffle
from typing import List, Callable, Tuple
from collections import deque
from math import inf
import heapq
import numpy as np


class Heuristic:
    """
    A class with basic heuristics for the A * algorithm
    """

    @staticmethod
    def octil(y0, x0, y1, x1):
        (ty, tx) = (abs(y0 - y1), abs(x0 - x1))
        return max(ty, tx) + (2 ** 0.5 - 1) * min(ty, tx)

    @staticmethod
    def manhattan(y0, x0, y1, x1):
        return abs(y0 - y1) + abs(x0 - x1)

    @staticmethod
    def chebyshev(y0, x0, y1, x1):
        return max(abs(y0 - y1), abs(x0 - x1))

    @staticmethod
    def euclidean(y0, x0, y1, x1):
        return ((y0 - y1) ** 2 + (x0 - x1) ** 2) ** 0.5


def ai_lab_solve(maze: List[List[bool]], beginNode: Tuple[int, int], endNode: Tuple[int, int]):
    pass


def bfs(maze: List[List[bool]], beginNode: Tuple[int, int], endNode: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Breadth-first search algorithm (iterative version)
    (!) This implementation uses a deque instead of a queue. This choice is only related to the speed.
    :param maze: 2D maze grid
    :param beginNode: initial position
    :param endNode: target position
    :return: path from initial to target position
    """
    current = (-1, -1)
    (y_max, x_max) = (len(maze), len(maze[0]))
    q = deque()
    q.append(endNode)
    directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
    parentNode = {endNode: endNode}
    while q:
        current = q.popleft()
        if current == beginNode:
            break
        for y_dir, x_dir in directions:
            (dy, dx) = (current[0] + y_dir, current[1] + x_dir)
            if dy >= y_max or dy < 0 or dx >= x_max or dx < 0 or maze[dy][dx]:
                continue
            neighbor = (dy, dx)
            if neighbor not in parentNode:
                q.append(neighbor)
                parentNode[neighbor] = current
    path = list()
    if current == beginNode:
        path.append(current)
        while current != endNode:
            current = parentNode[current]
            path.append(current)
    return path


def dijkstra(maze: List[List[bool]], beginNode: Tuple[int, int], endNode: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Dijkstra algorithm
    This implementation uses heapq - a priority queue. :param maze: 2D maze grid
    :param beginNode: initial position
    :param endNode: target position
    :return: path from initial to target position
    """

    current = (-1, -1)
    (y_max, x_max) = (len(maze), len(maze[0]))
    q = []
    heapq.heappush(q, (0, endNode))
    directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
    parentNode = {endNode: endNode}
    costMap = {endNode: 0}
    while q:
        current = heapq.heappop(q)[1]
        if current == beginNode:
            break
        cost = costMap[current] + 1
        for y_dir, x_dir in directions:
            (dy, dx) = (current[0] + y_dir, current[1] + x_dir)
            if dy >= y_max or dy < 0 or dx >= x_max or dx < 0 or maze[dy][dx]:
                continue
            neighbor = (dy, dx)
            if cost < costMap.get(neighbor, inf):
                costMap[neighbor] = cost
                parentNode[neighbor] = current
                heapq.heappush(q, (cost, neighbor))
    path = list()
    if current == beginNode:
        path.append(current)
        while current != endNode:
            current = parentNode[current]
            path.append(current)
    return path


# A-star
def a_star(maze: List[List[bool]], beginNode: Tuple[int, int], endNode: Tuple[int, int], heuristic: Callable = Heuristic.manhattan) -> List[Tuple[int, int]]:
    """
    A* algorithm
    This implementation uses heapq - a priority queue. :param maze: 2D maze grid
    :param beginNode: initial position
    :param endNode: target position
    :param heuristic: heuristic function
    :return: path from initial to target position
    """
    current = (-1, -1)
    (y_max, x_max) = (len(maze), len(maze[0]))
    q = []
    cost = 0
    heapq.heappush(q, (cost, endNode))
    directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
    parentNode = {endNode: endNode}
    costMap = {endNode: cost}
    while q:
        current = heapq.heappop(q)[1]
        if current == beginNode:
            break
        cost = costMap[current] + 1

        for y_dir, x_dir in directions:
            (dy, dx) = (current[0] + y_dir, current[1] + x_dir)
            if dy>=y_max or dy<0 or dx>=x_max or dx<0 or maze[dy][dx]:
                continue
            neighbor = (dy, dx)
            if cost < costMap.get(neighbor, inf):
                costMap[neighbor] = cost
                parentNode[neighbor] = current
                heapq.heappush(q, (cost + heuristic(current[0], current[1], beginNode[0], beginNode[1]), neighbor))
    path = list()
    if current == beginNode:
        path.append(current)
        while current != endNode:
            current = parentNode[current]
            path.append(current)
    return path


# PERFECT MAZES
# perfect maze - growing_tree
def growing_tree(y_max: int, x_max: int, backtrack_chance: float = 0.5) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = ((2, 0), (-2, 0), (0, 2), (0, -2))
    current_row, current_col = (randrange(1, y_max, 2), randrange(1, x_max, 2))
    grid[current_row][current_col] = False
    active = [(current_row, current_col)]
    while active:
        if random() < backtrack_chance:
            current_row, current_col = active[-1]
        else:
            current_row, current_col = choice(active)
        neighbors = ((current_row + dy, current_col + dx) for dy, dx in directions)
        neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max and grid[y][x]]
        if not neighbors:
            active = [a for a in active if a != (current_row, current_col)]
            continue
        nn_row, nn_col = choice(neighbors)
        active += [(nn_row, nn_col)]
        grid[nn_row][nn_col] = False
        grid[(current_row + nn_row) // 2][(current_col + nn_col) // 2] = False
    return grid


# Aldous-Broder 2D perfect maze
def aldous_broder(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = ((2, 0), (-2, 0), (0, 2), (0, -2))
    (current_row, current_col) = (randrange(1, y_max, 2), randrange(1, x_max, 2))
    grid[current_row][current_col] = False
    num_visited = 1
    max_visited = ((y_max - 1) // 2 * (x_max - 1) // 2)
    while num_visited < max_visited:
        neighbors = [(current_row + dy, current_col + dx) for dy, dx in directions]
        valid_neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max and grid[y][x]]
        if not valid_neighbors:
            free_neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max and not grid[y][x]]
            (current_row, current_col) = choice(free_neighbors)
            continue
        shuffle(valid_neighbors)
        for new_row, new_col in valid_neighbors:
            if grid[new_row][new_col]:
                grid[new_row][new_col] = grid[(new_row + current_row) // 2][(new_col + current_col) // 2] = False
                (current_row, current_col) = (new_row, new_col)
                num_visited += 1
                break
    return grid


# Wilson's 2D perfect maze
def wilson(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = ((2, 0), (-2, 0), (0, 2), (0, -2))
    free = {(y, x) for y in range(1, y_max, 2) for x in range(1, x_max, 2)}
    (y, x) = (2 * randrange(y_max // 2) + 1, 2 * randrange(x_max // 2) + 1)
    grid[y][x] = False
    free.remove((y, x))
    while free:
        y, x = key = choice(tuple(free))
        free.remove(key)
        path = [key]
        grid[y][x] = False
        neighbors = ((y + dy, x + dx) for dy, dx in directions)
        neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max]
        y, x = key = choice(neighbors)
        while grid[y][x]:
            grid[y][x] = grid[(y + path[-1][0]) // 2][(x + path[-1][1]) // 2] = False
            neighbors = ((y + dy, x + dx) for dy, dx in directions)
            neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max]
            neighbors.remove(path[-1])
            free.remove(key)
            path.append(key)
            y, x = key = choice(neighbors)
        if key in path:
            last_key = path.pop()
            free.add(last_key)
            grid[last_key[0]][last_key[1]] = True
            for key in reversed(path):
                free.add(key)
                grid[key[0]][key[1]] = grid[(last_key[0] + key[0]) // 2][(last_key[1] + key[1]) // 2] = True
                last_key = key
        else:
            grid[(y + path[-1][0]) // 2][(x + path[-1][1]) // 2] = False
    return grid


# Iterative version of depth-first search 2D perfect maze
def backtracking(y_max: int, x_max: int) -> List[List[bool]]:
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    stack = [(2 * randrange(y_max // 2) + 1, 2 * randrange(x_max // 2) + 1)]
    while stack:
        y, x = stack.pop()
        grid[y][x] = False
        neighbors = ((y + dy, x + dx) for dy, dx in directions)
        neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max and grid[y][x]]
        if len(neighbors) > 1:
            stack.append((y, x))
        if neighbors:
            ny, nx = choice(neighbors)
            grid[(y + ny) // 2][(x + nx) // 2] = False
            stack.append((ny, nx))
    return grid


# Binary tree 2D perfect maze generation
def binary_tree(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    grid[1][1] = False
    for x in range(3, x_max, 2):
        grid[1][x] = grid[1][x - 1] = False
    for y in range(3, y_max, 2):
        grid[y][1] = grid[y - 1][1] = False
        for x in range(3, x_max, 2):
            if randrange(2):
                grid[y][x] = grid[y][x - 1] = False
            else:
                grid[y][x] = grid[y - 1][x] = False
    return grid


# Division 2D perfect maze generation
def division(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    horizontal, vertical = 1, 0
    grid = [[False for _ in range(x_max)] for _ in range(y_max)]
    for i in range(len(grid[0])):
        grid[0][i] = grid[-1][i] = True
    for i in range(len(grid)):
        grid[i][0] = grid[i][-1] = True
    region_stack = [((1, 1), (y_max - 2, x_max - 2))]
    while region_stack:
        current_region = region_stack[-1]
        region_stack.pop()
        ((min_y, min_x), (max_y, max_x)) = current_region
        (height, width) = (max_y - min_y + 1, max_x - min_x + 1)
        if height <= 1 or width <= 1:
            continue
        if width < height:
            cut_direction = horizontal
        elif width > height:
            cut_direction = vertical
        else:
            if width == 2:
                continue

            cut_direction = randrange(2)
        cut_length = (height, width)[(cut_direction + 1) % 2]
        if cut_length < 3:
            continue
        cut_pos = randrange(1, cut_length, 2)
        door_pos = randrange(0, (height, width)[cut_direction], 2)
        if cut_direction == vertical:
            for row in range(min_y, max_y + 1):
                grid[row][min_x + cut_pos] = True
            grid[min_y + door_pos][min_x + cut_pos] = False
        else:
            for col in range(min_x, max_x + 1):
                grid[min_y + cut_pos][col] = True
            grid[min_y + cut_pos][min_x + door_pos] = False
        if cut_direction == vertical:
            region_stack.append(((min_y, min_x), (max_y, min_x + cut_pos - 1)))
            region_stack.append(((min_y, min_x + cut_pos + 1), (max_y, max_x)))
        else:
            region_stack.append(((min_y, min_x), (min_y + cut_pos - 1, max_x)))
            region_stack.append(((min_y + cut_pos + 1, min_x), (max_y, max_x)))
    return grid


# Eller's 2D perfect maze generation
def eller(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    parent = {x: {x} for x in range(1, x_max, 2)}
    for y in range(1, y_max - 2, 2):
        grid[y][1] = False
        for x in range(3, x_max, 2):
            if x not in parent[x - 2] and randrange(2):
                parent[x].update(parent[x - 2])
                for key in list(parent[x - 2]):
                    parent[key] = parent[x]
                grid[y][x - 1] = grid[y][x] = False
            else:
                grid[y][x] = False
        for members in {frozenset(x) for x in parent.values()}:
            walls = [list(), list()]
            for x in members:
                walls[randrange(2)].append(x)
            if not walls[0]:
                walls.reverse()
            for x in walls[0]:
                grid[y + 1][x] = False
            for x in walls[1]:
                for key in parent:
                    parent[key].discard(x)
                parent[x] = {x}
    y = y_max - 2
    grid[y][1] = False
    for x in range(3, x_max, 2):
        if x not in parent[x - 2]:
            parent[x].update(parent[x - 2])
            for key in list(parent[x - 2]):
                parent[key] = parent[x]
            grid[y][x - 1] = grid[y][x] = False
        else:
            grid[y][x] = False
    return grid


def kruskal(y_max: int, x_max: int) -> List[List[bool]]:
    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    parent = {(y, x): (y, x) for y in range(1, y_max, 2) for x in range(1, x_max, 2)}

    def find(x):
        temp = x[:]
        while parent[temp] != temp:
            temp = parent[temp]
        return temp
    walls = [(1, x) for x in range(2, x_max - 1, 2)]
    for y in range(2, y_max - 2, 2):
        walls.extend((y, x) for x in range(1, x_max, 2))
        y += 1
        walls.extend((y, x) for x in range(2, x_max - 1, 2))
    shuffle(walls)
    for y, x in walls:
        if y % 2:
            coord1 = (y, x + 1)
            coord2 = (y, x - 1)
        else:
            coord1 = (y + 1, x)
            coord2 = (y - 1, x)
        if find(coord1) != find(coord2):
            grid[y][x] = grid[coord1[0]][coord1[1]] = grid[coord2[0]][coord2[1]] = False
            parent[find(coord1)] = find(coord2)
    return grid


def modified_prim(y_max: int, x_max: int) -> List[List[bool]]:
    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = ((2, 0), (-2, 0), (0, 2), (0, -2))
    (y, x) = (2 * randrange(y_max // 2) + 1, 2 * randrange(x_max // 2) + 1)
    grid[y][x] = False
    cells = ((y + dy, x + dx) for dy, dx in directions)
    cells = {(y, x) for y, x in cells if 0 < y < y_max and 0 < x < x_max}
    while cells:
        y, x = choice(tuple(cells))
        cells.remove((y, x))
        neighbors = ((y + dy, x + dx) for dy, dx in directions)
        neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max]
        ny, nx = choice([(y, x) for y, x in neighbors if not grid[y][x]])
        grid[y][x] = grid[(ny + y) // 2][(nx + x) // 2] = False
        cells.update(((y, x) for y, x in neighbors if grid[y][x]))
    return grid


# Sidewinder 2D perfect maze
def sidewinder(y_max: int, x_max: int, skew: float = 0.5) -> List[List[bool]]:
    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    for x in range(1, x_max - 1):
        grid[1][x] = False
    for y in range(3, y_max, 2):
        run = []
        for x in range(1, x_max, 2):
            grid[y][x] = False
            run.append((y, x))
            carve_east = (random() >= skew)
            if carve_east and x < (x_max - 2):
                grid[y][x + 1] = False
            else:
                north = choice(run)
                grid[north[0] - 1][north[1]] = False
                run = []
    return grid


# IMPERFECT MAZES
# Serpentine 2D imperfect maze
def serpentine(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    if randrange(2):
        for row in range(1, y_max - 1):
            grid[row][randrange(1, x_max - 1)] = False
            for col in range(1, x_max - 1, 2):
                grid[row][col] = False
        for col in range(2, x_max - 1, 4):
            grid[1][col] = False
        for col in range(4, x_max - 1, 4):
            grid[y_max - 2][col] = False
    else:
        for row in range(1, y_max - 1, 2):
            for col in range(1, x_max - 1):
                grid[row][col] = False
        for row in range(2, y_max - 1, 4):
            grid[row][1] = False
            grid[row][randrange(2, x_max - 1)] = False
        for row in range(4, y_max - 1, 4):
            grid[row][x_max - 2] = False
            grid[row][randrange(1, x_max - 2)] = False
    return grid


# Small rooms 2D imperfect maze
def small_rooms(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    for y in range(1, y_max - 1):
        if y % 2:
            for x in range(1, x_max - 3, 4):
                grid[y][x] = False
                grid[y][x + 1] = False
                grid[y][x + 2] = False
                if x < x_max - 4 and not randrange(3):
                    grid[y][x + 3] = False
        else:
            for x in range(2, x_max - 1, 4):
                grid[y][x] = False
    y_mid = y_max // 2
    for x in range(1, x_max - 1):
        grid[y_mid][x] = False
    return grid


# Spiral 2D imperfect maze
def spiral(y_max: int, x_max: int) -> List[List[bool]]:

    # assert y_max % 2 and x_max % 2 and y_max >= 3 and x_max >= 3
    grid = [[True for _ in range(x_max)] for _ in range(y_max)]
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    if randrange(2):
        directions.reverse()
    current = (1, 1)
    grid[1][1] = False
    next_dir = 0
    while True:
        y, x = current
        new_y, new_x = (y + directions[next_dir][0], x + directions[next_dir][1])
        neighbors = ((y + dy, x + dx) for dy, dx in directions)
        neighbors = [(y, x) for y, x in neighbors if 0 < y < y_max and 0 < x < x_max if grid[y][x]]
        if (new_y, new_x) in neighbors:
            grid[(y + new_y) // 2][(x + new_x) // 2] = False
            grid[new_y][new_x] = False
            current = (new_y, new_x)
        elif not neighbors:
            break
        else:
            next_dir = (next_dir + 1) % 4
    for i in range(max(y_max, x_max)):
        grid[randrange(1, y_max - 1)][randrange(1, x_max - 1)] = False
    return grid


# количество тупиков в лабиринте
def num_of_dead_ends(maze):
    """
    :param maze: лабиринт
    :return: количество тупиков
    """
    dead_ends = 0
    for row in range(1, len(maze)-1):
        for block in range(1, len(maze[row])-1):
            count = 0
            if not maze[row+1][block]:
                count += 1
            if not maze[row-1][block]:
                count += 1
            if not maze[row][block+1]:
                count += 1
            if not maze[row][block-1]:
                count += 1
            if count == 3:
                dead_ends += 1
    return dead_ends - 1


# преобразование лабиринта в матрицу смежности
def transform_to_adjacency_table(maze: List[List[bool]]):
    weight = len(maze[0])
    height = len(maze)
    len_adjacency_table = weight * height
    adjacency_table = list(list(np.inf for j in range(len_adjacency_table)) for i in range(len_adjacency_table))
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if not maze[i][j]:
                adjacency_table[i * weight + j][i * weight + j] = 0
                if i - 1 >= 0:
                    if not maze[i - 1][j]:
                        adjacency_table[i * weight + j][(i - 1) * weight + j] = 1
                if i + 1 < height:
                    if not maze[i + 1][j]:
                        adjacency_table[i * weight + j][(i + 1) * weight + j] = 1
                if j - 1 >= 0:
                    if not maze[i][j - 1]:
                        adjacency_table[i * weight + j][i * weight + (j - 1)] = 1
                if j + 1 < weight:
                    if not maze[i][j + 1]:
                        adjacency_table[i * weight + j][i * weight + (j + 1)] = 1
    print(adjacency_table)
    return adjacency_table


def convert_adjacency_matrix_for_vertexes(adjacency_matrix: List[List]):
    edges = [tuple]
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[0])):
            if adjacency_matrix[i][j]:
                edge = (i, j)
                edges.append(edge)
    return edges
