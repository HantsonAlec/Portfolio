import numpy as np
import matplotlib.pyplot as plt
NOT_VISITED, VISITED, IN_PROGRESS = 0, 1, 2
EXIT = (12, 18)
START = (1, 1)


def extract_path(parent):
    p = EXIT
    path = set()
    path.add(p)
    while p != START:
        p = parent[p[0]][p[1]]
        path.add(p)

    return path


def print_trace(graph, visited, parent):
    path = extract_path(parent)
    rows = len(graph)
    cols = len(graph[0])
    for i in range(rows):
        for j in range(cols):
            if (i, j) in path:
                print("@", end="")
            else:
                print(graph[i][j], end="")
        print()


def neighbors(graph, pos):
    x = pos[0]
    y = pos[1]
    ns = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    res = []
    # Only the neighbors where there is a free space
    for n in ns:
        if graph[n[0]][n[1]] == " ":
            res.append(n)
    return res


def dfs(graph):
    rows = len(graph)
    cols = len(graph[0])
    visited = [[NOT_VISITED for _ in range(cols)] for _ in range(rows)]
    parent = [[None for _ in range(cols)] for _ in range(rows)]
    visited[START[0]][START[1]] = IN_PROGRESS
    stack = [START]
    stack_size = []

    while stack:
        n = stack.pop()
        stack_size.append(len(stack))
        for node in neighbors(graph, n):
            if visited[node[0]][node[1]] == NOT_VISITED:
                visited[node[0]][node[1]] = IN_PROGRESS
                parent[node[0]][node[1]] = n  # Set parent to previous node
                # Append node to stack (This one will be checked next)
                stack.append(node)
                if node == EXIT:  # Check if finished
                    stack.clear()
                    print(extract_path(parent))
                    print_trace(graph, visited, parent)
                    break

    return stack_size


# labyrinth = [
# "********************",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "*                  *",
# "********************",
# ]
labyrinth = [
    "********************",
    "*     ** **     ** *",
    "*****    ******    *",
    "**    ** ***    ** *",
    "***** ********* ****",
    "*****    ******    *",
    "******** ****** ****",
    "******** ****** ****",
    "*     ** **     ** *",
    "*****    ******    *",
    "**    ** ***    ** *",
    "***** ********* ****",
    "*****              *",
    "********************"
]

dfs(labyrinth)
