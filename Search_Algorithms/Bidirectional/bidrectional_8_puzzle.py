# Bidirectional Algorithm
BOARD = [[1, 8, 2],
         [0, 4, 3],
         [7, 6, 5]]
# BOARD = [[4, 3, 0],  # Alternative
#          [7, 1, 2],
#          [8, 6, 5]]
FINAL_BOARD = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 0]]

open_space = (0, 2)

print("START_BOARD:", BOARD)
print("FINAL_BOARD:", FINAL_BOARD)
print('Searching...')
path = []
# Check the path


def extract_path(visited, parent, final, start, reverse):
    p = final
    path = []
    path.append(p)
    while p != start:
        p = parent[visited.index(p)]
        if p not in path:
            path.append(p)
    if reverse:
        path.reverse()
    return path


# Find the open space in the puzzle
def find_open_space(n):
    global open_space
    for i in range(len(n)):
        for j in range(len(n[i])):
            if(n[i][j] == 0):
                open_space = (j, i)
                break


# Look for possible next steps
def neighbors(n):
    res = []
    x = open_space[0]
    y = open_space[1]
    options = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    for option in options:
        board_copy = list(map(list, n))
        if (option[0] in [0, 1, 2] and option[1] in [0, 1, 2]):
            replaced_number = board_copy[option[1]][option[0]]
            board_copy[option[1]][option[0]] = 0
            board_copy[y][x] = replaced_number
            res.append(board_copy)
    return res


def bs(board, final_board):
    visited_fb = []
    visited_bf = []

    actions_fb = [0 for _ in range(181440)]
    actions_bf = [0 for _ in range(181440)]

    queue_fb = [board]  # Add the initial cell into the queue
    queue_bf = [final_board]  # Add the initial cell into the queue

    parent_fb = [0 for _ in range(181440)]
    parent_bf = [0 for _ in range(181440)]
    # Front to back
    while queue_fb and queue_bf:
        if queue_fb:
            n = queue_fb.pop(0)
            find_open_space(n)
            for node in neighbors(n):
                if node not in visited_fb:
                    # Add to visited and count depth
                    visited_fb.append(node)
                    if(n not in visited_fb):
                        visited_fb.append(n)
                    actions_fb[visited_fb.index(
                        node)] = actions_fb[visited_fb.index(n)] + 1
                    parent_fb[visited_fb.index(node)] = n
                    queue_fb.append(node)
        # Back to front
        if queue_bf:
            n = queue_bf.pop(0)
            find_open_space(n)
            for node in neighbors(n):
                if node not in visited_bf:
                    # Add to visited and count depth
                    visited_bf.append(node)
                    if(n not in visited_bf):
                        visited_bf.append(n)
                    actions_bf[visited_bf.index(
                        node)] = actions_bf[visited_bf.index(n)] + 1
                    parent_bf[visited_bf.index(node)] = n
                    queue_bf.append(node)
                    # Check if middle has been found
                    if node in visited_fb:
                        print(
                            f"IDS / Depth of solution: {actions_fb[visited_fb.index(node)]+actions_bf[visited_bf.index(node)]}")
                        path_fb = extract_path(
                            visited_fb, parent_fb, node, BOARD, True)
                        path_bf = extract_path(
                            visited_bf, parent_bf, node, FINAL_BOARD, False)
                        print('Path:')
                        path_bf.pop(0)  # Pop the duplicate center
                        path = path_fb + path_bf
                        print(*path, sep='\n')
                        queue_bf.clear()
                        queue_fb.clear()
                        break


bs(BOARD, FINAL_BOARD)
