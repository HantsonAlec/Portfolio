from colorama import Fore

NOT_VISITED, VISITED = 0, 1
EXIT = (21, 81)
# EXIT = (12, 40)
START = (1, 0)
iterations = 0
steps = 0
depth = 0
visited = []
parent = []
children = []


def extract_path(parent):
    p = EXIT
    path = set()
    path.add(p)
    while p != START:
        p = parent[p[0]][p[1]]
        path.add(p)
    return path


# Pad tonen in console
def print_trace(maze, parent):
    global steps
    path = extract_path(parent)
    steps = len(path)
    rows = len(maze)
    cols = len(maze[0])
    for i in range(rows):
        for j in range(cols):
            if (i, j) in path:
                print(Fore.RED + "@", end="")
            else:
                print(Fore.WHITE + maze[i][j], end="")
        print()


# Getting al valid neighbors
def neighbors(maze, pos):
    x, y = pos[0], pos[1]
    ns = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    valid_ns = []
    for n in ns:
        if maze[n[0]][n[1]] == " ":
            valid_ns.append(n)
    return valid_ns


def IDS(board, limit):
    global visited, parent
    found = False
    maze = board
    rows = len(labyrinth)
    cols = len(labyrinth[0])
    visited = [[NOT_VISITED for _ in range(cols)] for _ in range(rows)]
    visited[START[0]][START[1]] = VISITED
    parent = [[None for _ in range(cols)] for _ in range(rows)]
    stack = [START]
    # Go to every depth
    for i in range(0, limit):
        if ids(maze, stack, i):
            print(Fore.GREEN + f"IDS / Depth of solution: {i}")
            found = True
            break
    if not found:
        print(Fore.GREEN + f"Not found within depth limit {limit}")


def ids(maze, stack, limit):
    global depth, iterations, visited, parent
    while stack:
        # Check if reached limit
        if depth <= limit:
            n = stack.pop()
            depth += 1
            # Check every neighbour
            for node in neighbors(maze, n):
                if visited[node[0]][node[1]] == NOT_VISITED:
                    visited[node[0]][node[1]] = VISITED
                    parent[node[0]][node[1]] = n
                    children.append(node)
                    iterations += 1
                    if node == EXIT:  # Check if finished
                        stack.clear()
                        # Show route
                        print_trace(maze, parent)
                        return True
            # If stack empty append firsy child
            if len(stack) == 0:
                for child in children:
                    stack.append(child)
                children.clear()
        else:
            depth = 0
            return False


# * is a wall. @ is the movement of the player
# labyrinth = [
#     "*****************************************",
#     "         **        ***         **********",
#     "*******      *******    ***  ************",
#     "**                    *****     *********",
#     "***** *************  *********    *******",
#     "*****    ***********  **     * ***      *",
#     "******** ****** ****  ****     *   *** **",
#     "******** ************  *   **    ***** **",
#     "*     **        ******  *  *********** **",
#     "****     ****** *******                 *",
#     "**    ** ***    ********  ***  **********",
#     "***  *****************   **********  ****",
#     "***         *                            ",
#     "*****************************************"
# ]
labyrinth = [
    "**********************************************************************************",
    "         **        ***         **********         **        ***         **********",
    "*******      *******    ***  *******************      *******    ***  ************",
    "**                    *****     *******                        *****     *********",
    "***** *************  *********    ***     **** *************  *********    *******",
    "*****    ************ **     * ***      ******    ************ **     * ***      *",
    "******** ****** ****  ****     *   *** ********** ****** ****  ****     *   *** **",
    "******** ************  *   **    ***** ********** ************  *   **    ***** **",
    "*     **        ******  *  *********** ***     **        ******  *  *********** **",
    "****     ****** *******            *** ******     ****** *******            *** **",
    "**    ** ***    ********  ***  **      ****    ** ***    ********  ***  **      **",
    "***  *****************   ****  ****  *******  *********  ******   ****  ****  ****",
    "***              *     *         **             ********         ********     * **",
    "***       *** ***     ************    *    ** ******  ** ***********      *****  *",
    "***           *******            * **********        *** ****    *     *         *",
    "***    ***** *********   ****  ****  *******  ********** ******   **********  ****",
    "***  ******* *******     *     *         **  ************        *****  **** *****",
    "***        *    ***              ********         **              *     *        *",
    "*****    *****  ***** **     * ***      ******    ************ **     * **********",
    "***             ********         *******         **        ***    *              *",
    "***       ********               *     *         **                    ********  *",
    "*******                 ***  ******************       *******    ***  *******     ",
    "**********************************************************************************"
]
IDS(labyrinth, 39)

print(f"Iterations (elke bekeken mogelijkheid): {iterations}")
print(f"Aantal stappen in oplossing: {steps}")
