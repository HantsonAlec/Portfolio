#BFS
def neighbors(n, states):
    return states[n]

def bfs(states):
    NOT_VISITED, VISITED = 0, 1
    FINAL_NODE = 25
    visited = [NOT_VISITED for _ in range(26)]
    actions = [0 for _ in range(26)]
    queue = [0] #Add the initial cell into the queue
    
    while queue:
        n = queue.pop(0)
        print("\tVisiting ", n)
        #visited[n] = VISITED
        for node in neighbors(n, states):
            if visited[node] == NOT_VISITED:
                visited[node] = VISITED
                actions[node] = actions[n] + 1  # Count the number of actions
                queue.append(node)
                if node == FINAL_NODE:
                    # I'm done!
                    # Display the number of actions
                    print(actions)
                    print("Minimum number of actions ", actions[node])
                    # Finish BFS
                    queue.clear()


#Search Space (Graph)
ROLL_1, ROLL_2, ROLL_3, ROLL_4, ROLL_5, ROLL_6 = 0, 1, 2, 3, 4, 5
#Basic actions
states = [0 for _ in range(26)]
for i in range(0, 25):
    states[i] = [i+1, i+2, i+3, i+4, i+5, i+6]
#Ladders
# Ladder from 2->13
states[0][ROLL_2] = 13
states[1][ROLL_1] = 13
# Ladder from 6->17
states[0][ROLL_6] = 17
states[1][ROLL_5] = 17
states[2][ROLL_4] = 17
states[3][ROLL_3] = 17
states[4][ROLL_2] = 17
states[5][ROLL_1] = 17
# Ladder from 12->23
states[7][ROLL_5] = 23
states[8][ROLL_4] = 23
states[9][ROLL_3] = 23
states[10][ROLL_2] = 23
states[11][ROLL_1] = 23
#Snakes
# Snake from 10->1
states[4][ROLL_6] = 1
states[5][ROLL_5] = 1
# states[6][ROLL_4] = 1 This is not possible
states[7][ROLL_3] = 1
states[8][ROLL_2] = 1
states[9][ROLL_1] = 1
# Snake from 14->8
states[8][ROLL_6] = 8
states[9][ROLL_5] = 8
states[10][ROLL_4] = 8
states[11][ROLL_3] = 8
# states[12][ROLL_2] = 8 This is not possible
states[13][ROLL_1] = 8

##############################################
bfs(states)