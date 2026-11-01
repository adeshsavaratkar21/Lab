#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_NODES 1000  // Maximum nodes pool

typedef struct {
    int board[3][3];
    int x, y;          // Coordinates of blank space (0)
    int g, h, f;       // g(x), h(x), f(x)
    int parent;        // Index of parent node in array (-1 for root)
    bool visited;      // Explored flag
} Node;

Node nodes[MAX_NODES];
int nodeCount = 0;

int goal[3][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 0}
};

// Calculates Manhattan Distance heuristic h(x)
int calculateHeuristic(int board[3][3]) {
    int dist = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int val = board[i][j];
            if (val != 0) {
                int goalX = (val - 1) / 3;
                int goalY = (val - 1) % 3;
                dist += abs(i - goalX) + abs(j - goalY);
            }
        }
    }
    return dist;
}

bool isGoal(int board[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] != goal[i][j]) return false;
        }
    }
    return true;
}

bool areBoardsEqual(int b1[3][3], int b2[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (b1[i][j] != b2[i][j]) return false;
        }
    }
    return true;
}

bool isVisited(int board[3][3]) {
    for (int i = 0; i < nodeCount; i++) {
        if (areBoardsEqual(nodes[i].board, board)) {
            return true;
        }
    }
    return false;
}

// Selects the unvisited node with lowest f(x)
int getBestNodeIndex() {
    int minF = 999999;
    int bestIndex = -1;
    for (int i = 0; i < nodeCount; i++) {
        if (!nodes[i].visited && nodes[i].f < minF) {
            minF = nodes[i].f;
            bestIndex = i;
        }
    }
    return bestIndex;
}

void printBoard(int board[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 0) printf("_ ");
            else printf("%d ", board[i][j]);
        }
        printf("\n");
    }
}

// Backtracks from goal node to initial state using parent indices
void printPath(int index) {
    if (index == -1) return;

    printPath(nodes[index].parent);

    if (nodes[index].parent == -1) {
        printf("=== INITIAL STATE ===\n");
    } else if (isGoal(nodes[index].board)) {
        printf("=== FINAL / GOAL STATE (Step %d) ===\n", nodes[index].g);
    } else {
        printf("=== ACCEPTED STATE (Step %d) ===\n", nodes[index].g);
    }

    printBoard(nodes[index].board);
    printf("g(x) = %d, h(x) = %d, f(x) = %d\n", nodes[index].g, nodes[index].h, nodes[index].f);
    printf("-------------------------------\n\n");
}

void solve(int start[3][3]) {
    int blankX = 0, blankY = 0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (start[i][j] == 0) {
                blankX = i;
                blankY = j;
            }
        }
    }

    // Initialize root node
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            nodes[0].board[i][j] = start[i][j];

    nodes[0].x = blankX;
    nodes[0].y = blankY;
    nodes[0].g = 0;
    nodes[0].h = calculateHeuristic(start);
    nodes[0].f = nodes[0].g + nodes[0].h;
    nodes[0].parent = -1;
    nodes[0].visited = false;
    nodeCount = 1;

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (1) {
        int currIdx = getBestNodeIndex();

        if (currIdx == -1) {
            printf("No Solution Found.\n");
            return;
        }

        nodes[currIdx].visited = true;

        if (isGoal(nodes[currIdx].board)) {
            printf("\n=====================================\n");
            printf("       OPTIMAL SOLUTION PATH         \n");
            printf("=====================================\n\n");
            printPath(currIdx);
            return;
        }

        // Branch 4 directions: Up, Down, Left, Right
        for (int k = 0; k < 4; k++) {
            int nx = nodes[currIdx].x + dx[k];
            int ny = nodes[currIdx].y + dy[k];

            if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3) {
                int newBoard[3][3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        newBoard[i][j] = nodes[currIdx].board[i][j];

                // Swap blank tile
                newBoard[nodes[currIdx].x][nodes[currIdx].y] = newBoard[nx][ny];
                newBoard[nx][ny] = 0;

                if (!isVisited(newBoard) && nodeCount < MAX_NODES) {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            nodes[nodeCount].board[i][j] = newBoard[i][j];

                    nodes[nodeCount].x = nx;
                    nodes[nodeCount].y = ny;
                    nodes[nodeCount].g = nodes[currIdx].g + 1;
                    nodes[nodeCount].h = calculateHeuristic(newBoard);
                    nodes[nodeCount].f = nodes[nodeCount].g + nodes[nodeCount].h;
                    nodes[nodeCount].parent = currIdx;
                    nodes[nodeCount].visited = false;
                    nodeCount++;
                }
            }
        }
    }
}

int main() {
    int start[3][3] = {
        {1, 2, 3},
        {0, 4, 6},
        {7, 5, 8}
    };

    solve(start);
    return 0;
}