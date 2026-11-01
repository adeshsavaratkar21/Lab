/**
 * Experiment 1: Search Algorithms
 * File: dfs.c
 * Description: Water Jug Problem solved using Depth-First Search (DFS).
 *              Explores along each branch to deepest node using a stack.
 */

#include <stdio.h>
#define MAX 100

typedef struct {
    int x, y;
} State;

State stack[MAX];
int top = -1;
int visited[100][100];

void push(State s) {
    stack[++top] = s;
}

State pop() {
    return stack[top--];
}

int isEmpty() {
    return top == -1;
}

int main() {
    int cap1, cap2, goal, goalJug;
    printf("Enter capacity of Jug 1: ");
    scanf("%d", &cap1);
    printf("Enter capacity of Jug 2: ");
    scanf("%d", &cap2);
    printf("Enter goal amount: ");
    scanf("%d", &goal);
    printf("Enter goal jug (1 for Jug1, 2 for Jug2): ");
    scanf("%d", &goalJug);

    State start = {0, 0};
    push(start);

    printf("\nDFS Traversal\n");
    while (!isEmpty()) {
        State curr = pop();

        if (visited[curr.x][curr.y])
            continue;
        visited[curr.x][curr.y] = 1;

        printf("\nCurrent State: (%d,%d)\n", curr.x, curr.y);

        if ((goalJug == 1 && curr.x == goal) || (goalJug == 2 && curr.y == goal)) {
            printf("\nGoal Reached!\n");
            printf("Final State = (%d,%d)\n", curr.x, curr.y);
            return 0;
        }

        State next[6];
        int transfer;

        // 1. Fill Jug 1
        next[0].x = cap1; next[0].y = curr.y;
        // 2. Fill Jug 2
        next[1].x = curr.x; next[1].y = cap2;
        // 3. Empty Jug 1
        next[2].x = 0; next[2].y = curr.y;
        // 4. Empty Jug 2
        next[3].x = curr.x; next[3].y = 0;

        // 5. Pour Jug 1 -> Jug 2
        transfer = (cap2 - curr.y < curr.x) ? (cap2 - curr.y) : curr.x;
        next[4].x = curr.x - transfer;
        next[4].y = curr.y + transfer;

        // 6. Pour Jug 2 -> Jug 1
        transfer = (cap1 - curr.x < curr.y) ? (cap1 - curr.x) : curr.y;
        next[5].x = curr.x + transfer;
        next[5].y = curr.y - transfer;

        // Push reverse to explore in chronological branch order
        for (int i = 5; i >= 0; i--) {
            int a = next[i].x;
            int b = next[i].y;
            if (a >= 0 && a <= cap1 && b >= 0 && b <= cap2 && !visited[a][b]) {
                push(next[i]);
                printf(" Next State -> (%d,%d)\n", a, b);
            }
        }
    }
    printf("\nGoal cannot be achieved.\n");
    return 0;
}