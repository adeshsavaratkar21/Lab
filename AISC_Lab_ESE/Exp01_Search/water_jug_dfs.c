/**
 * Experiment 1: Uninformed & Informed Search Algorithms
 * File: water_jug_dfs.c
 * Description: Water Jug Problem solved using Depth-First Search (DFS) with backtracking.
 *              DFS explores deep into state space until goal is reached or branch terminates.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_DEPTH 1000

typedef struct {
    int x;
    int y;
    char action[64];
} Step;

Step path[MAX_DEPTH];
bool visited[500][500];
int cap1 = 4, cap2 = 3, goal = 2;
bool solution_found = false;

int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

void print_solution(int depth) {
    printf("\n=======================================================\n");
    printf("DFS Solution Path Found (Total Steps: %d):\n", depth);
    printf("=======================================================\n");
    printf("%-6s | %-28s | %-12s\n", "Step", "Action Taken", "State (J1, J2)");
    printf("-------------------------------------------------------\n");
    for (int i = 0; i <= depth; i++) {
        printf("%-6d | %-28s | (%d, %d)\n", i, path[i].action, path[i].x, path[i].y);
    }
    printf("=======================================================\n");
}

bool dfs(int x, int y, int depth, const char *action) {
    if (depth >= MAX_DEPTH) {
        return false;
    }

    path[depth].x = x;
    path[depth].y = y;
    strncpy(path[depth].action, action, sizeof(path[depth].action) - 1);
    path[depth].action[sizeof(path[depth].action) - 1] = '\0';
    visited[x][y] = true;

    // Check goal condition
    if (x == goal || y == goal) {
        print_solution(depth);
        solution_found = true;
        return true;
    }

    // Define next 6 possible operations
    struct {
        int nx, ny;
        char desc[64];
    } next_moves[6];

    // 1. Fill Jug 1
    next_moves[0].nx = cap1;
    next_moves[0].ny = y;
    snprintf(next_moves[0].desc, sizeof(next_moves[0].desc), "Fill Jug 1");

    // 2. Fill Jug 2
    next_moves[1].nx = x;
    next_moves[1].ny = cap2;
    snprintf(next_moves[1].desc, sizeof(next_moves[1].desc), "Fill Jug 2");

    // 3. Empty Jug 1
    next_moves[2].nx = 0;
    next_moves[2].ny = y;
    snprintf(next_moves[2].desc, sizeof(next_moves[2].desc), "Empty Jug 1");

    // 4. Empty Jug 2
    next_moves[3].nx = x;
    next_moves[3].ny = 0;
    snprintf(next_moves[3].desc, sizeof(next_moves[3].desc), "Empty Jug 2");

    // 5. Pour Jug 1 -> Jug 2
    int pour12 = (x + y > cap2) ? (cap2 - y) : x;
    next_moves[4].nx = x - pour12;
    next_moves[4].ny = y + pour12;
    snprintf(next_moves[4].desc, sizeof(next_moves[4].desc), "Pour Jug 1 -> Jug 2 (%dL)", pour12);

    // 6. Pour Jug 2 -> Jug 1
    int pour21 = (x + y > cap1) ? (cap1 - x) : y;
    next_moves[5].nx = x + pour21;
    next_moves[5].ny = y - pour21;
    snprintf(next_moves[5].desc, sizeof(next_moves[5].desc), "Pour Jug 2 -> Jug 1 (%dL)", pour21);

    for (int i = 0; i < 6; i++) {
        int nx = next_moves[i].nx;
        int ny = next_moves[i].ny;

        if (nx >= 0 && nx <= cap1 && ny >= 0 && ny <= cap2) {
            if (!visited[nx][ny]) {
                if (dfs(nx, ny, depth + 1, next_moves[i].desc)) {
                    return true;
                }
            }
        }
    }

    return false;
}

int main(void) {
    int choice = 0;

    printf("=======================================================\n");
    printf("      WATER JUG PROBLEM SOLVER USING DFS (AISC)        \n");
    printf("=======================================================\n");
    printf("Choose Input Mode:\n");
    printf("1. Use default values (Jug 1: 4L, Jug 2: 3L, Goal: 2L)\n");
    printf("2. Enter custom values\n");
    printf("Enter choice (1 or 2): ");
    if (scanf("%d", &choice) != 1 || choice != 2) {
        printf("Using defaults: Jug 1 = %d, Jug 2 = %d, Goal = %d\n", cap1, cap2, goal);
    } else {
        printf("Enter capacity of Jug 1: ");
        scanf("%d", &cap1);
        printf("Enter capacity of Jug 2: ");
        scanf("%d", &cap2);
        printf("Enter target goal amount: ");
        scanf("%d", &goal);
    }

    if (goal > cap1 && goal > cap2) {
        printf("Error: Goal cannot exceed capacity of both jugs!\n");
        return 1;
    }
    if (goal % gcd(cap1, cap2) != 0) {
        printf("Error: Goal cannot be measured according to Bezout's identity!\n");
        return 1;
    }

    memset(visited, 0, sizeof(visited));
    solution_found = false;

    if (!dfs(0, 0, 0, "Initial State")) {
        printf("No solution exists to reach %dL.\n", goal);
    }

    return 0;
}
