#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<vector>
#include<map>
#include<cmath>
#include<stack>
using namespace std;

#define MAX_N 100000
int arr[MAX_N + 5];
int l[MAX_N + 5], r[MAX_N + 5];
int ltop = -1, rtop = -1;

int main() {
    int n, ans;
    stack<int> s;
    cin >> n;
    arr[0] = arr[n + 1] = -1;
    for (int i = 1; i <= n; i++) cin >> arr[i];
    for (int i = 1; i <= n; i++) {

    }

    return 0;
}