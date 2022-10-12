#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<vector>
#include<map>
#include<cmath>
#include<queue>
using namespace std;

# define MAX_N 300000

int q[MAX_N + 5];
long long arr[MAX_N + 5];
long long ans = 0;
int head = 0, tail = 0;

int main() {
    long long n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> arr[i], arr[i] += arr[i - 1];
    q[tail++] = 0;
    ans = arr[1];
    for (int i = 1; i <= n; i++) {
        ans = max(ans, arr[i] - arr[q[head]]);
        while (tail - head && arr[q[tail - 1]] >= arr[i]) tail--;
        q[tail++] = i;
        if (q[head] == i - m) head++;
    }
    cout << ans << endl;

    return 0;
}