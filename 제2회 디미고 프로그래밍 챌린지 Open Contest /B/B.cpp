#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    long long  n,m;
    long long  sum = 0;
    cin >> n >> m;
    for (long long  i=0;i<n;i++){
        long long  x;
        cin >> x;
        sum += (x-1);      
    }
    if (sum < m) {
        cout << "OUT";
    } else {
        cout << "DIMI";
    }

    return 0;
}
