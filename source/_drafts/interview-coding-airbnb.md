---
title: 【面经】Airbnb部分编程算法题
date: 2019-03-18 22:10:10
tags: [算法, 总结, 面经]
---

> 本文收集网络上部分编程面经以供复习
<!-- more -->
> 如果想要寻找测试用例可以参考[git题库](https://github.com/allaboutjst/airbnb)
> 本文参考了[参考资料1](https://yezizp2012.github.io/2017/06/01/airbnb%E9%9D%A2%E8%AF%95%E9%A2%98%E6%B1%87%E6%80%BB/)

## 数字分组
```
输入：一个数组，每一位表示角标值所在组号
输出：一个数组，同一分组在一起，整个数组保证字典序最小
```

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

bool cmp(const vector<int>& a, const vector<int>& b) {
    return a[0] < b[0];
}
vector<int> helper(vector<int>& nums) {
    int n = nums.size();
    unordered_map<int, vector<int>> m;
    for (int i = 0; i < n; i++) {
        m[nums[i]].push_back(i);
    }
    vector<vector<int>> ans;
    for (auto i : m) {
        ans.push_back(i.second);
    }
    sort(ans.begin(), ans.end(), cmp);
    
    vector<int> ret;
    for (auto i : ans) {
        for (auto j : i) {
            ret.push_back(j);
        }
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    for (auto i : helper(nums)) {
        cout << i << endl;
    }
    return 0;
}
```

## 吃奶酪
```
题目：一个二维的数组，数组中有三个值，0,1,2，分别代表可走，不可走和奶酪，老鼠杰瑞要从左上走到右下，需要吃到所有奶酪，问最短的路径长度。

输入：二维数组

输出：最短路径长度

思路：先用BFS找到所有奶酪和出入口两两之间的最短距离，然后用DFS找起点-奶酪全排列-终点的最短路径长 
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <climits>
using namespace std;

void dfs(vector<vector<int>>& dp, int index, vector<int> ans, int& ret) {
    int n = ans.size();
    if (index == n - 1) {
        int temp = 0;
        for (int i = 0; i < n - 1; i++) {
            temp += dp[ans[i]][ans[i + 1]];
        }
        ret = temp < ret ? temp : ret;
        return;
    }
    for (int i = index; i < n - 1; i++) {
        swap(ans[i], ans[index]);
        dfs(dp, index + 1, ans, ret);
    }
}

int helper(vector<vector<int>>& nums) {
    int m = nums.size(), n = nums[0].size();
    nums[0][0] = 2;
    nums[m - 1][n - 1] = 2;
    vector<pair<int, int>> keys;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (nums[i][j] == 2) {
                keys.push_back({i, j});
            }
        }
    }
    int len = keys.size();
    cout << len << endl;
    vector<vector<int>> dp(len, vector<int>(len, 0));
    vector<unordered_set<string>> visited(len);
    int cnt = 0;
    vector<vector<pair<int, int>>> cur(len);
    for (int i = 0; i < len; i++) {
        cur[i].push_back(keys[i]);
    }
    while (true) {
        int mark = 0;
        vector<vector<pair<int, int>>> next(len);
        for (int i = 0; i < len; i++) {
            for (auto j : cur[i]) {
                if (find(keys.begin(), keys.end(), j) != keys.end()) {
                    int index = (int)(find(keys.begin(), keys.end(), j) - keys.begin());
                    dp[i][index] = cnt;
                }
                visited[i].insert(to_string(j.first) + "-" + to_string(j.second));
                int a[4] = {0, 0, 1, -1}, b[4] = {-1, 1, 0, 0};
                for (int k = 0; k < 4; k++) {
                    int x = j.first + a[k], y = j.second + b[k];
                    if (x < m && x >= 0 && y < n && y >= 0 && (nums[x][y] == 0 || nums[x][y] == 2)) {
                        auto temp = to_string(x) + "-" + to_string(y);
                        if (visited[i].find(temp) == visited[i].end()) {
                            next[i].push_back({x, y});
                            mark ++;
                        }
                    }
                }
            }
        }
        cnt ++;
        if (mark) cur = next;
        else break;
    }
    
    for (auto i : dp) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }

    int ret = INT_MAX;
    vector<int> temp;
    for (auto i = 0; i < len; i++) temp.push_back(i);
    dfs(dp, 1, temp, ret);
    return ret;
}


int main() {
    int m, n;
    cin >> m >> n;
    vector<vector<int>> nums(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }
    cout << helper(nums) << endl;
    return 0;
}
```

## 表达式求值
```
输入：输入合法的表达式字符串，包括括号小数正负四则运算
输出：表达式答案 
```

``` cpp
#include <iostream>
#include <math.h>

using namespace std;

double getNum(string s, int& index) {
    int n = s.size();
    double ret = 0;
    int cnt = -1;
    while (index < n && (s[index] == '.' || isdigit(s[index]))) {
        if (s[index] == '.') cnt ++;
        else {
            ret = ret * 10 + (s[index] - '0');
            if (cnt >= 0) cnt ++;
        }
        index ++;
    }
    cnt = max(0, cnt);
    return ret / pow(10, cnt);
}

double helper(string s, int& index) {
    double ret = 0.0, cur_ret = 0.0;
    char op = '+';
    int n = s.size();
    while (index < n && s[index] != ')') {
        if (isdigit(s[index]) || s[index] == '(') {
            double temp = 0.0;
            if (isdigit(s[index])) temp = getNum(s, index);
            else {
                index ++;
                temp = helper(s, index);
                index ++;
            }
            switch (op) {
                case '+' : cur_ret += temp;break;
                case '-' : cur_ret -= temp;break;
                case '*' : cur_ret *= temp;break;
                case '/' : cur_ret /= temp;break;
            }
        }
        else {
            if (s[index] == '+' || s[index] == '-') {
                ret += cur_ret;
                cur_ret = 0.0;
            }
            op = s[index++];
        }
    }
    return ret + cur_ret;
}

int main() {
    string s;
    cin >> s;
    int index = 0;
    cout << helper(s, index) << endl;
    return 0;
}

```

## 跷跷板
```
题目：给定一组数组（包含0），每一位表示该位置的重量，根据力矩*重量，计算该数组中一共有多少平衡点
输入：一维数组
输出：平衡点个数
```
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int helper(vector<int>& nums) {
    int n = nums.size();
    int ret = 0;
    for (int i = 0; i <= n; i++) {
        int l = 0, r = 0;
        for (int j = 0; j < i; j++) {
            l += nums[j] * (i - j);
        }
        for (int j = 0; j < n; j++) {
            r += nums[j] * (j - i + 1);
        }
        if (l == r) ret ++;
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    cout << helper(nums) << endl;

    return 0;
}
```

## k拼接
```
题目：给定一个数组和一个数组k，从数组中选择最多k个数组，按原顺序组合，求能够组合的最大数字

输入：一维数组，k

输出：最大数字
```

``` cpp
#include <iostream>
#include <vector>

using namespace std;
bool check(string ans, string ret) {
    if (ans.size() != ret.size()) return ans.size() > ret.size();
    return ans > ret;
}

void dfs(vector<int>& nums, string ans, int index, int cnt, int k, string& ret) {
    int n = nums.size();
    if (cnt == k) {
        if (check(ans, ret)) ret = ans;
    }
    else if (index < n) {
        dfs(nums, ans + to_string(nums[index]), index + 1, cnt + 1, k, ret);
        dfs(nums, ans, index + 1, cnt, k, ret);
    }

}

string helper(vector<int>& nums, int k) {
    int n = nums.size();
    if (k > n) return "";
    string ret;
    dfs(nums, "", 0, 0, k, ret);
    return ret;
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (auto i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(nums, k) << endl;
    return 0;
}
```

## K遍历
```
题目：airbnb是一家与旅游相关的公司，给定一个pair的数组，和一个整型k。pair代表的是两地之间有通路，现在求需要遍历k个城市，最少需要多少次飞行。 
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
using namespace std;

int helper(vector<pair<int, int>>& nums, int n, int k) {
    vector<unordered_set<int>> adj(n);
    vector<int> cnt(n, 0);
    for (auto i : nums) {
        adj[i.first].insert(i.second);
        adj[i.second].insert(i.first);
        cnt[i.second] ++;
        cnt[i.first] ++;
    }
    
    vector<int> cur;
    for (int i = 0; i < n; i++) {
        if (cnt[i] == 1) cur.push_back(i);
    }

    int ans = 0;
    while (true) {
        vector<int> next;
        for (auto i : cur) {
            for (auto j : adj[i]) {
                cnt[j] --;
                if (cnt[j] == 1) next.push_back(j);
            } 
        }
        if (next.empty()) break;
        else cur = next;
        ans ++;
    }
    if (cur.size() == 1) ans *= 2;
    else ans = ans * 2 + 1;
    if (ans > k - 1) return ans;
    return k + (k - ans) * 2;
}

int main() {
    int n, t, k;
    cin >> n >> t >> k;
    vector<pair<int, int>> nums;
    for (int i = 0; i < t; i++) {
        int x, y;
        cin >> x >> y;
        nums.push_back({x, y});
    }
    cout << helper(nums, n, k) << endl;
    return 0;
}
```

## 状态机
题目：给定一个状态转移图，和一个目标字符集合

M=
|    |A   |B   |C   |
|----|:--:|:--:|:--:|
|A   |B,C |C   |A   |
|B   |A,C |C   |C   |
|C   |A   |A   |A,B |

S={A,B,C}

每两个字符可以转换成一个字符，例如AAB可以转成BC，判断是否能最后转化成目标集合中的字符

``` cpp
#include <iostream>
#include <unordered_map>
#include <unordered_set>

using namespace std;

bool dfs(string cur, string next, unordered_set<char>& target, int index, unordered_map<char, unordered_map<char, string>>& nums) {
    int n = cur.size();
    if (index == n - 1) {
        if (n == 1) {
            return target.find(cur[0]) != target.end();
        }
        return dfs(next, "", target, 0, nums);
    }
    for (auto i : nums[cur[index]][cur[index + 1]]) {
        if (dfs(cur, next + i, target, index + 1, nums)) return true;
    }
    return false;
}

bool helper(string s, string t, unordered_map<char, unordered_map<char, string>>& nums) {
    unordered_set<char> target;
    for (auto i : t) {
        target.insert(i);
    }
    return dfs(s, "", target, 0, nums);
}

int main() {
    int n;
    cin >> n;
    unordered_map<char, unordered_map<char, string>> nums;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            string s;
            cin >> s;
            nums['a' + i]['a' + j] = s;
        }
    }
    string start, target;
    cin >> start >> target;
    cout << helper(start, target, nums) << endl;
    return 0;
}
```


## ip2cidr
```
题目：给定一个子网ip起点和连续的个数，判断最少需要多少个子网掩码可以恰好覆盖子网掩码范围，例如127.0.0.0-127.0.0.1需要255.255.255.254/31就可以，但是127.0.0.1-127.0.0.2就需要255.255.255.254/31和255.255.255.252/30 
```
```
Input: ip = "255.0.0.7", n = 10
Output: ["255.0.0.7/32","255.0.0.8/29","255.0.0.16/32"]
```

``` cpp
#include <iostream>
#include <vector>

using namespace std;

string convert(long long ip, long long diff) {
    string ret;
    for (int i = 0; i < 4; i++) {
        ret = to_string(ip & 255) + "." + ret;
        ip >>= 8;
    }
    ret.pop_back();
    int cnt = 0;
    while (diff) diff /= 2, cnt ++;
    ret += "/" + to_string(32 - cnt + 1);
    return ret;
}

vector<string> helper(string s, int n) {
    long long ip = 0;
    long long ans = 0;
    for (auto i : s) {
        if (i == '.') {
            ip <<= 8;
            ip += ans;
            ans = 0;
        }
        else ans = ans * 10 + (i - '0');
    }
    ip <<= 8;
    ip += ans;
    ip += n;
    vector<string> ret;
    while (n) {
        cout << n << endl;
        long long temp = (ip - 1) & ip;
        long long diff = ip - temp;
        while (diff > n) diff /= 2;
        ip -= diff;
        n -= diff;
        ret.push_back(convert(ip, diff));
    }
    return ret;
}

int main() {
    string s;
    cin >> s;
    int n;
    cin >> n;
    for (auto i : helper(s, n)) {
        cout << i << endl;
    }
    return 0;
}
```

## 最大公约数和最小公倍数
题目：给定一个数组，求每两个数字中的最大公约数的最大值，如果存在多个最大值，则求出这些最大公约数数字对中的最小公倍数的最小值

思路：求出数组两两数据对的所有公约数，然后根据放到一个数组中，将所有公约数排序，排序最大的就是最大公约数，如果存在多个对有最大公约数，求出这些数据对的最小公倍数（相乘除以最大公约数），然后求出其中的最小值

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <math.h>
#include <climits>
using namespace std;

bool cmp(const pair<pair<int, int>, int>& a, const pair<pair<int, int>, int>& b) {
    return a.second > b.second;
}

int helper(vector<int>& nums) {
    int n = nums.size();
    vector<pair<pair<int, int>, int>> ans;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            for (int k = min(nums[i], nums[j]); k >= 1; k--) {
                if (nums[i] % k == 0 && nums[j] % k == 0) {
                    ans.push_back({{nums[i], nums[j]}, k});
                }
            }
        }
    }
    sort(ans.begin(), ans.end(), cmp);
    vector<pair<int, int>> cur;
    int len = ans.size();
    for (int i = 0; i < len && ans[0].second == ans[i].second; i++) {
        cur.push_back(ans[i].first);
    }
    if (cur.size() == 1) return ans[0].second;
    int ret = INT_MAX;
    for (auto i : cur) {
        ret = min(ret, i.first * i.second / ans[0].second);
    }
    return ret;
}

int main() {
    int n;
    cin >> n;

    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(nums) << endl;
    return 0;
}
```

## 挪数字（9张披萨）
题目：有一个3*3的矩阵，分别标着0-8，其中0代表空白，相邻的数字可以向其挪动，现给定一个任意的矩阵，求一共需要多少步可以还原该矩阵（顺序为0-8）

``` cpp
#include <vector>
#include <iostream>
#include <unordered_set>

using namespace std;

string to_string(vector<vector<int>>& nums) {
    string ret;
    for (auto i : nums) {
        for (auto  j : i) {
            ret += to_string(j) + " ";
        }
    }
    return ret;
}

bool check(vector<vector<int>>& nums) {
    int index = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            if (nums[i][j] != (index++)) return false;
        }
    }
    return true;
}

vector<vector<int>> decode(string s, int& x, int& y) {
    int ans = 0;
    int index = 0;
    vector<vector<int>> ret(3, vector<int>(3, 0));
    for (auto i : s) {
        if (i == ' ') {
            ret[index / 3][index % 3] = ans;
            if (ans == 0) {
                x = index / 3;
                y = index % 3;
            }
            ans = 0;
            index ++;
        }
        else ans = ans * 10 + (i - '0');
    }
    return ret;
} 

int helper(vector<vector<int>>& nums) {
    int ret = 0;
    vector<string> cur;
    unordered_set<string> m;
    cur.push_back(to_string(nums));
    while (true) {
        vector<string> next;
        for (auto i : cur) {
            m.insert(i);
            int x, y;
            auto ans = decode(i, x, y);
            if (check(ans)) return ret;
            int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
            for (int j = 0; j < 4; j++) {
                int X = x + a[j], Y = y + b[j];
                if (X < 3 && X >= 0 && Y < 3 && Y >= 0) {
                    swap(ans[x][y], ans[X][Y]);
                    auto key = to_string(ans);
                    if (m.find(key) == m.end()) next.push_back(key);
                    swap(ans[x][y], ans[X][Y]);
                }
            }
        }
        if (next.empty()) break;
        else cur = next;
        ret ++;
    }
    return -1;
}

int main() {
    vector<vector<int>> nums(3, vector<int>(3, 0));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> nums[i][j];
        }
    }

    cout << helper(nums) << endl;
    return 0;
}
```

## 分糖果
题目：一个二维矩阵，每个位置上表示其权重，现为每个位置上分糖果，每个位置分的糖果不能比相邻的权重大的个数大，求最少糖果总数

思路：建立有向图，大的指向小的，相等互连，DFS搜索每个点，直至不能找到比他更小的或者周围都访问过，该点分1个，上一层分的周围点个数的最大值加一，从而求得最少糖果数。

``` cpp
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

int dfs(vector<vector<int>>& nums, vector<vector<int>>& dp, int i, int j) {
    if (dp[i][j]) return dp[i][j];
    int m = nums.size(), n = nums[0].size();
    int ret = 0;
    int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
    for (int k = 0; k < 4; k++) {
        int x = i + a[k], y = j + b[k];
        if (x < m && x >= 0 && y < n && y >= 0 && nums[i][j] > nums[x][y]) {
            ret = max(ret, dfs(nums, dp, x, y));
        }
    }
    ret++;
    dp[i][j] = ret;
    return ret;
}

int helper(vector<vector<int>>& nums) {
    int ret = 0;
    int m = nums.size(), n = nums[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret += dfs(nums, dp, i, j);
        }
    }
    return ret;
}

int main() {
    int m, n;
    cin >> m >> n;
    vector<vector<int>> nums(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }

    cout << helper(nums) << endl;
    return 0;
}
```

## Leetcode 269
```
There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.

For example,
Given the following words in dictionary,
[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]
The correct order is: "wertf".

Note:
    You may assume all letters are in lowercase.
    If the order is invalid, return an empty string.
    There may be multiple valid order of letters, return any one of them is fine.
Hide Company Tags Google Facebook
Hide Tags Graph Topological Sort
Hide Similar Problems (M) Course Schedule II
```

``` cpp
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
using namespace std;

string helper(vector<string>& nums) {
    int n = nums.size();
    unordered_map<char, unordered_set<char>> adj;
    unordered_map<char, int> cnt;
    unordered_set<char> s;
    for (int i = 0; i < n - 1; i++) {
        int n1 = nums[i].size(), n2 = nums[i + 1].size();
        for (int j = 0; j < n1 && j < n2; j++) {
            if (nums[i][j] != nums[i + 1][j]) {
                adj[nums[i][j]].insert(nums[i + 1][j]);
                cnt[nums[i + 1][j]] ++;
                break;
            }
        }
    }
    for (auto i : nums) {
        for (auto j : i) {
            s.insert(j);
        }
    }

    string cur;
    for (auto i : s) {
        if (cnt.find(i) == cnt.end()) cur.push_back(i);
    }

    string ret;
    while (true) {
        string next;
        for (auto i : cur) {
            ret.push_back(i);
            for (auto j : adj[i]) {
                cnt[j]--;
                if (!cnt[j]) next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<string> nums;
    for (int i = 0; i < n; i++) {
        string s;
        cin >> s;
        nums.push_back(s);
    }

    cout << helper(nums) << endl;

    return 0;
}
```

## Leetcode 207
```
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example 1:

Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
using namespace std;

bool helper(int n, vector<pair<int, int>>& nums) {
    vector<int> cnt(n, 0);
    vector<unordered_set<int>> adj(n);
    for (auto i : nums) {
        cnt[i.second]++;
        adj[i.first].insert(i.second);
    }

    vector<int> cur;
    for (int i = 0; i < n; i++) {
        if (!cnt[i]) cur.push_back(i);
    }

    int ret = 0;
    while (true) {
        vector<int> next;
        for (auto i : cur) {
            ret ++;
            for (auto j : adj[i]) {
                cnt[j] --;
                if (!cnt[j]) next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    return ret == n;
}

int main() {
    int n, t;
    cin >> n >> t;
    vector<pair<int, int>> nums;
    for (int i = 0; i < t; i++) {
        int x, y;
        cin >> x >> y;
        nums.push_back({x, y});
    }
    cout << helper(n, nums) << endl;
    return 0;
}
```

## Leetcode 210
```
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:

Input: 2, [[1,0]] 
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] .
Example 2:

Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
using namespace std;

vector<int> helper(vector<pair<int, int>>& nums, int n) {
    vector<int> ret;
    if (!n) return ret;
    
    vector<int> cnt(n, 0);
    vector<unordered_set<int>> adj(n);
    for (auto i : nums) {
        adj[i.second].insert(i.first);
        cnt[i.first] ++;
    }

    vector<int> cur;
    for (int i = 0; i < n; i++) {
        if (!cnt[i]) cur.push_back(i);
    }

    while (true) {
        vector<int> next;
        for (auto i : cur) {
            ret.push_back(i);
            for (auto j : adj[i]) {
                cnt[j] --;
                if (!cnt[j]) next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    return ret.size() == n ? ret : vector<int>();
}

int main() {
    int n, t;
    cin >> n >> t;
    vector<pair<int, int>> nums;
    for (int i = 0; i < t; i++) {
        int x, y;
        cin >> x >> y;
        nums.push_back({x, y});
    }
    for (auto i : helper(nums, n)) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

## Leetcode 68
```
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
Example 1:

Input:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
Example 2:

Input:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be",
             because the last line must be left-justified instead of fully-justified.
             Note that the second line is also left-justified becase it contains only one word.
Example 3:

Input:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
```
```
#include <iostream>
#include <vector>

using namespace std;

vector<string> helper(vector<string>& nums, int w) {
    int cnt = 0;
    vector<vector<string>> ans;
    vector<string> cur;
    for (auto i : nums) {
        if (i.size() + 1 + cnt > w + 1) {
            ans.push_back(cur);
            cur = vector<string>({i});
            cnt = i.size() + 1;
        }
        else {
            cnt += i.size() + 1;
            cur.push_back(i);
        }
    }
    ans.push_back(cur);
    int n = ans.size();
    vector<string> ret;
    for (int i = 0; i < n - 1; i++) {
        if (ans[i].size() == 1) {
            while (ans[i][0].size() < w) ans[i][0].push_back(' ');
            ret.push_back(ans[i][0]);
            continue;
        }
        int temp = 0;
        for (auto j : ans[i]) {
            temp += (int)j.size();
        }
        int delta = (w - temp) / (ans[i].size() - 1);
        int c = (w - temp) % (ans[i].size() - 1);
        string cur_str;
        for (int j = 0; j < ans[i].size() - 1; j++) {
            cur_str += ans[i][j];
            for (int k = 0; k < delta; k++) cur_str.push_back(' ');
            if (c > 0) cur_str.push_back(' ');
            c--; 
        }
        cur_str += ans[i].back();
        cout << cur_str << endl;
        ret.push_back(cur_str);
    }

    string cur_str;
    for (auto i : ans.back()) {
        cur_str += i;
        if (cur_str.size() < w) cur_str.push_back(' ');
    }
    while (cur_str.size() < w) cur_str.push_back(' ');
    ret.push_back(cur_str);
    return ret;
}

int main() {
    int n, w;
    cin >> n >> w;
    vector<string> nums;
    for (int i = 0; i < n; i++) {
        string s;
        cin >> s;
        nums.push_back(s);
    }
    for (auto i : helper(nums, w)) {
        cout << i << endl;
    }
    return 0;
}
```

## Leetcode 336
```
Given a list of unique words, find all pairs of distinct indices (i, j) in the given list, so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.

Example 1:

Input: ["abcd","dcba","lls","s","sssll"]
Output: [[0,1],[1,0],[3,2],[2,4]] 
Explanation: The palindromes are ["dcbaabcd","abcddcba","slls","llssssll"]
Example 2:

Input: ["bat","tab","cat"]
Output: [[0,1],[1,0]] 
Explanation: The palindromes are ["battab","tabbat"]
```

``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;
bool check(string s) {
    int n = s.size();
    for (int i = 0; i < n / 2; i++) {
        if (s[i] != s[n - i - 1]) return false;
    }
    return true;
}

vector<vector<int>> helper(vector<string>& nums) {
    int n = nums.size();
    unordered_map<string, int> m;
    for (int i = 0; i < n; i++) {
        string s = nums[i];
        reverse(s.begin(), s.end());
        m[s] = i;
    }
    vector<vector<int>> ret;
    if (m.find("") != m.end()) {
        for (int i = 0; i < n; i++) {
            if (check(nums[i]) && m[""] != i) ret.push_back({m[""], i});
        }
    }

    for (int i = 0; i < n; i++) {
        string s = nums[i];
        for (int j = 0; j < s.size(); j++) {
            string left = s.substr(0, j), right = s.substr(j);
            if (m.find(left) != m.end() && check(right) && m[left] != i) ret.push_back({i, m[left]});
            if (m.find(right) != m.end() && check(left) && m[right] != i) ret.push_back({m[right], i});
        }
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<string> nums;
    for (int i = 0; i < n; i++) {
        string s;
        cin >> s;
        nums.push_back(s);
    }
    // vector<string> nums({"abcd","dcba","lls","s","sssll"});
    for (auto i : helper(nums)) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}
```

## Leetcode 79
```
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false. 
```

``` cpp
#include <iostream>
#include <vector>

using namespace std;

bool dfs(vector<vector<char>>& nums, int i, int j, string s, int index) {
    int m = nums.size(), n = nums[0].size();
    int len = s.size();
    if (s[index] != nums[i][j]) return false;
    if (index == len - 1) return true;
    char temp = s[index++];
    nums[i][j] = '\0';
    int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
    for (int k = 0; k < 4; k++) {
        int x = i + a[k], y = j + b[k];
        if (x < m && x >= 0 && y < n && y >= 0 && dfs(nums, x, y, s, index)) return true;
    }
    nums[i][j] = temp;
    return false;
}

bool helper(vector<vector<char>>& nums, string s) {
    if (nums.empty() || nums[0].empty()) return false;
    int m = nums.size(), n = nums[0].size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (dfs(nums, i, j, s, 0)) return true;
        }
    }
    return false;
}

int main() {
    int m, n;
    cin >> m >> n;
    vector<vector<char>> nums(m, vector<char>(n, ' '));
    for (auto i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }
    string s;
    cin >> s;
    cout << helper(nums, s) << endl;
    return 0;
}
```

## Leetcode 212
```
Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

Example:

Input: 
words = ["oath","pea","eat","rain"] and board =
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

Output: ["eat","oath"] 
```

``` cpp
#include <vector>
#include <iostream>
#include <cstring>
#include <unordered_set>
using namespace std;

struct TrieNode {
    bool isKey;
    TrieNode* child[26];
    TrieNode(): isKey(false) {
        memset(child, NULL, sizeof(child));
    }
};

TrieNode* build(vector<string>& nums) {
    TrieNode* root = new TrieNode();
    for (auto i : nums) {
        TrieNode* cur = root;
        for (auto j : i) {
            if (!cur->child[j - 'a']) cur->child[j - 'a'] = new TrieNode();
            cur = cur->child[j - 'a'];
        }
        cur->isKey = true;
    }
    return root;
}

void dfs(vector<vector<char>>& nums, int i, int j, TrieNode* cur, string ans, unordered_set<string>& ret) {
    int m = nums.size(), n = nums[0].size();
    int index = nums[i][j] - 'a';
    if (!cur->child[index]) return;
    ans.push_back(nums[i][j]);
    cur = cur->child[index];
    if (cur->isKey) {
        ret.insert(ans);
    }
    nums[i][j] = '\0';
    int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
    for (int k = 0; k < 4; k++) {
        int x = i + a[k], y = j + b[k];
        if (x < m && x >= 0 && y < n && y >= 0 && nums[x][y] != '\0') {
            dfs(nums, x, y, cur, ans, ret);
        }
    }
    nums[i][j] = 'a' + index;
}

vector<string> helper(vector<vector<char>>& nums, vector<string>& target) {
    if (nums.empty() || nums[0].empty()) return vector<string>();
    int m = nums.size(), n = nums[0].size();
    unordered_set<string> ans;
    auto root = build(target);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dfs(nums, i, j, root, "", ans);
        }
    }
    vector<string> ret;
    for (auto i : ans) ret.push_back(i);
    return ret;
}

int main() {
    int m, n;
    cin >> m >> n;
    vector<vector<char>> nums(m, vector<char>(n, '0'));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }
    int t;
    cin >> t;
    vector<string> target;
    for (int i = 0; i < t; i++) {
        string s;
        cin >> s;
        target.push_back(s);
    }
    for (auto i : helper(nums, target)) {
        cout << i << endl;
    }
    return 0;
}
```

## 判断麻将胡牌
```
存在一个对和四个三连张或者相同三张 
```
``` cpp
#include <iostream>
#include <unordered_map>
#include <algorithm>
using namespace std;

bool dfs(string s, int index, int a, int b, unordered_map<char, int>& m) {
    if (a == 1 && b == 4) return true;
    int n = s.size();
    if (index == n) return false;
    if (!a && m[s[index]] >= 2) {
        m[s[index]] -= 2;
        if (dfs(s, index + 1, a + 1, b, m)) return true;
        m[s[index]] += 2;
    }
    if (b < 4 && m[s[index]]) {
        if (m[s[index]] >= 3) {
            m[s[index]] -= 3;
            if (dfs(s, index + 1, a, b + 1, m)) return true;
            m[s[index]] += 3;
        }
        if (m[s[index]] && m[s[index] + 1] && m[s[index] + 2]) {
            m[s[index]] --;
            m[s[index] + 1] --;
            m[s[index] + 2] --;
            if (dfs(s, index + 1, a, b + 1, m)) return true;
            m[s[index]] ++;
            m[s[index] + 1] ++;
            m[s[index] + 2] ++;
        }
    }
    if (dfs(s, index + 1, a, b, m)) return true;
    return false;
}

bool helper(string s) {
    if (s.size() != 14) return false;
    sort(s.begin(), s.end());
    unordered_map<char, int> m;
    for (auto i : s) m[i] ++;
    return dfs(s, 0, 0, 0, m);
}

int main() {
    string s;
    cin >> s;
    cout << helper(s) << endl;
    return 0;
}
```

## 社交网络
```
给出社交网络中的关注关系（图的边），信息可以从被关注者流到关注者，挑选最少的人将一个信息传播到所有人那里 

思路：
先BFS找到从每个节点出发能到达的所有节点，然后dfs找
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
#include <unordered_map>
using namespace std;

void dfs(unordered_map<int, unordered_set<int>>& visited, vector<int> ans, vector<int>& ret, unordered_set<int> s, int index, vector<int>& key) {
    int n = visited.size();
    if (s.size() == n) {
        if (ans.size() < ret.size()) ret = ans;
        return;
    }
    if (index == n) return;
    dfs(visited, ans, ret, s, index + 1, key);
    ans.push_back(key[index]);
    for (auto i : visited[key[index]]) {
        s.insert(i);
    }
    dfs(visited, ans, ret, s, index + 1, key);
}

vector<int> helper(vector<pair<int, int>>& nums) {
    unordered_map<int, unordered_set<int>> adj;
    unordered_set<int> s;
    for (auto i : nums) {
        adj[i.first].insert(i.second);
        s.insert(i.first);
        s.insert(i.second);
    }
    unordered_map<int, unordered_set<int>> visited;
    int len = s.size();
    vector<vector<int>> cur;
    vector<int> index;
    for (auto i : s) {
        index.push_back(i);
        cur.push_back(vector<int>({i}));
    }
    while (true) {
        vector<vector<int>> next(len);
        int check = 0;
        for (int i = 0; i < len; i++) {
            for (auto j : cur[i]) {
                visited[index[i]].insert(j);
                for (auto k : adj[j]) {
                    if (visited[index[i]].find(k) == visited[index[i]].end()) {
                        next[i].push_back(k);
                        check ++;
                    }
                }
            }
        }
        if (check) cur = next;
        else break;
    }

    vector<int> ret = index;
    dfs(visited, vector<int>(), ret, unordered_set<int>(), 0, index);
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<pair<int, int>> nums;
    for (int i = 0; i < n; i++) {
        int x, y;
        cin >> x >> y;
        nums.push_back({x, y});
    }
    for (auto i : helper(nums)) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

## Preference List
每个人都有一个preference的排序，在不违反每个人的preference的情况下得到总体的preference的排序

``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

vector<int> helper(vector<vector<int>>& nums) {
    unordered_map<int, unordered_set<int>> adj;
    unordered_set<int> s;
    for (auto i : nums) {
        for (auto j : i) {
            s.insert(j);
        }
    }
    unordered_map<int, int> cnt;
    for (auto num : nums) {
        int len = num.size();
        for (int i = 0; i < len - 1; i++) {
            adj[num[i]].insert(num[i + 1]);
            cnt[num[i + 1]] ++;
        }
    }

    vector<int> cur;
    for (auto i : s) {
        if (cnt.find(i) == cnt.end()) cur.push_back(i);
    }

    vector<int> ret;
    while (true) {
        vector<int> next;
        for (auto i : cur) {
            ret.push_back(i);
            for (auto j : adj[i]) {
                cnt[j] --;
                if (!cnt[j]) next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<vector<int>> nums(n);
    for (int i = 0; i < n; i++) {
        int t;
        cin >> t;
        for (int j = 0; j < t; j++) {
            int x;
            cin >> x;
            nums[i].push_back(x);
        }
    }
    for (auto i : helper(nums)) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```
## 数字变英语
0-100，输出对应的英语

``` cpp
#include <iostream>
#include <vector>

using namespace std;

string helper(int n) {
    if (!n) return "zero";
    string a[10] = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"};
    if (n <= 10) return a[n - 1];
    string b[9] = {"eleven", "twelven", "thirteen", "forteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    if (n < 20) return b[n - 11];
    if (n == 100) return "one hundred";
    string c[8] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    string ret = c[n / 10 - 2];
    if (n % 10) ret += " " + a[n % 10 - 1];
    return ret;
}

int main() {
    for (int i = 0; i < 100; i ++) {
        cout << helper(i) << endl;
    }
    return 0;
}
```

## round number
```
When you book on airbnb the total price is:

Total price = base price + service fee + cleaning fee + …

input : array of decimals ~ X
output : array of int ~ Y
But they need to satisfy the condition:

sum(Y) = round(sum(x))
minmize (|y1-x1| + |y2-x2| + ... + |yn-xn|)
Example1:
input = 30.3, 2.4, 3.5
output = 30 2 4

Example2:
input = 30.9, 2.4, 3.9
output = 31 2 4
```

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>


using namespace std;

bool cmp(const pair<double, int>& a, const pair<int, int>& b) {
  return a.first - (int)a.first > b.first - (int)b.first;
}

vector<int> helper(vector<double>& nums) {
  int n = nums.size();
  
  double ans_d = 0.0;
  int ans_i = 0;
  
  vector<int> ret;
  vector<pair<double, int>> ans;
  
  for (int i = 0; i < n; i++) {
    ret.push_back((int)nums[i]);
    
    ans_i += ret.back();
    ans_d += nums[i];
    
    ans.push_back({nums[i], i});
  }
  
  if (ans_i == (int)(ans_d + 0.5)) return ret;
  
  
  int c = (int)(ans_d + 0.5) - ans_i;
  sort(ans.begin(), ans.end(), cmp);
  for (int i = 0; i < n && c; i++) {
    ret[ans[i].second] ++;
    c--;
  }
  
  return ret;
}


int main() {
  int n;
  cin >> n;
  
  vector<double> nums(n, 0);
  for (int i = 0; i < n; i++) {
    cin >> nums[i];
  }
  
  for (auto i : helper(nums)) {
    cout << i << endl;
  }
  
  return 0;
}
```

## 2D itertaor + remove()
```
#include <iostream>
#include <vector>

using namespace std;
class MyIter {
private:
    vector<vector<int>>::iterator ibegin, iend, icur;
    vector<int>::iterator jcur;
public:
    MyIter(vector<vector<int>>& nums) {
        ibegin = nums.begin();
        iend = nums.end();
        icur = ibegin;
        if (ibegin != iend) {
            jcur = ibegin->begin();
        }
    }
    bool hasNext() {
        if (icur == iend) return false;
        if (jcur != icur->end()) return true;
        icur ++;
        while (icur != iend && icur->begin() == icur->end()) icur++;
        if (icur == iend) return false;
        jcur = icur->begin();
        return true;
    }
    int next() {
        if (hasNext()) {
            int ret = *jcur;
            jcur ++;
            return ret;
        }
        return -1;
    }
    void erase() {
        if (hasNext()) {
            icur->erase(jcur);
        }
    }
};
int main() {
    int n;
    cin >> n;
    vector<vector<int>> nums;
    for (int i = 0; i < n; i++) {
        int t;
        cin >> t;
        vector<int> ans(t, 0);
        for (int j = 0; j < t; j++) {
            cin >> ans[j];
        }
        nums.push_back(ans);
    }
    auto iter = new MyIter(nums);
    iter->erase();
    while (iter->hasNext()) {
        cout << iter->next() << endl;
    }

    return 0;
} 
```

## 分页
```
第一轮实现分页显示。给了以下一些输入数据，要求将以下行分页显示，每页12行，其中每行已经按score排好序，分页显示的时候如果有相同host id的行，则将后面同host id的行移到下一页。

[
"host_id,listing_id,score,city",
"1,28,300.1,SanFrancisco",
"4,5,209.1,SanFrancisco",
"20,7,208.1,SanFrancisco",
"23,8,207.1,SanFrancisco",
"16,10,206.1,Oakland",
"1,16,205.1,SanFrancisco",
"6,29,204.1,SanFrancisco",
"7,20,203.1,SanFrancisco",
"8,21,202.1,SanFrancisco",
"2,18,201.1,SanFrancisco",
"2,30,200.1,SanFrancisco",
"15,27,109.1,Oakland",
"10,13,108.1,Oakland",
"11,26,107.1,Oakland",
"12,9,106.1,Oakland",
"13,1,105.1,Oakland",
"22,17,104.1,Oakland",
"1,2,103.1,Oakland",
"28,24,102.1,Oakland",
"18,14,11.1,SanJose",
"6,25,10.1,Oakland",
"19,15,9.1,SanJose",
"3,19,8.1,SanJose",
"3,11,7.1,Oakland",
"27,12,6.1,Oakland",
"1,3,5.1,Oakland",
"25,4,4.1,SanJose",
"5,6,3.1,SanJose",
"29,22,2.1,SanJose",
"30,23,1.1,SanJose"
]
```

``` cpp
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

vector<vector<string>> helper(vector<string>& nums, int n) {
    vector<vector<string>> ret;
    while (!nums.empty()) {
        vector<string> cur;
        unordered_set<string> m;
        for (auto i = nums.begin(); i != nums.end() && (int)cur.size() < n;) {
            if (m.find(*i) != m.end()) {
                i ++;
                continue;
            }
            cur.push_back(*i);
            m.insert(*i);
            nums.erase(i);
        }
        for (auto i = nums.begin(); i != nums.end() && (int) cur.size() < n;) {
            cur.push_back(*i);
            nums.erase(i);
        }
        ret.push_back(cur);
    }
    return ret;
}

int main() {
    vector<string> nums({
"host_id,listing_id,score,city",
"1,28,300.1,SanFrancisco",
"4,5,209.1,SanFrancisco",
"20,7,208.1,SanFrancisco",
"23,8,207.1,SanFrancisco",
"16,10,206.1,Oakland",
"1,16,205.1,SanFrancisco",
"6,29,204.1,SanFrancisco",
"7,20,203.1,SanFrancisco",
"8,21,202.1,SanFrancisco",
"2,18,201.1,SanFrancisco",
"2,30,200.1,SanFrancisco",
"15,27,109.1,Oakland",
"10,13,108.1,Oakland",
"11,26,107.1,Oakland",
"12,9,106.1,Oakland",
"13,1,105.1,Oakland",
"22,17,104.1,Oakland",
"1,2,103.1,Oakland",
"28,24,102.1,Oakland",
"18,14,11.1,SanJose",
"6,25,10.1,Oakland",
"19,15,9.1,SanJose",
"3,19,8.1,SanJose",
"3,11,7.1,Oakland",
"27,12,6.1,Oakland",
"1,3,5.1,Oakland",
"25,4,4.1,SanJose",
"5,6,3.1,SanJose",
"29,22,2.1,SanJose",
"30,23,1.1,SanJose"       
    });

    int n;
    cin >> n;
    auto ret = helper(nums, n);
    for (auto i : ret) {
        for (auto j : i) {
            cout << j << endl;
        }
        cout << endl;
    }
    return 0;
}
```
## menu order 
点菜，菜价格为double
- 是否可以花完钱
- 如何花完钱

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

bool helper(vector<double>& nums, double target) {
    int n = nums.size();
    vector<int> ans(n, 0);
    for (int i = 0; i < n; i++) {
        ans[i] = round(nums[i] * 100);
        // 这里必须使用round，因为浮点数乘法后取整是直接截取，在计算当中可能是近似值，所以可能直接截取，导致得不到想要的值
    }

    int m = round(target * 100);
    vector<bool> dp(m + 1, false);
    cout << m << endl;
    dp[0] = true;
    sort(ans.begin(), ans.end());
    for (int i = 0; i < n; i++) {
        for (int j = m; j >= ans[i]; j--) {
            if (j == m) {
                cout << j - ans[i]  << ans[i] << " " << dp[j - ans[i]] << endl;
            }
            dp[j] = dp[j] || dp[j - ans[i]];
        }
    }
    for (int i = 0; i <= m; i++) {
        if (dp[i]) cout << i << endl;
    }
    return dp[m];
}

int main() {
    int n;
    cin >> n;
    vector<double> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    double target;
    cin >> target;
    cout << helper(nums, target) << endl;
    return 0;
}
/*
3
2.3 0.3 6
8.6
*/

```

``` cpp
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>

using namespace std;

bool cmp(const pair<int, int>& a, const pair<int, int>& b) {
  return a.first < b.first;
}


void dfs(vector<pair<int, int>>& nums, int index, int ans, int target, vector<int> cur, vector<vector<int>>& ret) {
  if (ans > target) return;
  if (target == ans) {
    ret.push_back(cur);
    return;
  }
  
  int n = nums.size();
  if (index == n) return;
  
  dfs(nums, index + 1, ans, target, cur, ret);
  
  if (nums[index].first + ans > target) return;
  
  cur.push_back(nums[index].second);
  dfs(nums, index + 1, ans + nums[index].first, target, cur, ret);
}


vector<vector<int>> helper(vector<double>& nums, double s) {
  int n = nums.size();
  
  vector<pair<int, int>> ans;
  int target = s * 100;
  
  for (int i = 0; i < n; i++) {
    ans.push_back({nums[i] * 100, i});
  }
  
  sort(ans.begin(), ans.end(), cmp);
  
  vector<vector<int>> ret;
  dfs(ans, 0, 0, target, vector<int>(), ret);
  
  return ret;
  
}


int main() {
  int n;
  cin >> n;
  
  vector<double> nums(n, 0);
  for (int i = 0; i < n; i++) {
    cin >> nums[i];
  }
  
  double target;
  cin >> target;
  
  for (auto i : helper(nums, target)) {
    for (auto j : i) {
      cout << j << " ";
    }
    cout << endl;
  }
  return 0;
}
```

## Hilbert Curve
[Hilbert Curve参考文献](http://bit-player.org/extras/hilbert/hilbert-construction.html)

Hilbert曲线可以无限阶下去，从1阶开始，落在一个矩阵里，让你写个function，三个参数（x,y,阶数），return 这个点（x,y）是在这阶curve里从原点出发的第几步

``` cpp
#include <iostream>

using namespace std;

int helper(int x, int y, int iter) {
  if (iter == 0) return 1;
  int len = 1 << (iter - 1);
  int ans = 1 << (2 * (iter - 1));
  
  if (x < len && y < len) { // 第一象限旋转90度
    return helper(y, x, iter - 1);
  }
  else if (x < len && y >= len) { // 平移第一象限
    return ans + helper(x, y - len, iter - 1);
  }
  else if (x >= len && y >= len) { // 平移第一象限
    return 2 * ans + helper(x - len, y - len, iter - 1);
  }
  else { // 逆时针旋转90度
    return 3 * ans + helper(len - y - 1, len * 2 - x - 1, iter - 1);
  }
}


int main() {
  int x, y;
  cin >> x >> y;
  int iter;
  cin >> iter;
  
  cout << helper(x, y, iter) << endl;
  return 0;
}
```

## Meeting Time
一组pair，标明start,end，表示一个员工忙碌的时间段，现在求出至少有k个员工不忙碌的时间段

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <climits>
#include <math.h>
using namespace std;


vector<pair<int, int>> helper(vector<pair<int, int>>& nums, int n, int k) {
  int start = INT_MAX, end = INT_MIN;
  for (auto i : nums) {
    start = min(start, i.first);
    end = max(end, i.second);
  }
  
  int len = end - start + 1;
  
  vector<int> cnt(len, 0);
  for (auto i : nums) {
    cnt[i.first - start] ++;
    cnt[i.second - start] --;
  }
  
  vector<pair<int, int>> ret;
  
  int limit = 0;
  int ans = 0;
  for (int i = 0; i < len; i++) {
    if (ans + k <= n && ans + cnt[i] + k > n && limit != i) {
      ret.push_back({limit + start, i + start});
    }
    if (ans + k > n && ans + cnt[i] + k <= n){
      limit = i;
    }
    ans += cnt[i];
  }
  if (ans + k < n) {
    ret.push_back({limit + start, end});
  }
  return ret;
}

int main() {
  int n, k;
  cin >> n >> k;
  vector<pair<int, int>> nums;
  
  for (int i = 0; i < n; i++) {
    int x, y;
    cin >> x >> y;
    nums.push_back({x, y});
  }
  
  for (auto i : helper(nums, n, k)) {
    cout << i.first << " " << i.second << endl;
  }
  
  return 0;
}
```

## 小于k步的最小距离

有权有向图，给定起点和终点，求小于k次的最小距离

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <climits>
#include <unordered_set>
using namespace std;

int helper(vector<pair<int, int>>& nums, vector<int>& val, int n, int k, int start, int end) {
    int len = nums.size();
    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));
    vector<unordered_set<int>> adj(n);
    for (int i = 0; i < len; i++) {
        dp[nums[i].first][nums[i].second] = val[i];
        adj[nums[i].first].insert(nums[i].second);
    }
    dp[start][start] = 0;
    vector<int> cur;
    cur.push_back(start);

    for (int i = 0; i < k; i++) {
        vector<int> next;
        for (auto a : cur) {
            for (auto j : adj[a]) {
                if (dp[a][j] != INT_MAX)
                    dp[start][j] = min(dp[start][j], dp[start][a] + dp[a][j]);
                next.push_back(j);
            }
        }
        cur = next;
    }
    return dp[start][end];
}

int main() {
    int n, t, k;
    cin >> n >> t >> k;
    vector<pair<int, int>> nums;
    vector<int> val;
    for (int i = 0; i < t; i++) {
        int x, y, v;
        cin >> x >> y >> v;
        nums.push_back({x, y});
        val.push_back(v);
    }
    int s, e;
    cin >> s >> e;
    cout << helper(nums, val, n, k, s, e) << endl;
    return 0;
}
```

## URL Shortener
看描述好像是url里的id如果有某些位置大小写换了会导致原来的url decode有问题，需要重写encode方法，回溯改某些位的大小写判断

``` cpp
#include <iostream>

using namespace std;

int decode(string s) {
  if (s == "kljJJ324hijkS_") return 848662;
  else return -1;
}

char convert(char c) {
  if (c <= 'Z' && c >= 'A') return 'a' + (c - 'A');
  else return 'A' + (c - 'a');
}

int helper(string s, int index) {
  int n = s.size();
  if (index == n) {
    return decode(s);
  }
  
  if ((s[index] <= 'Z' && s[index] >= 'A') || (s[index] <= 'z' && s[index] >= 'a')) {
    int l = helper(s, index + 1);
    string temp = s.substr(0, index) + convert(s[index]) + s.substr(index + 1);
    int r = helper(temp, index + 1);
    if (l != -1 || r != -1) {
      return l == -1 ? r : l;
    }
    return -1;
  }
  return helper(s, index + 1);
}


int main() {
  string s = "kljJJ324hijks_";
  cout << helper(s, 0) << endl;
  return 0;
}
```

## wizards
There are 10 wizards, 0-9, you are given a list that each entry is a list of wizards known by wizard. Define the cost between wizards and wizard as square of different of i and j. To find the min cost between 0 and 9.

``` cpp
#include <iostream>
#include <utility>
#include <vector>
#include <math.h>
#include <unordered_set>
using namespace std;

int helper(vector<vector<int>>& nums) {
    int n = nums.size();
    
    vector<unordered_set<int>> adj(n);
    for (int i = 0; i < n; i++) {
        for (auto j : nums[i]) {
            adj[i].insert(j);
        }
    }

    unordered_set<int> s;
    vector<int> dp(n, INT_MAX);
    for (int i = 0; i < n; i++) {
        s.insert(i);
    }
    dp[0] = 0;
    while (!s.empty()) {
        int check = 0;
        int index = -1, ans = INT_MAX;
        for (auto i : s) {
            if (dp[i] < ans) {
                index = i;
                ans = dp[i];
            }
        }

        s.erase(index);
        for (auto i : adj[index]) {
            dp[i] = min(dp[i], dp[index] + (int)pow((i - index), 2));
            check ++;
        }
        if (!check) break;
    }
    return dp[n - 1];
}

int main() {
    vector<vector<int>> nums(
        {{1, 5, 9}, {2, 3, 9}, {4}, {}, {}, {9}, {}, {}, {}, {}}
    );
    cout << helper(nums) << endl;
    return 0;
}
```

## water drop
```
Input is a array represent how the height of water is at each position, the number of water will be poured, and the pour position. Print the land after all water are poured.

Example: input land height int[]{5,4,2,1,3,2,2,1,0,1,4,3} The land is looks ike:

+
++        +
++  +     ++
+++ +++   ++
++++++++ +++
++++++++++++
012345678901
water quantity is 8 and pour at position 5. The land becomes:

+
++        +
++www+    ++
+++w+++www++
++++++++w+++
++++++++++++
012345678901 
```

``` cpp
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

bool check(vector<int>& nums, int index, int& l, int& r) {
    int n = nums.size();
    l = r = index;
    while (l > 0 && nums[l - 1] <= nums[l]) l--;
    while (r < n - 1 && nums[r + 1] <= nums[r]) r ++;
    return nums[index] == nums[l] && nums[index] == nums[r];
}

void helper(vector<int>& nums, int index, int cap) {
    int n = nums.size();
    while (cap) {
        int l, r;
        if (check(nums, index, l, r)) nums[index] ++, cap --;
        else {
            if (l >= 0) helper(nums, l, 1), cap --;
            if (cap > 0 && r < n) helper(nums, r, 1), cap --;
        }
    }
}

void print(vector<int>& ret, vector<int>& nums) {
    int n = ret.size();
    int m = 0;
    for (auto i : ret) m = max(m, i);
    vector<vector<char>> ans(m, vector<char>(n, ' '));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nums[i]; j++) {
            ans[j][i] = '*';
        }
        for (int j = nums[i]; j < ret[i]; j++) {
            ans[j][i] = '+';
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m / 2; j++) {
            swap(ans[j][i], ans[m - 1 - j][i]);
        }
    }
    for (int i = 0; i < m; i++) {
        for (auto j : ans[i]) {
            cout << j << " ";
        }
        cout << endl;
    }
}

int main() {
    int index = 5, cap = 8;
    vector<int> nums({5,4,2,1,2,3,2,1,0,1,2,4});
    vector<int> ret(nums);
    helper(ret, index, cap);
    print(ret, nums);
    return 0;
}
```


## 从大文件中找中值
- 二叉树
``` cpp
#include <iostream>
#include <vector>
using namespace std;

struct TreeNode {
    int val;
    int overlap;
    int cnt;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), overlap(1), cnt(0), left(NULL), right(NULL) {}
};


void buildTree(TreeNode* &cur, int num) {
    if (!cur) cur = new TreeNode(num);
    else {
        if (cur->val == num) cur->overlap ++;
        if (cur->val > num) {
            cur->cnt ++;
            buildTree(cur->left, num);
        }
        if (cur->val < num) {
            buildTree(cur->right, num);
        }
    }
}

int dfs(TreeNode* root, int index) {
    if (root->cnt >= index) return dfs(root->left, index);
    if (root->cnt + root->overlap >= index) return root->val;
    return dfs(root->right, index - root->cnt - root->overlap);
}


int helper(vector<int>& nums) {
    
    TreeNode* root = NULL;
    for (auto i : nums) {
        buildTree(root, i);
    }
    int n = nums.size();
    return (dfs(root, (n + 1) / 2) + dfs(root, (n + 2) / 2)) / 2.0;
}

int main() {
    int n;
    cin >> n;
    
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }  
    cout << helper(nums)  << endl;

    return 0;
}
```
- 二分
``` cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int helper(vector<int>& nums, long long index) {
    // 偶数会最多遍历文件64次，奇数遍历文件32次
    long long left = INT_MIN, right = INT_MAX;
    int n = nums.size();
    while (left <= right) {
        long long l = 0, r = 0, cnt = 0;
        long long mid = left + (right - left) / 2;
        for (auto i : nums) {
            if (mid > i) l++;
            else if (mid < i) r++;
            else cnt ++;
        }
        if (l < index && l + cnt >= index) return mid;
        else if (l >= index) right = mid - 1;
        else if (l + cnt < index) left = mid + 1;
    }
    return left;
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << (helper(nums, (n + 1) / 2) + helper(nums, (n + 2) / 2)) / 2.0 << endl;
    return 0;
}
```

## 字符串乘法包含正负号
``` cpp
#include <iostream>
#include <algorithm>

using namespace std;

string helper(string a, string b) {
  int syn = 1;
  if (a[0] == '+' || a[0] == '-') {
    if (a[0] == '-') syn ^= 1;
    a.erase(a.begin());
  }
  if (b[0] == '+' || b[0] == '-') {
    if (b[0] == '-') syn ^= 1;
    b.erase(b.begin());
  }
  
  string ret((int)a.size() + (int)b.size(), '0');
  reverse(a.begin(), a.end());
  reverse(b.begin(), b.end());
  
  int len1 = a.size();
  int len2 = b.size();
  
  for (int i = 0; i < len1; i++) {
    int C = 0;
    int A = a[i] - '0';
    for (int j = 0; j < len2; j++) {
      int B = b[j] - '0';
      int temp = (ret[i + j] - '0') + A * B + C;
      ret[i + j] = '0' + (temp % 10);
      C = temp / 10;
    }
    ret[i + len2] = ret[i + len2] + C;
  }
  
  reverse(ret.begin(), ret.end());
  
  while (!ret.empty() && ret[0] == '0') ret.erase(ret.begin());
  if (ret.empty()) return "0";
  if (!syn) ret = "-" + ret;
  return ret;
}



int main() {
  string a, b;
  cin >> a >> b;
  cout << helper(a, b) << endl;
  return 0;
}
```

## csv parser
```
Input:	csvformat	
John,Smith,john.smith@gmail.com,Los	Angeles,1	
Jane,Roberts,janer@msn.com,"San	Francisco,	CA",0
"Alexandra	""Alex""",Menendez,alex.menendez@gmail.com,Miami,1	"""Alexandra	Alex"""	
Output:	escaped	string
John|Smith|john.smith@gmail.com|Los	Angeles|1	
Jane|Roberts|janer@msn.com|San	Francisco,	CA|0
Alexandra	"Alex"|Menendez|alex.menendez@gmail.com|Miami|1	"Alexandra	Alex"
```


``` cpp
#include <iostream>
#include <vector>
using namespace std;

string helper2(string s, int& index) {
  string ret;
  int n = s.size();
  while (index < n && (s[index] != '"' || (index + 1 < n && s[index + 1] == '"'))) {
    string ans;
    while (index < n && s[index] != '"') ans.push_back(s[index++]);
    if (index  + 1 < n && s[index + 1] == '"') {
      index += 2;
      string temp;
      temp += '"';
      while (index < n && s[index] != '"') temp.push_back(s[index++]);
      temp += '"';
      index += 2;
      ans += temp;
    }
    ret += ans;
  }
  return ret;
}

string helper1(string s, int& index) {
  int n = s.size();
  string ret;
  while (index < n && s[index] != ',') {
    string ans;
    while (index < n && s[index] != '"' && s[index] != ',') ans.push_back(s[index ++]);
    if (index < n && s[index] == '"') {
      index ++;
      string temp;
      temp += helper2(s, index);
      index ++;
      ans += temp;
    }
    ret += ans;
  }
  return ret;
}

string helper(string s) {
  int n = s.size();
  int index = 0;
  
  vector<string> ans;
  while (index < n) {
    ans.push_back(helper1(s, index));
    index ++;
  }
  
  string ret;
  int len = ans.size();
  for (int i = 0; i < len - 1; i ++) {
    ret += ans[i] + "|";
  }
  ret += ans.back();
  return ret;
}


int main() {
  string s = "\"Alexandra  \"\"Alex\"\"\",Menendez,alex.menendez@gmail.com,Miami,1  \"\"\"Alexandra  Alex\"\"\"";
  cout << s << endl;
  cout << helper(s) << endl;
  
  return 0;
}
```

## boggle game
从二维数组里找一条路径，包含字典数中最多的单词数
``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>

using namespace std;

struct TrieNode {
  bool isKey;
  TrieNode* child[26];
  TrieNode() : isKey(false) {
    memset(child, NULL, sizeof(child));
  }
};

TrieNode* buildTrie(vector<string>& words) {
  TrieNode* root = new TrieNode();
  for (auto s : words) {
    auto cur = root;
    for (auto i : s) {
      int index = i - 'a';
      if (!cur->child[index]) cur->child[index] = new TrieNode();
      cur = cur->child[index];
    }
    cur->isKey = true;
  }
  return root;
}

void dfs(vector<vector<char>>& nums, int i, int j, TrieNode* cur, TrieNode* root, int cnt, int& ret) {
  
  int m = nums.size(), n = nums[0].size();
  if (nums[i][j] == '\0') return;
  
  int index = nums[i][j] - 'a';
  if (!cur->child[index]) return;
  
  cur = cur->child[index];
  
  nums[i][j] = '\0';
  
  if (cur->isKey) {
    ret = max(ret, cnt + 1);
  }
  
  int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
  for (int k = 0; k < 4; k++) {
    int x = i + a[k], y = j + b[k];
    if (x < m && x >= 0 && y < n && y >= 0) {
      dfs(nums, x, y, cur, root, cnt, ret);
      if (cur->isKey) {
        dfs(nums, x, y, root, root, cnt + 1, ret);
      }
    }
  }
  
  nums[i][j] = 'a' + index;
}

int helper(vector<vector<char>>& nums, vector<string>& words) {
  auto root = buildTrie(words);
  
  int m = nums.size(), n = nums[0].size();
  
  int ret = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dfs(nums, i, j, root, root, 0, ret);
    }
  }
  
  return ret;
}

int main() {
  int m, n;
  cin >> m >> n;
  vector<vector<char>> nums(m, vector<char>(n, '\0'));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cin >> nums[i][j];
    }
  }
  
  int t;
  cin >> t;
  vector<string> s;
  for (int i = 0; i < t; i++) {
    string temp;
    cin >> temp;
    s.push_back(temp);
  }
  
  cout << helper(nums, s) << endl;
  
  return 0;
}
```


## Collatz	Conjecture
```
题目是给你公式，比如偶数的话除 2，奇数的话就变成 3*n+1，对于任何一个正数，数学猜想是最
终他会变成 1。每变一步步数加 1，给一个上限，让找出范围内最长步数。

比如 7，变换到 1 是如下顺序：7->22->11->34->17->52->26->13->40->20->10->5->16->8->4->2->1,	总
共需要 17 步。
```

``` cpp
#include <iostream>
#include <unordered_map>

using namespace std;

int helper(int n, unordered_map<int, int>& m) {
  if (n == 1) return 1;
  if (m.find(n) != m.end()) return m[n];
  
  if (n & 1) m[n] = 1 + helper(n * 3 + 1, m);
  else m[n] = 1 + helper(n / 2, m);
  return m[n];
}


int main() {
  int n;
  cin >> n;
  unordered_map<int, int> m;
  cout << helper(n, m) << endl;
  return 0;
}
```

## Queue - Array Implementation
固定长度数组实现队列
``` cpp
#include <iostream>

using namespace std;

class MyQueue {

private:
  int nums[10];
  int len;
  int begin;

public:
  
  MyQueue(): len(0), begin(0) {};
  
  bool empty() {
    return len == 0;
  }
  
  void push(int val) {
    if (len != 10) {
      int index = (begin + len + 10) % 10;
      nums[index] = val;
      len++;
    }
    else {
      throw "Error";
    }
  }
  
  void pop() {
    if (!len) throw "Error";
    len--;
    begin = (begin + 1 + 10) % 10;
  }
  
  int front() {
    if (!len) throw "Error";
    else return nums[begin];
  }
  int back() {
    if (!len) throw "Error";
    else return nums[begin + len - 1 + 10] % 10;
  }
};


int main() {
  auto q = new MyQueue();
  q->push(1);
  q->pop();
  cout << q->empty() << endl;
  q->pop();
  return 0;
}
```


## 文件系统 
```
int(fs.create('/a', 1)) # True
print(fs.get('/a')) # 1
print(fs.create('/a/b', 2)) # True
print(fs.create('/b/c', 3)) # False
print(fs.watch('/a/b', callback0))
print(fs.watch('/a', callback1))
print(fs.set('/a/b', 10)) # trigger 2 callbacks and True
```

``` cpp
// 没有实现watch
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;
class Node {
public:
  int val;
  
  unordered_map<string, Node*> m;
  Node(int val) : val(val) {};
  
};
class FileSystem {
private:
  Node* root;
  vector<string> convert(string s) {
    vector<string> ret;
    int n = s.size();
    string ans;
    for (auto i : s) {
      if (i == '/') {
        if (ans != "") ret.push_back(ans);
        ans = "";
      }
      else {
        ans.push_back(i);
      }
    }
    ret.push_back(ans);
    return ret;
  }
  
public:
  
  FileSystem() : root(new Node(-1)) {};
  
  int create(string s, int val) {
    auto nums = convert(s);
    auto cur = root;
    int n = nums.size();
    for (int i = 0; i < n - 1; i++) {
      if (cur->m.find(nums[i]) == cur->m.end()) return -1;
      cur = cur->m[nums[i]];
    }
    cur->m[nums.back()] = new Node(val);
    return 1;
  }
  
  int get(string s) {
    auto nums = convert(s);
    auto cur = root;
    int n = nums.size();
    for (auto i : nums) {
      if (cur->m.find(i) == cur->m.end()) return -1;
      cur = cur->m[i];
    }
    return cur->val;
  }
  
};

int main() {
  FileSystem* t = new FileSystem();
  cout << t->create("/a", 1) << endl;
  cout << t->get("/a") << endl;
  cout << t->create("/a/b", 2) << endl;
  cout << t->create("/b/b", 3) << endl;
  return 0;
}
```

## 正则表达式
```
包含+,.,*
```
``` cpp
#include <vector>
#include <iostream>


using namespace std;

bool helper(string s, string p) {
  int m = s.size(), n = p.size();
  vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
  
  dp[0][0] = true;
  for (int i = 1; i <= n; i++) {
    dp[0][i] = p[i - 1] == '*' ? dp[0][i - 2] : false;
  }
  
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      if (p[j - 1] == '*') {
        if (s[i - 1] != p[j - 2] && p[j - 2] != '.') dp[i][j] = dp[i][j - 2];
        else {
          dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
        }
      }
      else if (p[j - 1] == '+') {
        if (s[i - 1] == p[j - 2] || p[j - 2] == '.') {
          dp[i][j] = dp[i - 1][j - 2] || dp[i - 1][j];
        }
      }
      else {
        dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
      }
    }
  }
  return dp[m][n];
}


int main() {
  string s, p;
  cin >> s >> p;
  
  cout << helper(s, p) << endl;
  
  return 0;
}
```

## Maximum	Number	a	Night	You	Can	Accommodate
```
Leetcode 相似问题:	Leetcode	198/213/337	House	Robber	I/II/III
Provide	a	set	of	positive	integers	(an	array	of	integers).	Each	integer	represents	number	of	nights	user	
request	on	Airbnb.com.	If	you	are	a	host,	you	need	to	design	and	implement	an	algorithm	to	find	out	the	
maximum	number	a	night	you	can	accommodate.	The	constrain	is	that	you	have	to	reserve	at	least	one	
day	between	each	request,	so	that	you	have	time	to	clean	the	room.	
Given	a	set	of	numbers	in	an	array	which	represent	number	of	consecutive	days	of	AirBnB	reservation	
requested,	as	a	host,	pick	the	sequence	which	maximizes	the	number	of	days	of	occupancy,	at	the	same	
time,	leaving	at	least	1	day	gap	in	between	bookings	for	cleaning.	Problem	reduces	to	finding	max-sum	
of	non-consecutive	array	elements.
//	[5,	1,	1,	5]	=>	10
The	above	array	would	represent	an	example	booking	period	as	follows	-
//	Dec	1	– 5
//	Dec	5	– 6
//	Dec	6	– 7
//	Dec	7	- 12
The	answer	would	be	to	pick	dec	1-5	(5	days)	and	then	pick	dec	7-12	for	a	total	of	10	days	of	occupancy,	
at	the	same	time,	leaving	at	least	1	day	gap	for	cleaning	between	reservations.
Similarly,
//	[3,	6,	4]	=>	7
//	[4,	10,	3,	1,	5]	=>	15
```

``` cpp
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

int helper(vector<int>& nums) {
    int n = nums.size();
    if (!n) return 0;
    int ret = 0;
    vector<int> dp(n + 1, 0);
    if (n == 1) return nums[0];
    if (n == 2) return max(nums[0], nums[1]);
    dp[1] = nums[0], dp[2] = nums[1];
    for (int i = 2; i <= n; i++) {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        ret = max(ret, dp[i]);
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    cout << helper(nums) << endl;

    return 0;
}
```

## Find	Case	Combinations	of	a	String
```
Find	all	the	combinations	of	a	string	in	lowercase	and	uppercase.	For	example,	string	"ab"	>>> "ab",	
"Ab",	"aB",	"AB".	So,	you	will	have	2^n	(n	=	number	of	chars	in	the	string)	output	strings.	The	goal	is	for	
you	to	test	each	of	these	strings and	see	if	it	matchs a	hidden	string.
```

``` cpp
#include <iostream>
#include <vector>

using namespace std;

char convert(char a) {
  if (a <= 'Z' && a >= 'A') {
    return 'a' + (a - 'A');
  }
  else {
    return 'A' + (a - 'a');
  }
}

void helper(string s, int index, vector<string>& ret) {
  int n = s.size();
  if (index == n) {
    ret.push_back(s);
    return;
  }
  
  helper(s, index + 1, ret);
  s[index] = convert(s[index]);
  helper(s, index + 1, ret);
}


int main() {
  string s;
  cin >> s;
  
  vector<string> ret;
  helper(s, 0, ret);
  
  for (auto i : ret) {
    cout << i << endl;
  }
  
  return 0;
}
```

## 两两编辑距离
Find	all	words	from	a	dictionary	that	are	k edit	distance	away. 

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <climits>
#include <math.h>
using namespace std;

int editDistance(string a, string b) {
  int m = a.size(), n = b.size();
  vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MAX));
  
  for (int i = 0; i <= m; i++) dp[i][0] = i;
  for (int i = 0; i <= n; i++) dp[0][i] = i;
  
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
      dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]));
    }
  }
  return dp[m][n];
}

vector<pair<string, string>> helper(vector<string>& nums, int k) {
  vector<pair<string, string>> ret;
  int n = nums.size();
  
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      auto t = editDistance(nums[i], nums[j]);
      if (t <= k && !((k - t) & 1)) 
        ret.push_back({nums[i], nums[j]});
    }
  }
  
  return ret;
}

int main() {
  
  int n, k;
  cin >> n >> k;
  
  vector<string> nums;
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    nums.push_back(s);
  }
  
  for (auto i : helper(nums, k)) {
    cout << i.first << " " << i.second << endl; 
  }
  
  return 0;
}

/*
3
5               
horse ros hourse
horse ros
*/
```

## Finding	Ocean
```
/**
*	Given:	An	array	of	strings	where	L	indicates	land	and	W	indicates	water,
* and	a	coordinate	marking	a	starting	point	in	the	middle	of	the	ocean.
*
*	Challenge:	Find	and	mark	the	ocean	in	the	map	by	changing	appropriate	Ws	to	Os.
* An	ocean	coordinate	is	defined	to	be	the	initial	coordinate	if	a	W,	and
* any	coordinate	directly	adjacent	to	any	other	ocean	coordinate.
*
*	void	findOcean(String[]	map,	int	row,	int	column);
*
*	String[]	map	=	new	String[]{
* "WWWLLLW",
* "WWLLLWW",
* "WLLLLWW"
*	};
*	printMap(map);
*
*	STDOUT:
*	WWWLLLW
*	WWLLLWW
*	WLLLLWW
*
*	findOcean(map,	0,	1);
*
*	printMap(map);
*
*	STDOUT:
*	OOOLLLW
*	OOLLLWW
*	OLLLLWW
*/
```

``` cpp
#include <iostream>
#include <utility>
#include <vector>
using namespace std;


vector<string> helper(vector<string>& nums, int x, int y) {
  
  int m = nums.size(), n = nums[0].size();
  
  if (x >= 0 && x < m && y >= 0 && y < n);
  else return nums;
  vector<pair<int, int>> cur;
  cur.push_back({x, y});
  
  int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
  
  while (true) {
    vector<pair<int, int>> next;
    
    for (auto i : cur) {
      nums[i.first][i.second] = 'O';
      
      for (int j = 0; j < 4; j++) {
        int X = i.first + a[j], Y = i.second + b[j];
        if (X >= 0 && X < m && Y >= 0 && Y < n && nums[X][Y] == 'W') {
          next.push_back({X, Y});
        }
      }
    }
    
    if (next.empty()) break;
    else cur = next;
    
  }
  
  return nums;
  
}

int main() {
  // int m;
  // cin >> m;
  
  vector<string> nums({"WWWLLLW","WWLLLWW","WLLLLWW"});
  // for (int i = 0; i < m; i++) {
  //   string temp;
  //   cin >> temp;
  //   nums.push_back(temp);
  // }
  
  int x = 0, y = 1;
  
  for (auto i : helper(nums, x, y)) {
    cout << i << endl;
  }
  
  return 0;
}
```

## 最大的矩形
二维数组中只有0和1，寻找数组中最大的1矩形
``` cpp
// stack
#include <vector>
#include <iostream>
#include <math.h>
#include <climits>
#include <stack>
using namespace std;

int helper(vector<vector<char>>& nums) {
    if (nums.empty() || nums[0].empty()) return 0;
    int m = nums.size(), n = nums[0].size();
    
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        int ans = 0;
        for (int j = 0; j < n; j++) {
            if (nums[i][j] == '1') {
                ans ++;
                dp[i][j] = ans;
            }
            else {
                ans = 0;
            }
        }
    }
    int ret = 0;
    for (int i = 0; i < n; i++) {
        stack<int> s;
        for (int j = 0; j < m; j++) {
            int cnt = 0;
            while (!s.empty() && s.top() > dp[j][i]) {
                cnt ++;
                ret = max(ret, cnt * s.top());
                s.pop();
            }
            while (cnt >= 0) {
                cnt --;
                s.push(dp[j][i]);
            }
        }
        int cnt = 0;
        while (!s.empty()) {
            cnt ++;
            ret = max(ret, cnt * s.top());
            s.pop();
        }
    }
    return ret;
}

int main() {
    int m, n;
    cin >> m >> n;
    
    vector<vector<char>> nums(m, vector<char>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }

    cout << helper(nums) << endl;

    return 0;
}
/*
6 6
101101
111111
011011
111010
011111
110111

*/


// dp
#include <vector>
#include <iostream>
#include <math.h>
#include <climits>
#include <stack>
using namespace std;

int helper(vector<vector<char>>& nums) {
    if (nums.empty() || nums[0].empty()) return 0;
        
    int m = nums.size(), n = nums[0].size();
    int ret = 0;
    vector<int> height(n, 0), left(n, 0), right(n, n);
    for (int i = 0; i < m; i++) {
        int index_l = 0, index_r = n - 1;
        
        for (int j = 0; j < n; j++) {
            if (nums[i][j] == '1') height[j]++;
            else height[j] = 0;
        }
        
        for (int j = 0; j < n; j++) {
            if (nums[i][j] == '1') left[j] = max(left[j], index_l);
            else left[j] = 0, index_l = j + 1;
        }
        for (int j = n - 1; j >= 0; j--) {
            if (nums[i][j] == '1') right[j] = min(right[j], index_r);
            else right[j] = n, index_r = j - 1;
        }
        for (int j = 0; j < n; j++) {
            ret = max(ret, (right[j] - left[j] + 1) * height[j]);
        }
    }
    return ret;
}

int main() {
    int m, n;
    cin >> m >> n;
    
    vector<vector<char>> nums(m, vector<char>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }

    cout << helper(nums) << endl;

    return 0;
}
/*
6 6
101101
111111
011011
111010
011111
110111

*/
```

## 规律打印
```
Input:1
Output
 /\ 
/__\


Input:2
Output
   /\   
  /__\  
 /\  /\ 
/__\/__\


Input:3
Output
       /\       
      /__\      
     /\  /\     
    /__\/__\    
   /\      /\   
  /__\    /__\  
 /\  /\  /\  /\ 
/__\/__\/__\/__\ 
```

``` cpp
#include <vector>
#include <iostream>

using namespace std;

vector<string> helper(int iter) {
    if (iter == 1) {
        return vector<string>({
            " /\\ ",
            "/__\\"
        });
    }
    vector<string> ans = helper(iter - 1);
    int m = ans.size();
    int n = ans[0].size();
    vector<string> ret(m * 2, string(n * 2, ' '));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret[i][j + n / 2] = ans[i][j];
            ret[i + m][j] = ans[i][j];
            ret[i + m][j + n] = ans[i][j];
        }
    }
    return ret;
}

void print(vector<string> nums) {
    for (auto i : nums) {
        cout << i << endl;
    }
}

int main() {
    int n;
    cin >> n;
    print(helper(n));
    return 0;
}
``` 


