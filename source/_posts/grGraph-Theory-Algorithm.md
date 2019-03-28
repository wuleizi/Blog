---
title: 图相关内容总结
date: 2018-06-21 23:40:05
tags: [算法, 总结]
---
> 此部分是遇到的图论相关算法总结，正在不断更新....
<!-- more -->
> 关于图论的相关基础知识和每个算法的具体推导，可以参考[演算法笔记](http://www.csie.ntnu.edu.tw/~u91029/)中的*Graph Theory*和*Combinatorial Optimization*部分。

# 路径相关算法

## 欧拉回路和欧拉路径

欧拉回路是指不令笔离开纸面，可画过图中每条边仅一次，且可以回到起点的一条回路。同样，欧拉路径就是从一点出发，能遍历所有图中的边，从而形成的路径。

判断欧拉回路的条件从图论里面可以知道，是整个图连通，然后每个点的入度等于出度就可以确定从任何一点出发都可遍历所有图。

寻找欧拉回路和路径的算法，也是一种贪心，每次尽量选择出度未走路径多的点作为下一个遍历的点，否则如果选择了该点，则无法再回来。因为这个算法是一种贪心，又保证连通，所以可以用栈保存行走的路径，如果不能再走向其他点，则说明不能再走到别的点，此时说明路径无法再扩展路径已经形成，所以。

> 因此如果用dfs回溯也是可以一边迭代完成。

欧拉路径的算法验证可以参考[leetcode332](https://leetcode.com/problems/reconstruct-itinerary/description/)

这里给出leetcode332的算法解答，也可以看做模板：
```cpp
class Solution {
public:
    vector<string> findItinerary(vector<pair<string, string>> tickets) {
        vector<string> ret;
        unordered_map<string, vector<string>> map;
        for (auto i : tickets) {
            map[i.first].push_back(i.second);
        }
        for (auto &i : map) {
            sort(i.second.begin(), i.second.end());
        }
        stack<string> s;
        s.push("JFK");
        while (!s.empty()) {
            auto temp = s.top();
            if (map[temp].empty()) {
                ret.push_back(temp);
                s.pop();
            }
            else {
                s.push(map[temp][0]);
                map[temp].erase(map[temp].begin());
            }
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }
};
```
## 单源最短路径算法

单元最短路径的意思为给定七点，求出起点到图上每个点的最短路径，一对多。

通常求最短路径的思路有两种
- 第一种是逐步确定每个点的最短路径长度，一旦确定后就不会再更改了，负边不适用。
- 第二种是某点确定最短路径长度之后，还要不断修正，整个过程就是不断修正的过程，负边也适用。

### Dijkstra算法
> 最短路径最经典的算法，是贪心发的一种应用，对于处理非负权的图比较有效。

> 同样，此算法也可以用于计算最长路，但是需要把所有的权值都改成负数。

Dijkstra算法模板
``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
using namespace std;

vector<int> helper(vector<vector<int>>& matrix, vector<int>& parents, vector<unordered_set<int>>& adj) {
    int n = parents.size();

    unordered_set<int> s;       // 没有访问的点
    vector<int> dp(n, INT_MAX); // 保存当前每个点最短距离
    for (int i = 0; i < n; i++) {
        s.insert(i);
    }
    // 初始化
    dp[0] = 0;
    parents[0] = 0;
    while (!s.empty()) {
        int index, ans = INT_MAX;
        // 查找当前距离起点最近的未访问点
        for (auto i : s) {
            if (ans > dp[i]) {
                ans = dp[i];
                index = i;
            }
        }
        // 访问该节点
        s.erase(index);
        // 根据已经访问的节点更新距离 relaxation
        for (auto j : adj[index]) {
            if (matrix[index][j] + dp[index] < dp[j]) {
                // 这里不需要判断是不是已经访问过了，因为根据贪心保存过得一定是最小的
                dp[j] = matrix[index][j] + dp[index];
                parents[j] = index;
            }
        }
    }
    return dp;
}

void find_path(vector<int>& parents, int x) {
    if (x != parents[x]) find_path(parents, parents[x]);
    cout << x << " ";
}
int main() {
    int n, k;
    cin >> n >> k;
    vector<vector<int>> matrix(n, vector<int>(n, INT_MAX));
    vector<unordered_set<int>> edges(n, unordered_set<int>());
    for (int i = 0; i < k; i++) {
        int x, y, d;
        cin >> x >> y >> d;
        matrix[x][y] = d;
        matrix[y][x] = d;
        edges[x].insert(y);
        edges[y].insert(x);
    }
    vector<int> parents(n, 0);
    vector<int> dp = helper(matrix, parents, edges);
    for (int i = 0; i < n; i++) {
        cout << i << ":" << dp[i] << endl;
        find_path(parents, i);
        cout << endl;
    }
    return 0;
}
/*
5 7
0 1 100
0 2 30
0 4 10
2 1 60
2 3 60
3 1 10
4 3 50
*/


```


Dijkstra看图论书容易陷入误区，会以为每次保存最新纳入的节点作为下次搜索的起始节点，然后进行下次搜索最小值并更新。然而这搞错了步骤，首先，更新过距离的未访问点已经是该点当前距离起点最近的距离了，从其中选择最短路然后访问，再通过当前最短路更新其余点的最短路，这样稳定的进行贪心算法。

当前代码的时间复杂度是O(V<sup>2</sup>)，但是如果使用V个元素的斐波那契堆，用decrease key函数来进行relaxtion，使用extract min来找下一个点，就可以将时间简化到O(E+VlgV)，所以理论上Dijkstra算法为O(E+VlgV)，此处的代码之后更新。

关于单源最短路径还可以有更多优化，由于面试中可能涉及的较少这里不做总结，感兴趣的同学可以参考[最短路径的优化](http://www.csie.ntnu.edu.tw/~u91029/Path.html)。

### Dijkstra算法扩展
给定起始点，寻找其间第K小的路径。

参考文献为[csdn](https://blog.csdn.net/sharpdew/article/details/446510)

代码模板参考[链接](https://sourceforge.net/projects/ksp/)

### SPFA算法
> 这部分内容参考资料[台湾师范大学的推算法笔记](http://www.csie.ntnu.edu.tw/~u91029/Path2.html)，这里只做算法实现的总结。

主题思想是用BFS的思想，不断访问已经访问的点并扩展到邻节点，并更新节点距离，同时要检测负环，防止负环边数大于V-1而循环到负无穷。如果路径中有负环，那么负环无限循环路径权值就为负无穷，也就不存在最短路径，算法结束。

SPFA算法模板：
``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <climits>
using namespace std;

void find_negative_cycle(int index, vector<int>& parents, int n) {
    cout << "负环:" << endl;
    int x = parents[index];
    while (index != parents[x]) {
        cout << x << endl;
        x = parents[x];
    }
}

vector<int> helper(vector<vector<int>>& matrix, vector<int>& parents, vector<unordered_set<int>>& adj) {
    int n = parents.size();
    vector<bool> inqueue(n, false);   // 已经在队列中
    vector<int> dp(n, INT_MAX); // 保存当前每个点最短距离
    vector<int> cnt(n, 0);      // 记录最短路径的边数

    // 初始化
    dp[0] = 0;
    parents[0] = 0;
    cnt[0] = 0;

    queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int index = q.front();
        q.pop();
        inqueue[index] = false;
        if (inqueue[parents[index]])
        // 加速：queue中已经有了parents[index]，舍弃index继续，说明index已经稳定
            continue;
        // 根据已经访问的节点更新距离
        for (auto j : adj[index]) {
            if (matrix[index][j] + dp[index] < dp[j]) {
                dp[j] = matrix[index][j] + dp[index];
                parents[j] = index;
                cnt[j] = cnt[index] + 1; // 边数增加

                if (cnt[j] >= n) {
                    // 有负环
                    find_negative_cycle(j, parents, n);
                    return dp;
                }
                if (!inqueue[j]) {
                    // 如果队列中没有就加入
                    inqueue[j] = true;
                    q.push(j);
                }
            }
        }
    }
    return dp;
}


void find_path(vector<int>& parents, int x) {
    if (x != parents[x]) find_path(parents, parents[x]);
    cout << x << " ";
}
int main() {
    int n, k;
    cin >> n >> k;
    vector<vector<int>> matrix(n, vector<int>(n, INT_MAX));
    vector<unordered_set<int>> edges(n, unordered_set<int>());
    for (int i = 0; i < k; i++) {
        int x, y, d;
        cin >> x >> y >> d;
        matrix[x][y] = d;
        matrix[y][x] = d;
        edges[x].insert(y);
        edges[y].insert(x);
    }
    vector<int> parents(n, 0);
    vector<int> dp = helper(matrix, parents, edges);
    for (int i = 0; i < n; i++) {
        cout << i << ":" << dp[i] << endl;
        find_path(parents, i);
        cout << endl;
    }
    return 0;
}

```


## 图的直径
### 树形图的直径
> 这个算法也是较为经典的算法，使用BFS从所有悬挂点出发，并把悬挂点剥离，最后根据BFS迭代的次数判断无向图的最长路，也就是图的直径。
> 注意此算法的前提是保证无向图为连通图，如果不是连通图首先要判断一下连通性。

> 验证地址[leetcode 310](https://leetcode.com/problems/minimum-height-trees/description/)

> 从方法无法处理有环图，因为有环存在只有一个叶子节点的情况。

``` cpp
#include <iostream>
#include <vector>
#include <unordered_set>
#include <utility>
using namespace std;


int helper(int n, vector<pair<int, int>>& grap) {
    if (n == 1) return 0;
    vector<unordered_set<int>> count(n, unordered_set<int>());
    for (auto i : grap) {
        if (i.first == i.second)
            continue;
        // 去掉环
        count[i.first].insert(i.second);
        count[i.second].insert(i.first);
    }
    int ret = 0;
    vector<int> current;
    for (int i = 0; i < n; i++) {
        int len = count[i].size();
        if (len == 1)
            current.push_back(i);
    }
    while (true) {
        vector<int> next;
        ret ++;
        for (auto i : current) {
            // 默认所有current中的点都是悬挂点
            // 所以遍历叶子节点的相邻点并去掉与之相连的边，这些边就不能被访问到。
            for (auto j : count[i]) {
                count[j].erase(i);
                if (count[j].size() == 1)
                    next.push_back(j);
            }
        }
        if (next.empty()) break;
        current = next;
    }
    // 终止集合是单点是迭代次数的二倍，集合为两个点还要多算连接这两个点的边
    // 因为终止条件是next为空，所以会将终止集合也多算一次迭代，所以需要减去。
    return current.size() == 1 ? ret * 2 - 2: ret * 2 - 1;
}


int main() {
    int n, k;
    cin >> k >> n;
    vector<pair<int, int>> grap;
    for (int i = 0; i < k; i ++) {
        int x, y;
        cin >> x >> y;
        grap.push_back(make_pair(x, y));
    }
    cout << helper(n, grap) << endl;
    return 0;
}
/*
5 4
0 1
1 2
2 3
0 2
3 3
*/
```

### 无向图的直径
如果是要求非树形连通图，就需要定义几个概念：

- <b> 偏心距 </b>：以最短路径长度作为距离，一张无向图中距离一点最远的距离被称为该点的偏心距。
- <b> 直径与半径 </b>: 一张无向图的直径是所有偏心距中最大的一个，半径是途中所有偏心距离中最小的一个。直径也可以直接认为是图上最长的一条最短路径的长度。


直径与半径代码模板（Floyd-Warshall算法）:
``` cpp
#include <iostream>
#include <vector>
#include <climits>
#include <cmath>
using namespace std;
void helper(int n, vector<vector<int>>& matrix, int& d, int& r) {
    // Floyd-Warshall Algorithm
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
            // 注意溢出
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j]);
            }
        }
    }
    // 计算偏心距
    vector<int> ecc(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] != 1E9) {
                ecc[i] = max(ecc[i], matrix[i][j]);
            }
        }
    }

    // 半径与直径
    d = 0, r = 1E9;
    for (int i = 0; i < n; i++) {
        d = max(d, ecc[i]);
        r = min(r, ecc[i]);
    }

}

int main() {
    int n, k;
    cin >> n >> k;
    vector<vector<int>> matrix(n, vector<int>(n, 1E9));
    for (int i = 0; i < k; i++) {
        int x, y, v;
        cin >> x >> y >> v;
        matrix[x][y] = v;
        matrix[y][x] = v;
    }
    int d, r;
    helper(n, matrix, d, r);
    cout << "diameter: " << d << endl;
    cout << "radius: " << r << endl;
    return 0;
}


```

# 图的连通性
> 使用并查集，测试样例可以参考[cnblog](https://www.cnblogs.com/wxjor/p/5713402.html)

这里先给出并查集的模板：
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int find(int x, vector<int>& set) {
    int y = x;
    while (set[x] != x) {
        x = set[x];
    }
    while (x != y) {
        int t = set[y];
        set[y] = x;
        y = t;
    }
    return x;
}

void merge(int x, int y, vector<int>& set) {
    int p1 = find(x, set);
    int p2 = find(y, set);
    if (p1 != p2) {
        set[p1] = p2;
    }
}
int main() {
    int n, k;
    cin >> n >> k;
    vector<int> set(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        set[i] = i;
    }
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >> y;
        merge(x, y, set);
    }
    int ret = 0;
    for (int i = 1; i <= n; i++) {
        if (set[i] == i) ret ++;
    }
    cout << ret << endl;
    return 0;
}

```


使用图结构后计算并查集并计算连通分量：
``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;


int find(int x, vector<int>& set) {
    int y = x;
    while (x != set[x]) {
        x = set[x];
    }
    while (y != x) {
        int t = set[y];
        set[y] = x;
        y = t;
    }
    return x;
}

void merge(int x, int y, vector<int>& set) {
    int p1 = find(x, set);
    int p2 = find(y, set);
    if (p1 != p2) {
        set[p1] = p2;
    }
}

void helper(vector<unordered_set<int>>& edges) {
    int n = edges.size();
    vector<int> set(n, 0);
    for (int i = 0; i < n; i++) {
        set[i] = i;
    }
    for (int i = 0; i < n; i++) {
        for (auto j : edges[i]) {
            merge(i, j, set);
        }
    }
    vector<unordered_set<int>> ret(n, unordered_set<int>());
    for (int i = 0; i < n; i++) {
        ret[set[i]].insert(i);
    }
    for (auto i : ret) {
        if (i.empty()) continue;
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<unordered_set<int>> edges(n, unordered_set<int>());
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >> y;
        edges[x].insert(y);
        edges[y].insert(x);
    }
    helper(edges);
    return 0;
}


/*
4 5
0 1
0 2
1 2
1 1
3 3
*/

```

# 拓扑排序
拓扑排序的要点是要注意加入队列的条件，需要入度为0的时候才可以加入。可以使用leetcode的[Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/description/)进行验证。


``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
#include <unordered_set>

using namespace std;


void helper(int n, vector<pair<int, int>>& edges) {
    vector<int> degree(n, 0);
    vector<unordered_set<int>> count(n, unordered_set<int>());
    for (auto i : edges) {
        degree[i.second]++;
        count[i.first].insert(i.second);
    }
    vector<int> cur;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 0) {
            cur.push_back(i);
        }
    }
    vector<vector<int>> ret;
    while (!cur.empty()) {
        vector<int> ans;
        for (auto i : cur) {
            for (auto j : count[i]) {
                degree[j] --;
                if (degree[j] == 0) ans.push_back(j);
            }
        }
        ret.push_back(cur);
        cur = ans;
    }
    for (auto i : ret) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> edges;
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >>y;
        edges.push_back(make_pair(x, y));
    }
    helper(n, edges);
    return 0;
}

```
# 网络流相关算法
## 最大流
> 本文的参考文献是[最大流（网络流基础概念+三个算法）](https://blog.csdn.net/x_y_q_/article/details/51999466)，可以用[POJ1273](http://poj.org/problem?id=1273)验证。

### EK（Edmond—Karp）算法

第一种算法EK（Edmond—Karp）算法，主要的要去是每做一次更新都会建立一条反向的边，从而建立返回的渠道减少替代回溯的高消耗。模板如下
``` cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;


int BFS(vector<int>& pre, int start, int end, vector<vector<int>>& cap) {
    queue<int> q;
    int n = cap.size();
    for (int i = 0; i < n; i++) {
        pre[i] = -1;
    }
    vector<int> flow(n, 0); // 保存流
    // 初始化
    pre[start] = start;
    flow[start] = 0x7FFFFFFF;
    q.push(start);
    while (!q.empty()) {
        // 用bfs搜索整个路
        int index = q.front();
        q.pop();
        if (index == end) break; // 到达终点为可增路
        for (int i = 0; i < n; i++) {
            if (i != start && cap[index][i] > 0 && pre[i] == -1) {
                // 更新流
                pre[i] = index;
                flow[i] = min(flow[index], cap[index][i]);
                q.push(i);
            }
        }
    }
    // 可增路无法到达终点
    if (pre[end] == -1) return -1;
    else return flow[end];
}

int helper(int n, vector<vector<int>>& cap, vector<int>& pre, int start, int end) {
    int ret = 0;
    int increase = 0; // 单次可增路的增大流容量
    while ((increase = BFS(pre, start, end, cap)) != -1) { // 返回-1代表没有可增路
        int temp = end;
        while (temp != start) {
            // 更新可增路上的容量
            int last = pre[temp];
            cap[last][temp] -= increase; // 反向也要增加
            cap[temp][last] += increase;
            temp = last;
        }
        ret += increase;
    }
    return ret;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> cap(n, vector<int>(n, 0));
    vector<int> pre(n, -1); // 用来标记可增路的路径
    for (int i = 0; i < m; i++) {
        int x, y, v;
        cin >> x >> y >> v;
        if (x == y) continue;
        cap[x][y] = v;
    }
    cout << helper(n, cap, pre, 0, n - 1) << endl;;
    return 0;
}


/*
4 5
0 1 40
1 2 30
1 3 20
0 3 20
2 3 10
*/

```

### Ford-Fulkerson算法
Ford-Fulkerson算法可认为是DFS版本的EK，由于是递归，会带来空间消耗。

``` cpp
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int dfs(int start, int end, vector<vector<int>>& cap, vector<bool>& visited, int flow) {
    int n = visited.size();
    if (start == end) {
            return flow;
    }
    for (int i = 0; i < n; i++) {
        if (cap[start][i] > 0 && !visited[i]) {
            visited[i] = true;
            int f = dfs(i, end, cap, visited, min(flow, cap[start][i]));
            if (f > 0) {
                cap[start][i] -= f;
                cap[i][start] += f;
                return f;
            }

        }
    }
    // 没有可增路返回0
    return 0;
}

int helper(int n, vector<vector<int>>& cap, int start, int end) {
    int ret = 0;
    while (true) {
        vector<bool> visited(n, false);
        int flow = dfs(start, end, cap, visited, 0x7FFFFFFF);
        if (flow == 0) break;
        ret += flow;

    }
    return ret;

}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> cap(n, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        int x, y, v;
        cin >> x >> y >> v;
        cap[x][y] = v;
    }
    cout << helper(n, cap, 0, n - 1) << endl;
    return 0;
}
/*
4 5
0 1 40
1 2 30
1 3 20
0 3 20
2 3 10
*/
```
### Dinic算法
Dinic算法是网络流最大流的优化算法之一，每一步对原图进行分层，然后用DFS求增广路。时间复杂度是O(n^2*m)，Dinic算法最多被分为n个阶段，每个阶段包括建层次网络和寻找增广路两部分。

Dinic算法的思想是分阶段地在层次网络中增广。它与最短增广路算法不同之处是：最短增广路每个阶段执行完一次BFS增广后，要重新启动BFS从源点Vs开始寻找另一条增广路;而在Dinic算法中，只需一次BFS过程就可以实现多次增广。

观察前面的dfs算法，对于层次相同的边，会经过多次重复运算，很浪费时间，那么，可以考虑先对原图分好层产生新的层次图，即保存了每个点的层次，注意，很多人会把这里的边的最大容量跟以前算最短路时的那个权值混淆，其实这里每个点之间的距离都可以看作单位距离，然后对新图进行dfs，这时的dfs就非常有层次感，有筛选感了，同层次的点不可能在同一跳路径中，直接排除。那么运行速度就会快很多了。

Dinic算法模板：
``` cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>

using namespace std;

int bfs(int start, int end, vector<vector<int>>& cap, vector<int>& dep) {
    queue<int> q;
    q.push(start);
    int n = dep.size();
    for (int i = 0; i < n; i++) {
        dep[i] = -1;
    }
    dep[start] = 0;
    while (!q.empty()) {
        int index = q.front();
        q.pop();
        for (int i = 0; i < n; i++) {
            if (cap[index][i] > 0 && dep[i] == -1) {
                dep[i] = dep[index] + 1;
                q.push(i);
            }
        }
    }
    return dep[end] != -1;
}


int dfs(int start, int end, vector<vector<int>>& cap, vector<int>& dep, int flow) {
    if (start == end) return flow;
    int n = dep.size();
    for (int i = 0; i < n; i++) {
        if (cap[start][i] > 0 && dep[i] == dep[start] + 1) {
            int temp = dfs(i, end, cap, dep, min(flow, cap[start][i]));
            if (temp != 0) {
                cap[start][i] -= temp;
                cap[i][start] += temp;
                return temp;
            }
        }
    }
    return 0;
}

int helper(int n, int start, int end, vector<vector<int>>& cap) {
    int ret = 0;
    vector<int> dep(n, -1);
    while (bfs(start, end, cap, dep)) {
        while (true) {
            int temp = dfs(start, end, cap, dep, 0x7FFFFFFF);
            if (!temp) break;
            ret += temp;
        }
    }
    return ret;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> cap(n, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        int x, y, v;
        cin >> x >> y >> v;
        cap[x][y] = v;
    }
    cout << helper(n, 0, n - 1, cap) << endl;
    return 0;
}
/*
4 5
0 1 40
1 2 30
1 3 20
0 3 20
2 3 10
*/
```

> 最大流问题还有很多种变种，因为本身面试中涉及基础图论比较少，不做涉及，更多了解可以参考[资料](http://www.csie.ntnu.edu.tw/~u91029/Flow.html)

## 最小割
### SW（Stoer-Wagner）算法
最小割算法有很多算法，这里只提供一种模板，就是比较常用的Stoer-Wagner Algorithm，其中的推导和更多算法可以访问[资料](http://www.csie.ntnu.edu.tw/~u91029/Cut.html)

这个算法的验证地址是[POJ2914](http://poj.org/problem?id=2914)
模板如下（代码参考自[cnblog](https://blog.csdn.net/i_love_home/article/details/9698791)）：
``` cpp
const int maxn = 550;
const int inf = 1000000000;
int n, r;
int edge[maxn][maxn], dist[maxn];
bool vis[maxn], bin[maxn];
void init()
{
    memset(edge, 0, sizeof(edge));
    memset(bin, false, sizeof(bin));
}
int contract( int &s, int &t )          // 寻找 s,t
{
    memset(dist, 0, sizeof(dist));
    memset(vis, false, sizeof(vis));
    int i, j, k, mincut, maxc;
    for(i = 1; i <= n; i++)
    {
        k = -1; maxc = -1;
        for(j = 1; j <= n; j++)if(!bin[j] && !vis[j] && dist[j] > maxc)
        {
            k = j;  maxc = dist[j];
        }
        if(k == -1)return mincut;
        s = t;  t = k;
        mincut = maxc;
        vis[k] = true;
        for(j = 1; j <= n; j++)if(!bin[j] && !vis[j])
            dist[j] += edge[k][j];
    }
    return mincut;
}
int Stoer_Wagner()
{
    int mincut, i, j, s, t, ans;
    for(mincut = inf, i = 1; i < n; i++)
    {
        ans = contract( s, t );
        bin[t] = true;
        if(mincut > ans)mincut = ans;
        if(mincut == 0)return 0;
        for(j = 1; j <= n; j++)if(!bin[j])
            edge[s][j] = (edge[j][s] += edge[j][t]);
    }
    return mincut;
}

```


# BFS/DFS
待更新...
