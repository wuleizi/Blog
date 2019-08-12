---
title: Leetcode 周赛模拟 【1】
date: 2018-09-16 23:56:59
tags: [Leetcode, 周赛]
---

<!-- more -->
# Weekly Contest 85
> [参赛地址](https://leetcode.com/contest/weekly-contest-85/)

## 836. Rectangle Overlap [AC]
> [Leetcode 836](https://leetcode.com/contest/weekly-contest-85/problems/rectangle-overlap/)

A rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) are the coordinates of its bottom-left corner, and (x2, y2) are the coordinates of its top-right corner.

Two rectangles overlap if the area of their intersection is positive.  To be clear, two rectangles that only touch at the corner or edges do not overlap.

Given two (axis-aligned) rectangles, return whether they overlap.

Example 1:
```
Input: rec1 = [0,0,2,2], rec2 = [1,1,3,3]
Output: true
```
Example 2:
```
Input: rec1 = [0,0,1,1], rec2 = [1,0,2,1]
Output: false
```

``` cpp
class Solution {
public:
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
        long long x11 = rec1[0], x12 = rec1[2], x21 = rec2[0], x22 = rec2[2];
        bool r1 = max(x11, x21) < min(x12, x22);
        long long y11 = rec1[1], y12 = rec1[3], y21 = rec2[1], y22 = rec2[3];
        bool r2 = max(y11, y21) < min(y12, y22);
        return r1 && r2;
    }
};
```


## 838. Push Dominoes [AC]
> [Leetcode 838](https://leetcode.com/contest/weekly-contest-85/problems/push-dominoes/)

There are N dominoes in a line, and we place each domino vertically upright.

In the beginning, we simultaneously push some of the dominoes either to the left or to the right.

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/05/18/domino.png)

After each second, each domino that is falling to the left pushes the adjacent domino on the left.

Similarly, the dominoes falling to the right push their adjacent dominoes standing on the right.

When a vertical domino has dominoes falling on it from both sides, it stays still due to the balance of the forces.

For the purposes of this question, we will consider that a falling domino expends no additional force to a falling or already fallen domino.

Given a string "S" representing the initial state. S[i] = 'L', if the i-th domino has been pushed to the left; S[i] = 'R', if the i-th domino has been pushed to the right; S[i] = '.', if the i-th domino has not been pushed.

Return a string representing the final state. 

Example 1:
```
Input: ".L.R...LR..L.."
Output: "LL.RR.LLRRLL.."
```
Example 2:
```
Input: "RR.L"
Output: "RR.L"
Explanation: The first domino expends no additional force on the second domino.
```

``` cpp
class Solution {
public:
    string pushDominoes(string A) {
        char pre = 0;
        int index = -1, n = A.size();
        for (int i = 0; i < n; i++) {
            if (A[i] == 'L') {
                if (pre == 'R') {
                    int k = (index + i + 1) / 2 - index;
                    for (int j = 0; j < k; j++) {
                        A[j + index] = 'R', A[i - j] = 'L';
                    }
                }
                else {
                    for (int j = index + 1; j <= i; j++) {
                        A[j] = 'L';
                    }
                }
                pre = 'L';
                index = i;
            }
            else if (A[i] == 'R'){
                if (pre == 'R') {
                    for (int j = index + 1; j < i; j++) A[j] = 'R';
                }
                pre = 'R';
                index = i;
            }
        }
        if (pre == 'R') {
            for (int i = index + 1; i < n; i++) A[i] = 'R';
        }
        return A;
    }
};
```

## 837. New 21 Game [unsolved]
> [Leetcode 837](https://leetcode.com/contest/weekly-contest-85/problems/new-21-game/)

Alice plays the following game, loosely based on the card game "21".

Alice starts with 0 points, and draws numbers while she has less than K points.  During each draw, she gains an integer number of points randomly from the range [1, W], where W is an integer.  Each draw is independent and the outcomes have equal probabilities.

Alice stops drawing numbers when she gets K or more points.  What is the probability that she has N or less points?

Example 1:
```
Input: N = 10, K = 1, W = 10
Output: 1.00000
Explanation:  Alice gets a single card, then stops.
```
Example 2:
```
Input: N = 6, K = 1, W = 10
Output: 0.60000
Explanation:  Alice gets a single card, then stops.
In 6 out of W = 10 possibilities, she is at or below N = 6 points.
```
Example 3:
```
Input: N = 21, K = 17, W = 10
Output: 0.73278
```

``` java
class Solution {
    public double new21Game(int N, int K, int W) {
        double[] dp = new double[N+3];
        dp[0] = 1;
        dp[1] = -1;
        double val = 0;
        for(int i = 0;i < K;i++){
            val += dp[i];
            if(i+1 <= N)dp[i+1] += val / W;
            if(i+W+1 <= N)dp[i+W+1] -= val / W;
        }
        double ret = 0;
        for(int i = K;i <= N;i++){
            val += dp[i];
            ret += val;
        }
        return ret;
    }
}   
```

## 839. Similar String Groups [AC]
> [Leetcode 839](https://leetcode.com/contest/weekly-contest-85/problems/similar-string-groups/)


Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.

For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, but "star" is not similar to "tars", "rats", or "arts".

Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}.  Notice that "tars" and "arts" are in the same group even though they are not similar.  Formally, each group is such that a word is in the group if and only if it is similar to at least one other word in the group.

We are given a list A of strings.  Every string in A is an anagram of every other string in A.  How many groups are there?

Example 1:
```
Input: ["tars","rats","arts","star"]
Output: 2
```

``` cpp
// 并查集
class Solution {
public:
    
    bool helper(string a, string b) {
        if (a == b) return true;
        if (a.size() != b.size()) return false;
        if (a.empty()) return true;
        int cnt = 0;
        char x1 = '\0', y1 = '\0', x2 = '\0', y2 = '\0';
        for (int i = 0; i < a.size(); i++) {
            if (a[i] == b[i]) continue;
            cnt ++;
            if (cnt == 1) x1 = a[i], y1 = b[i];
            if (cnt == 2) x2 = a[i], y2 = b[i];
        }
        return cnt == 2 && x1 == y2 && x2 == y1;
        
    }
    
    int MyFind(vector<int>& nums, int x) {
        int y = x;
        while (nums[x] != x) {
            x = nums[x];
        }
        while (x != y) {
            int t = nums[y];
            nums[y] = x;
            y = t;
        }
        return x;
    }
    
    void MyMerge(vector<int>& nums, int x, int y) {
        int p1 = MyFind(nums, x);
        int p2 = MyFind(nums, y);
        if (p1 != p2) nums[p1] = p2;
    }
    
    int numSimilarGroups(vector<string>& A) {
        int n = A.size();
        if (!n) return 0;
        vector<int> nums(n, 0);
        for (int i = 0; i < n; i++) {
            nums[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                if (helper(A[i], A[j])) MyMerge(nums, i, j);
            }
        }
        int ret = 0;
        for (int i = 0; i < n; i++) {
            if (MyFind(nums, i) == i) ret ++;
        }
        return ret;
    }
};
```


# Weekly Contest 84
> [参赛地址](https://leetcode.com/contest/weekly-contest-84)

## 832. Flipping an Image [AC]
> [Leetcode 832](https://leetcode.com/contest/weekly-contest-84/problems/flipping-an-image)

Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting image.

To flip an image horizontally means that each row of the image is reversed.  For example, flipping [1, 1, 0] horizontally results in [0, 1, 1].

To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0. For example, inverting [0, 1, 1] results in [1, 0, 0].

Example 1:
```
Input: [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]
```

Example 2:
```
Input: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
Output: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
Explanation: First reverse each row: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]].
Then invert the image: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
```

``` cpp
class Solution {
public:
    vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
        int m = A.size(), n = A[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n / 2; j++) {
                swap(A[i][j], A[i][n - 1 - j]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] ^= 1;
            }
        }
        return A;
    }
};
```

## 833. Find And Replace in String [AC]
> [Leetcode 833](https://leetcode.com/contest/weekly-contest-84/problems/find-and-replace-in-string)

To some string S, we will perform some replacement operations that replace groups of letters with new ones (not necessarily the same size).

Each replacement operation has 3 parameters: a starting index i, a source word x and a target word y.  The rule is that if x starts at position i in the original string S, then we will replace that occurrence of x with y.  If not, we do nothing.

For example, if we have S = "abcd" and we have some replacement operation i = 2, x = "cd", y = "ffff", then because "cd" starts at position 2 in the original string S, we will replace it with "ffff".

Using another example on S = "abcd", if we have both the replacement operation i = 0, x = "ab", y = "eee", as well as another replacement operation i = 2, x = "ec", y = "ffff", this second operation does nothing because in the original string S[2] = 'c', which doesn't match x[0] = 'e'.

All these operations occur simultaneously.  It's guaranteed that there won't be any overlap in replacement: for example, S = "abc", indexes = [0, 1], sources = ["ab","bc"] is not a valid test case.

Example 1:
```
Input: S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
Output: "eeebffff"
Explanation: "a" starts at index 0 in S, so it's replaced by "eee".
"cd" starts at index 2 in S, so it's replaced by "ffff".
```

Example 2:
```
Input: S = "abcd", indexes = [0,2], sources = ["ab","ec"], targets = ["eee","ffff"]
Output: "eeecd"
Explanation: "ab" starts at index 0 in S, so it's replaced by "eee". 
"ec" doesn't starts at index 2 in the original S, so we do nothing.
```

``` cpp
class Solution {
public:
    bool static cmp(pair<int, int>& a, pair<int, int>& b) {
        return a.first < b.first;
    }
    string helper(string s, vector<pair<int, int>>& m, vector<string>& src, vector<string>& targets) {
        string ret = "";
        int n = s.size();
        for (int i = 0; i < n; i++) {
            if (!m.empty() && i == m[0].first) {
                auto t = m[0];
                m.erase(m.begin());
                string a = src[t.second], b = targets[t.second];
                if (i + a.size() <= n && a == s.substr(i, a.size())) {
                    ret += b;
                    i += a.size() - 1;
                    continue;
                }
            }
            ret.push_back(s[i]);
        }
        return ret;
    }
    string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
        vector<pair<int, int>> m;
        int n = indexes.size();
        for (int i = 0; i < n; i++) 
            m.push_back(make_pair(indexes[i], i));
        sort(m.begin(), m.end(), cmp);
        return helper(S, m, sources, targets);
    }
};
```

## 835. Image Overlap [AC]
> [Leetcode 835](https://leetcode.com/contest/weekly-contest-84/problems/image-overlap/)

Two images A and B are given, represented as binary, square matrices of the same size.  (A binary matrix has only 0s and 1s as values.)

We translate one image however we choose (sliding it left, right, up, or down any number of units), and place it on top of the other image.  After, the overlap of this translation is the number of positions that have a 1 in both images.

(Note also that a translation does not include any kind of rotation.)

What is the largest possible overlap?

Example 1:
```
Input: A = [[1,1,0],
            [0,1,0],
            [0,1,0]]
       B = [[0,0,0],
            [0,1,1],
            [0,0,1]]
Output: 3
Explanation: We slide A to right by 1 unit and down by 1 unit.
```

``` cpp
class Solution {
public:
    string convert(int i, int j) {
        return to_string(i) + "-" + to_string(j);
    }
    void helper(int i, int j, vector<vector<int>>& A, vector<vector<int>>& B, int& ret, unordered_set<string>& hash) {
        int m = A.size(), n = A[0].size();
        if (i <= -m || i >= m || j <= -n || j >= n) return;
        int ans = 0;
        for (int x = 0; x < m; x ++) {
            for (int y = 0; y < n; y++) {
                int X = x + i, Y = y + j;
                if (X < m && X >= 0 && Y < n && Y >= 0 && A[X][Y] == 1 && B[x][y] == 1) ans ++;
            }
        }
        ret = max(ret, ans);
        int a[4] = {0, 0, -1, 1}, b[4] = {1, -1, 0, 0};
        for (int k = 0; k < 4; k++) {
            int x = i + a[k], y = j + b[k];
            auto s = convert(x, y);
            if (hash.find(s) != hash.end()) continue;
            hash.insert(s);
            helper(x, y, A, B, ret, hash);
        }
        
    }
    int largestOverlap(vector<vector<int>>& A, vector<vector<int>>& B) {
        int ret = 0;
        unordered_set<string> m;
        m.insert("0-0");
        helper(0, 0, A, B, ret, m);
        return ret;
    }
};
```

## 834. Sum of Distances in Tree  [TLE]
> [Leetcode 834](https://leetcode.com/contest/weekly-contest-84/problems/sum-of-distances-in-tree/)

An undirected, connected tree with N nodes labelled 0...N-1 and N-1 edges are given.

The ith edge connects nodes edges[i][0] and edges[i][1] together.

Return a list ans, where ans[i] is the sum of the distances between node i and all other nodes.

Example 1:
```
Input: N = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
Output: [8,12,6,10,10,10]
Explanation: 
Here is a diagram of the given tree:
  0
 / \
1   2
   /|\
  3 4 5
We can see that dist(0,1) + dist(0,2) + dist(0,3) + dist(0,4) + dist(0,5)
equals 1 + 1 + 2 + 2 + 2 = 8.  Hence, answer[0] = 8, and so on.

```

``` cpp
// BFS 超时版本....
class Solution {
public:
    int helper(int start, vector<unordered_set<int>> adj, ) {
        vector<int> cur(1, start);
        int ret = 0, cnt = 0;
        while (!cur.empty()) {
            vector<int> next;
            cnt ++;
            for (auto i : cur) {
                for (auto j : adj[i]) {
                    ret += cnt;
                    adj[j].erase(i);
                    next.push_back(j);
                }
            }
            cur = next;
        }
        return ret;
    }
    vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges) {
        vector<unordered_set<int>> adj(N);
        for (auto t : edges) {
            adj[t[0]].insert(t[1]);
            adj[t[1]].insert(t[0]);
        }
        vector<int> ret;
        for (int i = 0; i < N; i++) {
            ret.push_back(helper(i, adj));
        }
        return ret;
    }
};
```

``` cpp
// 后用dfs保存中间状态AC
const int N = 1e4 + 10;
vector<int> a[N];
int dp0[N], dp1[N];

void DFS(int u, int parent) {
    dp0[u] = 1;
    dp1[u] = 0;
    for (auto& v : a[u]) {
        if (v == parent) continue;
        DFS(v, u);
        dp0[u] += dp0[v];
        dp1[u] += dp1[v] + dp0[v];
    }
}

void DFS2(int u, int parent, int n, vector<int>& ret) {
    for (auto& v : a[u]) {
        if (v == parent) continue;
        ret[v] = ret[u] + (n - dp0[v]) - (dp0[v]);
//        cout << u << " " << v << " " << ret[u] << " " << dp0[u] << " " << dp0[v] << " " << ret[v] << endl;
        DFS2(v, u, n, ret);
    }
}

class Solution {
public:
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        for (int i = 0; i < n; ++i) a[i].clear();
        for (auto& it : edges) {
            int x = it[0], y = it[1];
            a[x].push_back(y);
            a[y].push_back(x);
        }
        DFS(0, -1);
        vector<int> ret(n);
        ret[0] = dp1[0];
        DFS2(0, -1, n, ret);
        return ret;
    }
};
```