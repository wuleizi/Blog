---
title: 【面经】微软部分编程算法题
date: 2019-04-13 17:04:18
tags: [算法, 总结, 面经]
---

> 本文收集2018-2019年实验室实习面试部分编程面经以供复习
<!-- more -->
> 另外微软特别喜欢考剑指offer

## 股票题目1-5
### Best Time to Buy and Sell Stock
> 验证地址为[Leetcode 121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

``` cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;
        int ret = 0;
        int n = prices.size();
        int low = prices[0];
        for (int i = 1; i < n; i++) {
            low = min(low, prices[i]);
            ret = max(ret, prices[i] - low);
        }
        return ret;
    }
};
```

### Best Time to Buy and Sell Stock II
> 验证地址为[Leetcode 122](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

``` cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int ret = 0;
        for (int i = 1; i < n; i++) {
            if (prices[i] > prices[i - 1]) {
                ret += prices[i] - prices[i - 1];
            }
        }
        return ret;
    }
};
```
### Best Time to Buy and Sell Stock III
> 验证地址[Leetcode 123](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

``` cpp
// 正常解法，先从左往右扫一遍，算出到i为止一次交易收益最大值
// 然后从右往左扫，算出第二次交易与第一次交易加和最大值
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> dp(n, 0);
        int buy = INT_MAX;
        int sell = 0;
        // 第一次交易的最大值
        for (int i = 0; i < n; i++) {
           sell = max(sell, prices[i] - buy);
           buy = min(buy, prices[i]);
           dp[i] = sell;
        }
        int ret = 0;
        sell = 0;
        for (int i = n - 1; i >= 0; i--) {
            ret = max(ret, dp[i] + (sell - prices[i]));
            sell = max(sell, prices[i]);
        }
        return ret;
    }
};
```

``` cpp
// 自动机做法，每次遇到一个价格，先更新最终售出收益，然后按影响逐步更新其余三个值
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int buy1 = INT_MIN, buy2 = INT_MIN;
        int sell1 = 0, sell2 = 0;
        for (auto i : prices) {
            sell2 = max(sell2, buy2 + i);
            buy2 = max(buy2, sell1 - i);
            // 这里不好理解的是这一点，为什么取buy的最大值，
            // 通过表达式其实就能看出来，因为sell2是通过buy2+i更新的，
            // 所以这里保证局部最大就能保证加上去之后也是局部最大
            sell1 = max(sell1, i + buy1);
            buy1 = max(buy1, -i);
        }
        return sell2;
    }
};

```
### Best Time to Buy and Sell Stock IV
> 验证地址[Leetcode 188](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

``` cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<int> dp(n, 0);
        
        if (k >= n / 2) {
            int ret = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i - 1]) ret += prices[i] - prices[i - 1];
            }
            return ret;
        }
        
        for (int i = 1; i <= k; i++) {
            int buy = dp[0] - prices[0];
            for (int j = 1; j < n; j++) {
                int pre_sell = dp[j];
                dp[j] = max(dp[j - 1], buy + prices[j]);
                buy = max(buy, pre_sell - prices[j]);
            }
            // 或者二维数组
            /*
            int buy = dp[i - 1][0] - prices[0];
            for (int j = 1; j < n; j++) {
                dp[i][j] = max(dp[i][j - 1], buy + prices[j]);
                buy = max(buy, dp[i - 1][j] - prices[j]);
            }
            */
        }
        
        return dp[n - 1];
    }
};
```

### Best Time to Buy and Sell Stock with Cooldown
> 验证地址[Leetcode 309](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

``` cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int sell(0), buy(INT_MIN), pre_sell(0);
        for (auto i : prices) {
            int pre_buy = buy;
            buy = max(pre_sell - i, buy);
            pre_sell = sell;
            sell = max(sell, pre_buy + i);
        }
        return sell;
    }
};
```

``` cpp
// 方法二，便于理解，因为pre_sell必须要在sell之前的那一天
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n <= 1) return 0;
        int sell = 0, buy = INT_MIN, pre_sell = 0;
        for (int i = 0; i < n; i++) {
            int temp = pre_sell;
            pre_sell = sell;
            sell = max(sell, buy + prices[i]);
            buy = max(buy, temp - prices[i]);
        }
        return sell;
    }
};
```

### Best Time to Buy and Sell Stock with Transaction Fee
> 验证地址[Leetcode 714](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

``` cpp
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        if (n <= 1) return 0;
        int sell = 0, buy = INT_MIN;
        for (auto i : prices) {
            int pre_sell = sell;
            sell = max(sell, buy + i);
            buy = max(pre_sell - i - fee, buy);
        }
        return sell;
    }
};
```

## Open the Lock
> 验证地址[Leetcode 752](https://leetcode.com/problems/open-the-lock/)
``` cpp
// 加速方法是从两端出发，可以将速度加快一倍，也会使一端无法继续循环提前完成
class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        set<string> dead(deadends.begin(), deadends.end());
        if (dead.count("0000")) return -1;
        if (target == "0000") return 0;
        set<string> v;
        queue<string> q;
        q.push("0000");
        for (int d = 1; !q.empty(); d++) {
            for (int n = q.size(); n > 0; n--) {
                string cur = q.front(); q.pop();
                for (int i = 0; i < 4; i++) {
                    for (int dif = 1; dif <= 9; dif += 8) {
                        string s = cur;
                        s[i] = (s[i] - '0' + dif) % 10 + '0';
                        if (s == target) return d;
                        if (!dead.count(s) && !v.count(s)) q.push(s);
                        v.insert(s);
                    }
                }
            }
        }
        return -1;
    }
};
```

## Bomb Enemy
> 验证地址[Leetocde 361](https://leetcode.com/problems/bomb-enemy)

> 由于是收费的，所以看题可以在[题干](https://www.cnblogs.com/grandyang/p/5599289.html)

``` cpp
// 方法是先横着扫一遍，把线段中的敌人数标出来，然后再竖着扫，和横着的加起来，就是一颗炸弹能炸的人数
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

int helper(vector<string>& nums) {
    if (nums.empty() || nums[0].empty()) return 0;
    
    int m = nums.size(), n = nums[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));

    for (int i = 0; i < m; i++) {
        int ans = 0;
        for (int j = 0; j <= n; j++) {
            if (j == n || nums[i][j] == 'W') {
                for (int k = j - 1; k >= 0 && nums[i][k] != 'W'; k--) {
                    dp[i][k] = ans;
                }
                ans = 0;
            }
            else if (nums[i][j] == 'E') {
                ans ++;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        int ans = 0;
        for (int j = 0; j <= m; j++) {
            if (j == m || nums[j][i] == 'W') {
                for (int k = j - 1; k >= 0 && nums[k][i] != 'W'; k--) {
                    dp[k][i] += ans;
                }
                ans = 0;
            }
            else if (nums[j][i] == 'E') {
                ans ++;
            }
        }
    }
    int ret = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (nums[i][j] == 'E') ret = max(ret, dp[i][j] - 1);
            else ret = max(ret, dp[i][j]);
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
    cout << helper(nums) << endl;
    return 0;
}
/*
5 
0E00E
EW0E0
EE0E0
E00W0
0E0E0
*/
```

## Fence Repair
> 验证地址[POJ 3253](http://poj.org/problem?id=3253)
``` cpp
/*
这道题的答案可以分解为，每次切一次，被包含在其中的每一个片都要被加到结果中一次
也就是说 8 5 8三个数
第一次切参与了8 + 5 + 8， 被分成了13和8
第二次切13，其中8和5参与其中
所以结果可以写成板子的长度乘以参与个数的和
即 5 * 2 + 8 * 2 + 8 * 1

所以这满足哈夫曼树，哈夫曼树的特点是最优化二叉树，即权值乘以层数的加和最小

构建的思路可以参考 https://blog.csdn.net/dongfei2033/article/details/80657360

*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <stdio.h>
using namespace std;

int main() {
    int n;
    while (~scanf("%d", &n)) {
        priority_queue<int, vector<int>, greater<int> > q;
        for (int i = 0; i < n; i++) {
            int t;
            scanf("%d", &t);
            q.push(t);
        }
        long long sum = 0;
        if (q.size() == 1) {
            int a = q.top();
            sum += a;
            q.pop();
        }
        while (q.size() > 1) {
            int a = q.top();
            q.pop();
            int b = q.top();
            q.pop();
            sum += a + b;
            q.push(a + b);
        }
        printf("%lld\n", sum);
    }
    return 0;
}
```

## 背包
### 01背包
有N件物品和一个容量为V的背包。第i建物品的费用是c[i],价值是w[i]。求解将哪些物品装入背包可使价值总和最大(不过这里表示的是正好被填满V)
``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>
using namespace std;

int helper(vector<int>& w, vector<int>& c, int v) {
    if (w.empty()) return 0;
    int n = w.size();
    vector<int> dp(v + 1, INT_MIN);
    // INT_MIN是正好装满
    // 0是装不满也可以
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = v; j >= c[i]; j--) {
            dp[j] = max(dp[j], dp[j - c[i]] + w[i]);
        }
    }
    return dp[v];
}

int main() {
    int n, v;
    cin >> n >> v;
    vector<int> w(n, 0);
    vector<int> c(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> c[i] >> w[i];
    }
    cout << helper(w, c, v) << endl;
    return 0;
}
```

### 完全背包
有N种物品和一个容量为V的背包，每种物品都有无限件可用。第i种物品的费用是c[i]，价格是w[i].求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大

``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>

using namespace std;

int helper(vector<int>& w, vector<int>& c, int v) {
    if (w.empty() || !v) return 0;
    
    int n = w.size();
    vector<int> dp(v + 1, INT_MIN);
    // INT_MIN是正好装满
    // 0是装不满也可以
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = c[i]; j <= v; j++) {
            dp[j] = max(dp[j], dp[j - c[i]] + w[i]);
        }
    }
    return dp[v];
}

int main() {
    int n, v;
    cin >> n >> v;
    vector<int> w(n, 0);
    vector<int> c(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> c[i] >> w[i];
    }
    cout << helper(w, c, v) << endl;
    return 0;
}
```

### 多重背包
有N种物品和一个容量为V的背包。第i种物品最多有n[i]件，每件费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大

> 考虑成每个物品做k次01背包，但是有一点是每次的下限需要比上次提高一个c[i]

``` cpp
// 此方法和背包九讲实现不同，但是原理相同，用了一维dp
#include <iostream>
#include <vector>
#include <climits>
#include <math.h>

using namespace std;

int helper(vector<int>& w, vector<int>& c, vector<int>& k, int v) {
    if (w.empty() || !v) return 0;
    int n = w.size();
    vector<int> dp(v + 1, INT_MIN);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int x = 1; x <= k[i]; x ++) {
            for (int j = v; j >= x * c[i]; j--) {
                // 这里下限为x*c[i]为了保证一定能包含最少x个c[i]
                dp[j] = max(dp[j], dp[j - c[i]] + w[i]);
                if (j == v) 
                cout << dp[j] << ":" << i << " " << x << endl;
            }
        }
    }
    return dp[v];
    
}

int main() {
    int n, v;
    cin >> n >> v;
    
    vector<int> w(n, 0);
    vector<int> c(n, 0);
    vector<int> k(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> c[i] >> w[i] >> k[i]; 
    }
    
    cout << helper(w, c, k, v) << endl;
    return 0;
}
```

### 背包九讲其他题
参考[背包九讲](https://www.cnblogs.com/jbelial/articles/2116074.html)



## 迷宫穿梭
左上到右下，0为空，1为墙，找出任意一条路径
``` cpp
// 找出一条路径用dfs，比较好写
// 找出最短距离用bfs，如果存在一般用时间比较少
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>
using namespace std;

vector<pair<int, int>> dfs(vector<vector<int>>& nums, int x, int y, unordered_set<string>& hash) {
    string temp = to_string(x) + "-" + to_string(y);
    if (hash.find(temp) != hash.end()) return vector<pair<int, int>>();
    
    hash.insert(temp);
    int m = nums.size(), n = nums[0].size();
    if (x == m - 1 && y == n - 1) return vector<pair<int, int>>({{x, y}});
    int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
    for (int i = 0; i < 4; i ++) {
        int X = a[i] + x, Y = b[i] + y;
        if (X < m && X >= 0 && Y < n && Y >= 0 && !nums[X][Y]) {
            auto ret = dfs(nums, X, Y, hash);
            if (!ret.empty()) {
                ret.push_back({x, y});
                return ret;
            }
        }
    }
    return vector<pair<int, int>>();
} 

vector<pair<int, int>> helper(vector<vector<int>>& nums) {
    if (nums.empty() || nums[0].empty()) return vector<pair<int, int>>();
    
    int m = nums.size(), n = nums[0].size();
    unordered_set<string> hash;
    return dfs(nums, 0, 0, hash);
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
    for (auto i : helper(nums)) {
        cout << i.first << " " << i.second << endl;
    }
    return 0;
}
```

## Excel Sheet Column Number
> 验证地址[Leetcode 171](https://leetcode.com/problems/excel-sheet-column-number/)

``` cpp
class Solution {
public:
    int titleToNumber(string s) {
        int ret = 0;
        for (auto i : s) {
            int index = i - 'A' + 1;
            ret = ret * 26 + index;
        }
        return ret;
    }
};
```


## Reverse Linked List II
> 验证地址[Leetcode 92](https://leetcode.com/problems/reverse-linked-list-ii/)
``` cpp
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if (!head) return NULL;
        auto ret = new ListNode(-1);
        ret->next = head;
        auto cur = ret;
        for (int i = 0; i < m - 1; i++) {
            cur = cur->next;
        }
        if (!cur->next) return ret->next;
        auto tail = cur->next;
        head = cur;
        cur = cur->next;
        for (int i = m; i <= n; i++) {
            auto temp = cur->next;
            cur->next = head->next;
            head->next = cur;
            cur = temp;
        }
        tail->next = cur;
        return ret->next;
        
    }
};
```


## 去除重复元素
一个数组，比如[1,2,1,1,2,3,4],剔除重复的，然后还有个要求就是交换的操作要是原址的，就是比如前面的传入数组也要返回这个数组，[1,2,3,4,x,x,x,x]后面是什么不重要，前面一定要是和去重前一致。

``` cpp
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

int helper(vector<int>& nums) {
    int index = 0;
    unordered_set<int> m;
    for (auto i : nums) {
        if (m.find(i) != m.end()) continue;
        m.insert(i);
        nums[index++] = i;
    }
    return index;
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    n = helper(nums);
    for (int i = 0; i < n; i++) {
        cout << nums[i] << " ";
    }
    cout << endl;
    return 0;;
}
```

## 平方和
a^2+b^2=c，abc为正整数，给一个c问是否存在值ab使等式成立，O(c)解法
``` cpp
// dp思路
#include <iostream>
#include <unordered_map>

using namespace std;

bool helper(int n) {
    if (n <= 1) return false;
    unordered_map<int, int> m;
    
    for (int i = 1; i < n; i++) {
        int ans = i * i;
        if (ans >= n) continue;
        m[ans] = i;
        int delt = n - ans;
        if (m.find(delt) != m.end()) {
            cout << i << " " << m[delt] << endl;
            return true;
        } 
    }
    return false;
}

int main() {
    int n;
    cin >> n;
    cout << helper(n) << endl;
    return 0;
}
```

## Search in Rotated Sorted Array
> 验证地址[Leetcode 33](https://leetcode.com/problems/search-in-rotated-sorted-array/)

``` cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();
        if (!n) return -1;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid= left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            if (nums[mid] == nums[right]) right --;
            else if (nums[mid] > nums[right]) {
                if (nums[mid] > target && nums[left] <= target) right = mid - 1;
                else left = mid + 1;
            }
            else {
                if (nums[right] >= target && nums[mid] < target) left = mid + 1;
                else right = mid - 1; 
            }
        }
        return -1;
    }
};
```


## Binary Tree Right Side View
> 验证地址[Leetcode 199](https://leetcode.com/problems/binary-tree-right-side-view/)

``` cpp

// 利用先序遍历，且先遍历右节点
class Solution {
public:
    void helper(TreeNode* root, int level, vector<int>& ret) {
        if (!root) return;
        if (ret.size() < level) ret.push_back(root->val);
        helper(root->right, level + 1, ret);
        helper(root->left, level + 1, ret);
    }
    vector<int> rightSideView(TreeNode* root) {
        vector<int> ret;
        helper(root, 1, ret);
        return ret;
    }
};
```

## 最小编辑距离
``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>
using namespace std;
int helper(string a, string b) {
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

int main() {
    string a, b;
    cin >> a >> b;
    cout << helper(a, b) << endl;
    return 0;
}
```

## 24点
四个数，输出加减乘除括号组合起来等于24的所有表达式
``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
using namespace std;

unordered_map<int, vector<string>> dfs(vector<int> nums, int index) {
    int n = nums.size();
    unordered_map<int, vector<string>> ret;
    if (index == n - 1) {
        ret[nums[index]].push_back(to_string(nums[index]));
        return ret;
    }
    for (int i = index; i < n; i++) {
        if (i != index && nums[index] == nums[i]) continue;
        swap(nums[i], nums[index]);
        int a = nums[index];
        for (auto j : dfs(nums, index + 1)) {
            int b = j.first;
            for (auto k : j.second) {
                ret[a + b].push_back("(" + to_string(a) + "+" + k + ")");
                ret[a - b].push_back("(" + to_string(a) + "-" + k + ")");
                ret[a * b].push_back("(" + to_string(a) + "*" + k + ")");
                if (b) ret[a / b].push_back("(" + to_string(a) + "/" + k + ")");
            }
        }
    }
    return ret;
}

unordered_set<string> helper(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    unordered_set<string> ret;
    for (auto i : dfs(nums, 0)) {
        if (i.first == 24) {
            for (auto j : i.second) {
                ret.insert(j);
            }
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

## n以内素数个数
> 验证地址[Leetcode 204](https://leetcode.com/problems/count-primes/)

``` cpp
class Solution {
public:
    int countPrimes(int n) {
        vector<bool> dp(n + 1, true);
        int limit = sqrt(n);
        dp[0] = false;
        for (int i = 2; i <= limit; i++) {
            if (dp[i - 1]) {
                for (int j = 2; j * i <= n; j++) {
                    dp[j * i - 1] = false;
                }
            }
        }
        int ret = 0;
        for (int i = 0; i < n - 1; i++) {
            ret += dp[i];
        }
        return ret;
    }
};
```

## Gas Station
> 验证地址[Leetcode 134](https://leetcode.com/problems/gas-station/)

``` cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int ans = 0;
        int index = 0;
        int g = 0, c = 0;
        int n = gas.size();
        for (int i = 0; i < n; i++) {
            ans += gas[i] - cost[i];
            g += gas[i];
            c += cost[i];
            if (ans < 0) ans = 0, index = i + 1;
        }
        return g >= c ? index : -1;
    }
};
```

## 最长递增子序列 O(nlogn)版本(*)
> 参考[资料](https://www.cnblogs.com/frankchenfu/p/7107019.html)
``` cpp
#include<cstdio>
#include<algorithm>
const int MAXN=200001;

int a[MAXN];
int d[MAXN];

int main()
{
    int n;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
        scanf("%d",&a[i]);
    d[1]=a[1];
    int len=1;
    for(int i=2;i<=n;i++)
    {
        if(a[i]>d[len])
            d[++len]=a[i];
        else
        {
            int j=std::lower_bound(d+1,d+len+1,a[i])-d;
            d[j]=a[i]; 
        }
    }
    printf("%d\n",len);    
    return 0;
}
```

## Word Break II
> 验证地址[Leetcode 140](https://leetcode.com/problems/word-break-ii/)

``` cpp
class Solution {
public:
    vector<string> helper(string s, unordered_set<string>& m, unordered_map<string, vector<string>>& hash) {
        if (hash.find(s) != hash.end()) return hash[s];
        vector<string> ret;
        if (m.find(s) != m.end()) ret.push_back(s);
        int n = s.size();
        for (int i = 1; i < n; i++) {
            string pre = s.substr(0, i);
            if (m.find(pre) == m.end()) continue;
            for (auto j : helper(s.substr(i), m, hash)) {
                ret.push_back(pre + " " + j);
            }
        }
        hash[s] = ret;
        return ret;
    }
    
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> m;
        unordered_map<string, vector<string>> hash;
        for (auto i : wordDict) m.insert(i);
        return helper(s, m, hash);
    }
};
```

## 判断单链表是否有环，将环的入口返回
> 验证地址[牛客网](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=13&tqId=11208&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
``` cpp
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* head) {
        if (!head || !head->next) return NULL;
        
        auto fast = head, slow = head;
        while (fast->next && fast->next->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                fast = head;
                while (fast != slow) {
                    fast = fast->next;
                    slow = slow->next;
                }
                return fast;
            }
        }
        return NULL;
    }
};
```

## 复杂链表复制
> 验证地址[牛客网](https://www.nowcoder.com/practice/f836b2c43afc4b35ad6adc41ec941dba?tpId=13&tqId=11178&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

``` cpp

class Solution {
public:
    RandomListNode* Clone(RandomListNode* head) {
        if (!head) return head;
        RandomListNode* ret = new RandomListNode(-1);
        ret->next = head;
        auto cur = ret->next;
        while (cur) {
            auto temp = cur->next;
            cur->next = new RandomListNode(cur->label);
            cur->next->random = cur->random;
            cur->next->next = temp;
            cur = temp;
        }
        cur = ret->next;
        while (cur) {
            if (cur->random) cur->next->random = cur->random->next;
            cur = cur->next->next;
        }
        head = ret->next;
        cur = ret;
        while (head) {
            cur->next = head->next;
            head->next = head->next->next;
            head = cur->next->next;
            cur = cur->next;
        }
        return ret->next;
    }
};
```

## 概率类相关题目
### 水塘抽样
思路：总是选择第一个，然后以1/2的概率选择第二个，然后以1/3选择第三个...所以第i个被选择的概率是<code>[1/i]*[i/(i+1)]...[n-1/(n)]=1/n</code>，所以是等概率的。

``` cpp
// rand():[0, 正无穷]
int helper(vector<int>& nums) {
    if (nums.empty()) return -1;
    int ret = -1;
    for (int i = 0; i < n; i++) {
        int index = rand() % (i + 1);
        if (!index) ret = nums[i];
    }
    return ret;
}
```

### 从n个数里面等概率选出m个
原题目的场景大体是这样的：服务器每天会收到数以亿计的请求，但是目前服务器端不希望保存所有的请求，只想随机保存这些请求中的m个。试设计一种算法，能够使服务器实时保存m个请求，并使这些请求是从所有请求中的大致等概率被选中的结果。注意：不到一天的结束，是不能提前知道当天所有请求数n是多少的。

思路：先选取前m个数字，那么当n<=m的时候，就是100%会被选中，当选择第n个数的时候，选取的标准是，以m/n的概率选择这个数，然后随机替换已经保存的m个数中的其中一个。以下是证明。

``` 
当选择第m+1的时候
(1)第m+1个数字被选取的概率是m/(m+1)
(2)前m个数字被选择的情况是最后一个没被选择或该数字没有被替换，则概率为m/(m+1) * (m-1)/m + (1 – m/(m+1)) * 1 = m/(m+1)

前n个满足条件，数学归纳法，当选择第n+1个数的时候，当然该数被选取的概率为m/(1+n)，之前被选择的数，首先他在之前被选择的概率为m/n，他能被保留的情况为后一个没被选择或被选择了但是自己没有被替换，所以概率为[m/(N+1) * (m-1)/m + (1-m/(N+1))]* m/N = m/(N+1)
```

代码：
``` cpp
// rand(): [0, 1]
// nums.size() >> m
vector<int> helper(vector<int>& nums, int m) {
    vector<int> ret;
    int n = nums.size();
    for (int i = 0; i < m; i++) {
        ret.push_back(nums[i]);
    }
    for (int i = m; i < n; i++) {
        double percent = (double)m / (i + 1);
        if (rand() <= percent) {
            int index = m * rand();
            ret[index] = nums[i];
        }
    }
    return ret;
}
```

### 随机数发生器（*）
> 本节参考了[一道概率题](https://www.cnblogs.com/qiaozhoulin/p/5278171.html)

用50%拼p：p为20%，30%这种粒度，思路是构造0-9的随机数，50%产生0和1，那就产生4个数，共代表16个数，如果是10-15就舍去，重新生成，如果是小于p*10就生成0，否则生成1

用p拼50%：这种比较好处理，因为p*(1-p)和(1-p)*p的概率相同，所以用01和10代表50%，其余舍去

### 其余概率题
可以参考[资料1](https://blog.csdn.net/huazhongkejidaxuezpp/article/details/73662357)和[资料2](https://blog.csdn.net/kakulukia/article/details/49175811)


## 汉诺塔
``` cpp
#include <iostream>
#include <vector>

using namespace std;

void helper(vector<vector<int>>& nums, int start, int end, int blank, int cnt) {
    if (cnt == 1) {
        nums[end].push_back(nums[start].back());
        nums[start].pop_back();
        for (auto i : nums) {
            for (auto j : i) {
                cout << j << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    else {
        helper(nums, start, blank, end, cnt - 1);
        helper(nums, start, end, end, 1);
        helper(nums, blank, end, start, cnt - 1);
    }
}

int main() {
    int n;
    cin >> n;
    vector<vector<int>> nums(3);
    for (int i = n; i >= 1; i--) nums[0].push_back(i);
    helper(nums, 0, 2, 1, n);
    return 0;
}
```