---
title: 【总结】零星算法模板总结
date: 2018-09-08 13:01:13
tags: [算法, 总结, Leetcode, OJ]
---


> 这里整理一些零星的算法，作为记录...
<!-- more -->
# 水塘抽样
## Linked List Random Node
> [Leetcode 382](https://leetcode.com/problems/linked-list-random-node/description/)

Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.

Follow up:

What if the linked list is extremely large and its length is unknown to you? Could you solve this efficiently without using extra space?

Example:
```
// Init a singly linked list [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);

// getRandom() should return either 1, 2, or 3 randomly. Each element should have equal probability of returning.
solution.getRandom();
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* root;
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head) {
        root = head;
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        if (!root) return -1;
        int ret = root->val;
        ListNode* cur = root->next;
        int cnt = 2;
        while (cur) {
            int k = rand() % cnt;
            cnt ++;
            if (!k) ret = cur->val;
            cur = cur->next;
        }
        return ret;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(head);
 * int param_1 = obj.getRandom();
 */
```

## Random Pick Index
> [Leetcode 398](https://leetcode.com/problems/random-pick-index/description/)

Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Note:
The array size can be very large. Solution that uses too much extra space will not pass the judge.

Example:
```
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);

// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);

// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
```

``` cpp
class Solution {
public:
    vector<int> ans;
    Solution(vector<int> nums) {
        ans = nums;
    }
    
    int pick(int target) {
        int ret = -1, cnt = 1;
        for (int i = 0; i < ans.size(); i++) {
            if (ans[i] == target) {
                int key = rand() % cnt;
                if (!key) ret = i;
                cnt ++;
            }
        }
        return ret;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int param_1 = obj.pick(target);
 */
```

# 线段树
## Count of Smaller Numbers After Self
> [Leetcode 315](https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/)

You are given an integer array nums and you have to return a new counts array. The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

Example:
```
Input: [5,2,6,1]
Output: [2,1,1,0] 
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
```

``` cpp
// 此题先用二分查找树实现，线段树实现方法之后正在更新
struct MyTreeNode {
    int val;
    int duplicate;
    int cnt;
    MyTreeNode* left, *right;
    MyTreeNode(int val) : val(val), duplicate(1), cnt(0), left(NULL), right(NULL) {}
};
class Solution {
public:
    int helper(int target, MyTreeNode* &root) {
        if (!root) {
            root = new MyTreeNode(target);
            return 0;
        }
        if (root->val < target) {
            return root->duplicate + root->cnt + helper(target, root->right);
        }
        else if (root->val > target) {
            root->cnt ++;
            return helper(target, root->left);
        }
        else {
            root->duplicate ++;
            return root->cnt;
        }
    }
    vector<int> countSmaller(vector<int>& nums) {
        if (nums.empty()) return vector<int>();
        int n = nums.size();
        vector<int> ret(n, 0);
        MyTreeNode* root = new MyTreeNode(nums[n - 1]);
        for (int i = n - 2; i >= 0; i--) {
            ret[i] = helper(nums[i], root);
        }
        return ret;
    }
};
```

# 最大公约数
``` cpp
int gcd(int a, int b) {
    if (!b) return a;
    return gcd(b, a % b);
}
```

# 子串和满足条件的最短长度

## Minimum Size Subarray Sum
> [Leetcode 209](https://leetcode.com/problems/minimum-size-subarray-sum/description/)

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example: 
```
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

``` cpp
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int ret = INT_MAX;
        int n = nums.size(), left = 0, right = 0, ans = 0;
        while (right < n) {
            ans += nums[right++];
            while (left < right && ans - nums[left] >= s) ans -= nums[left++];
            if (ans >= s) ret = min(ret, right - left);
        }
        return ret == INT_MAX ? 0 : ret;
    }
};
```

## 上一题数组中包含正负，找和为target的最小长度

``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <climits>
using namespace std;

int helper(int target, vector<int>& nums) {
    int ret = INT_MAX;
    unordered_map<int, int> m;
    int ans = 0;
    for (int i = 0; i < nums.size(); i++) {
        ans += nums[i];
        int t = target - ans;
        m[ans] = i;
        if (m.count(t)) {
            ret = min(ret, i - m[t]);
        }
    }
    return ret == INT_MAX ? 0 : ret;
}

int main() {
    int target, n;
    cin >> target >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(target, nums) << endl;
    return 0;
}

```

## 上一题的follow up，包含正负，找出大于等于k的最小长度
> [Leetcode 862](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/)

Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K.

If there is no non-empty subarray with sum at least K, return -1.

 

Example 1:
```
Input: A = [1], K = 1
Output: 1
```
Example 2:
```
Input: A = [1,2], K = 4
Output: -1
```
Example 3:
```
Input: A = [2,-1,2], K = 3
Output: 3
```

``` cpp
#include <vector>
#include <iostream>
#include <queue>
#include <climits>

using namespace std;

int helper(vector<int>& nums, int k) {
    int ret = INT_MAX;
    deque<int> s;
    int n = nums.size();
    vector<int> ans(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        ans[i] = ans[i - 1] + nums[i - 1];
    }
    for (int i = 0; i <= n; i++) {
        while (!s.empty() && ans[i] - ans[s.front()] >= k) {
            ret = min(ret, i - s.front());
            s.pop_front();
        }
        while (!s.empty() && ans[s.back()] >= ans[i]) s.pop_back();
        s.push_back(i);
    }
    return ret == INT_MAX ? -1 : ret;
}


int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(nums, k) << endl;
    return 0;
}

```


# 快排相关

## 快排模板
``` cpp
#include <vector>
#include <iostream>

using namespace std;

int partition(vector<int>& nums, int left, int right) {
    int base = nums[left];
    while (left < right) {
        while (left < right && nums[right] >= base) right --;
        if (left < right) nums[left] = nums[right];
        while (left < right && nums[left] < base) left ++;
        if (left < right) nums[right] = nums[left];
    }
    nums[left] = base;
    return left;
}

void helper(vector<int>& nums, int left, int right) {
    if (left < right) {
        int mid = partition(nums, left, right);
        helper(nums, left, mid);
        helper(nums, mid + 1, right);
    }

}


int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    helper(nums, 0, n);
    for (auto i : nums) {
        cout << i << endl;
    }
    return 0;
}

```
## 快排优化

> 快排的优化可以参考[此处文献](https://www.cnblogs.com/c4isr/p/4603980.html)

- 基准随机化算法
``` cpp
#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace std;

int partition(vector<int>& nums, int left, int right) {
    int base = nums[left];
    while (left < right) {
        while (left < right && nums[right] >= base) right --;
        if (left < right) nums[left] = nums[right];
        while (left < right && nums[left] < base) left ++;
        if (left < right) nums[right] = nums[left];
    }
    nums[left] = base;
    return left;
}

void helper(vector<int>& nums, int left, int right) {
    if (left < right) {
    // 随机产生基准
        int index = left + rand() % (right - left + 1);
        swap(nums[left], nums[right]);
        int mid = partition(nums, left, right);
        helper(nums, left, mid - 1);
        helper(nums, mid + 1, right);
    }
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    helper(nums, 0, n - 1);
    for (int i : nums) {
        cout << i << endl;
    }
    return 0;
}

```

- 将前后中间三个数取中值作为标兵元素
- 在数组大小等于一定范围的时候，改为插入排序，防止排序退化
- 将相等的数字聚集到一起，然后跳过此处的排序


## 快速选择算法模板

> 验证地址[Leetcode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

``` cpp
#include <vector>
#include <iostream>

using namespace std;

int partition(vector<int>& nums, int left, int right) {
    int base = nums[left];
    while (left < right) {
        while (left < right && nums[right] <= base) right--;
        if (left < right) nums[left] = nums[right];
        while (left < right && nums[left] > base) left ++;
        if (left < right) nums[right] = nums[left];
    }
    nums[left] = base;
    return left;
}

int helper(vector<int>& nums, int left, int right, int k) {
    if (left <= right) {
        int mid = partition(nums, left, right);
        if (mid == k - 1) return nums[mid];
        if (mid < k - 1) return helper(nums, mid + 1, right, k);
        else return helper(nums, left, mid - 1, k);
    }
    return -1;
}

int findKthLargest(vector<int>& nums, int k) {
    int n = nums.size();
    if (n < k) return -1;
    return helper(nums, 0, n - 1, k);
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << findKthLargest(nums, k) << endl;
    return 0;
}
```

## 快速选择算法复杂度为O(n)的证明

[参考文献](https://blog.csdn.net/zzyu5ds/article/details/52818771)


# 最大子串和
> 贪心算法，最小类似

``` cpp
#include <vector>
#include <iostream>
#include <climits>
#include <stdlib.h>

using namespace std;

int helper(vector<int>& nums) {
    int ans = 0, n = nums.size();
    int ret = INT_MIN;
    for (int i = 0; i < n; i++) {
        ans += nums[i];
        ret = max(ret, ans);
        if (ans <= 0) ans = 0;
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

# 并查集模板
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int find(vector<int>& nums, int x) {
    int y = x;
    while (nums[x] != x) {
        x = nums[x];
    }
    while (y != x) {
        int t = nums[y];
        nums[y] = x;
        y = t;
    }
    return x;
}

void merge(vector<int>& nums, int x, int y) {
    int p1 = find(nums, x);
    int p2 = find(nums, y);
    if (p1 != p2) {
        nums[p1] = p2;
    }
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) nums[i] = i;
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >> y;
        merge(nums, x, y);
    }
    int ret = 0;
    for (int i = 0; i < n; i++) {
        if (i == find(nums, i)) ret ++;
    }
    cout << ret << endl;
    return 0;
}

```

# 牛顿法
> [Leetcode 69](https://leetcode.com/problems/sqrtx/description/)

Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:
```
Input: 4
Output: 2
```

Example 2:
```
Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.

```

``` cpp
class Solution {
public:
    int mySqrt(int x) {
        long long ret = x;
        while (ret * ret > x) {
            ret = (ret + x / ret) / 2;
        }
        return ret;
    }
};
```

# 堆排序

> [验证地址](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

``` cpp
#include <iostream>
#include <vector>

using namespace std;

void heapfy(vector<int>& nums, int index, int max) {
    int left = index * 2 + 1;
    int right = left + 1;
    int smallest = index;
    if (left < max && nums[left] < nums[smallest]) smallest = left;
    if (right < max && nums[right] < nums[smallest]) smallest = right;
    if (smallest != index) {
        swap(nums[index], nums[smallest]);
        heapfy(nums, smallest, max);
    }
}

int helper(vector<int>& nums, int k) {
    int n = nums.size();
    if (n < k) return -1;
    vector<int> ans;
    for (int i = 0; i < k; i++) {
        ans.push_back(nums[i]);
    }
    for (int i = k / 2 - 1; i >= 0; i--) heapfy(ans, i, k);
    for (int i = k; i < n; i++) {
        if (nums[i] < ans[0]) continue;
        ans[0] = nums[i];
        for (int j = k / 2 - 1; j >= 0; j--) heapfy(ans, j, k);
    }
    return ans[0];
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(nums, k) << endl;
    return 0;
}

```

# 拓扑排序

``` cpp
#include <iostream>
#include <utility>
#include <vector>
#include <unordered_set>

using namespace std;

vector<vector<int>> helper(vector<unordered_set<int>>& adj, vector<int>& cnt) {
    int n = cnt.size();
    unordered_set<int> m;
    vector<int> cur;
    vector<vector<int>> ret;
    for (int i = 0; i < n; i++) {
        if (!cnt[i]) {
            cur.push_back(i);
            m.insert(i);
        }
    }
    while (true) {
        ret.push_back(cur);
        vector<int> next;
        for (auto i : cur) {
            for (auto j : adj[i]) {
                cnt[j]--;
                if (!cnt[j]) {
                    if (m.find(j) != m.end()) return vector<vector<int>>();
                    next.push_back(j);
                    m.insert(j);
                }
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    if (n != m.size()) return vector<vector<int>>();
    return ret;
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> cnt(n, 0);
    vector<unordered_set<int>> adj(n);
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >> y;
        cnt[y]++;
        adj[x].insert(y);
    }
    auto ret = helper(adj, cnt);
    for (auto i : ret) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}

```
# 快速幂
## Pow(x, n)
> [Leetcode 50](https://leetcode.com/problems/powx-n/description/)

Implement pow(x, n), which calculates x raised to the power n (x<sup>n</sup>).

```
Input: 2.00000, 10
Output: 1024.00000
```
#### 代码
``` cpp
class Solution {
public:
    double myPow(double x, int n) {
        double ret = 1.0;
        // 注意要整型溢出
        long N = abs((long) n);
        while (N) {
            if (N & 1) ret *= x;
            N >>= 1;
            x *= x;
        }
        return n > 0 ? ret : 1 / ret;
    }
};
```

## 矩阵快速幂
> 矩阵乘法还可以做dp优化，之后更新....

``` cpp
#include <vector>
#include <iostream>

#define MOD 1000000007

using namespace std;

vector<vector<int>> mul(vector<vector<int>>& A, vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> ret(n, vector<int>(n, 0));
    for (int i = 0;i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                ret[i][j] += A[i][k] * B[k][j];
                ret[i][j] %= MOD;
            }
        }
    }
    return ret;
}

vector<vector<int>> Qpow(vector<vector<int>> nums, int k) {
    int n = nums.size();
    cout << n << endl;
    vector<vector<int>> ret(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        ret[i][i] = 1;
    }
    while (k) {
        if (k & 1) ret = mul(ret, nums);
        nums = mul(nums, nums);
        k >>= 1;
    }
    return ret;
}


int main() {
    int n, k;
    cin >> n >> k;
    vector<vector<int>> nums(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> nums[i][j];
        }
    }
    auto ret = Qpow(nums, k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << ret[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}

```
> [验证地址](http://acm.hdu.edu.cn/showproblem.php?pid=1575)


# 全排列相关
## Permutation Sequence
> [Leetcode 60](https://leetcode.com/problems/permutation-sequence/description/)

对于数字全排列可以用这种方法

```
string getPermutation(int n, int k) {
    long long ans[n + 1] = {1};
    string ret;
    vector<int> nums;
    for (int i = 1; i <= n; i++) {
        ans[i] = ans[i - 1] * i;
        nums.push_back(i);
    }
    for (int i = 0; i < n; i++) {
        int index = (k - 1) / ans[n - i - 1];
        ret += to_string(nums[index]);
        nums.erase(nums.begin() + index);
        k = k - ans[n - i - 1] * index;
    }
    return ret;
}
```

## 字母全排列
> 数字可以用上面的方法，但是对于字母来说就不行了，因为long long也存不下所有全排列的种类数

第二个思路是使用nextPermutation，此函数的原理是
1. 从最后向前找相邻的两个元素是的i < ii
2. 然后再从最后向前找一个元素使得i < j
3. i和j交换，然后将ii之后的元素reverse排序

> nextPermutation的头文件是algorithm

``` cpp
string getPermutation(int n, int k) {
    string ret;
    for (int i = 1; i <= n; i++) {
        ret += to_string(i);
    }
    for (int i = 1; i < k; i++) {
        next_permutation(ret.begin(), ret.end());
    }
    return ret;
}
```

# 面试题集合

## 面试题1
> 一个树形的图，要遍历其中k个节点，最少需要走多少步

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>

using namespace std;

int helper(vector<unordered_set<int>>& adj) {
    int n = adj.size();
    int ret = 0;
    vector<int> cur;
    for (int i = 0; i < n; i++) {
        if (adj[i].size() == 1) cur.push_back(i);
    }
    while (true) {
        ret ++;
        vector<int> next;
        for (auto i : cur) {
            for (auto j : adj[i]) {
                adj[j].erase(i);
                if (adj[j].size() == 1)
                    next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    if (cur.size() == 1) return 2 *(ret - 1);
    else return 2 * (ret - 1) + 1;
}

int main() {
    int n, k, d;
    cin >> n >> k >> d;
    vector<unordered_set<int>> adj(n, unordered_set<int>());
    for (int i = 0; i < k; i++) {
        int x, y;
        cin >> x >> y;
        adj[x].insert(y);
        adj[y].insert(x);
    }
    int len = helper(adj), ret;
    if (len >= d) ret = d;
    else ret = len + (d - len) * 2;
    cout << ret << endl;
    return 0;
}
```



## 面试题2

> 表达式求值

``` cpp
#include <iostream>
#include <math.h>

using namespace std;

double getNum(string s, int& index) {
    double ret = 0.0;
    int cnt = -1, n = s.size();
    while (index < n && (s[index] == '.' || isdigit(s[index]))) {
        if (s[index] == '.') cnt ++;
        else {
            ret = ret * 10 + (s[index] - '0');
            if (cnt >= 0) cnt ++;
        }
        index ++;
    }
    if (cnt == -1) cnt = 0;
    return ret / pow(10, cnt);
}

double helper(string s, int& index) {
    double ret = 0.0, cur_ret = 0.0;
    char op = '+';
    int n = s.size();
    while (index < n && s[index] != ')') {
        if (s[index] == ' ') {
            index ++;
            continue;
        }
        if (isdigit(s[index]) || s[index] == '(') {
            double ans;
            if (isdigit(s[index])) ans = getNum(s, index);
            else {
                index ++;
                ans = helper(s, index);
                index ++;
            }
            switch(op) {
                case '+' : cur_ret += ans; break;
                case '-' : cur_ret -= ans; break;
                case '*' : cur_ret *= ans; break;
                case '/' : cur_ret /= ans; break;
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

## 面试题3
> 寻找能使数组跷跷板平衡的支点有几个

``` cpp
#include <vector>
#include <iostream>

using namespace std;

int helper(vector<int>& nums) {
    int n = nums.size(), ret = 0;
    for (int i = 0; i <= n; i++) {
        int left = 0, right = 0;
        for (int j = 0; j < i; j++) {
            left += (i - j) * nums[j];
        }
        for (int j = i; j < n; j++) {
            right += (j - i + 1) * nums[j];
        }
        if (left == right) ret ++;
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

## 面试题4
> 给定一组数，表示他所在组的大小，输出一个数组，同一分组的在一起，数组保证字典序最小

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
    vector<int> ret;
    vector<vector<int>> ans;
    unordered_map<int, vector<int>> m;
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        m[nums[i]].push_back(i);
        if (m[nums[i]].size() == nums[i]) {
            ans.push_back(m[nums[i]]);
            m[nums[i]].clear();
        }
    }
    sort(ans.begin(), ans.end(), cmp);
    for (auto i : ans) {
        ret.insert(ret.end(), i.begin(), i.end());
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

## 面试题5
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
#include <vector>
#include <map>
#include <iostream>
#include <unordered_set>
using namespace std;

bool helper(map<char, map<char, string>>& trans, unordered_set<char>& goals, string cur, string next, int index) {
    int n = cur.size();
    if (index == n) {
        if (n == 1) return goals.find(cur[0]) != goals.end();
        return helper(trans, goals, next, "", 1);
    }
    for (auto i : trans[cur[index - 1]][cur[index]]) {
        if (helper(trans, goals, cur, next + i, index + 1)) return true;
    }
    return false;
}

int main() {
    int n, k;
    cin >> n >> k;
    map<char, map<char, string>> trans;
    unordered_set<char> goals;
    for (int i = 0; i < n * n; i++) {
        char x, y;
        string s;
        cin >> x >> y >> s;
        trans[x][y] = s;
    }
    for (int i = 0; i < k; i++) {
        char x;
        cin >> x;
        goals.insert(x);
    }
    string s;
    cin >> s;
    cout << helper(trans, goals, s, "", 1) << endl;
    return 0;
}

/*
3 3
A A BC
A B C
A C A
B A AC
B B C
B C C
C A A
C B A
C C AB
A B C
AAB
*/

```


