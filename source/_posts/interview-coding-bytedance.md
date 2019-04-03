---
title: 【面经】今日头条部分编程算法题
date: 2019-03-14 22:17:17
tags: [算法, 总结, 面经]
---

> 本文收集2018年实验室实习面试部分编程面经以供复习
<!-- more -->
## 有序链表归并
``` cpp
#include <iostream>


using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};

ListNode* helper(ListNode* A, ListNode* B) {
    ListNode* ret = new ListNode(-1);
    ListNode* cur = ret;
    while (A && B) {
        if (A->val < B->val) {
            cur->next = A;
            A = A->next;
        }
        else {
            cur->next = B;
            B = B->next;
        }
        cur = cur->next;
    }
    while (A) {
        cur->next = A;
        A = A->next;
        cur = cur->next;
    }
    while (B) {
        cur->next = B;
        B = B->next;
        cur = cur->next;
    }
    return ret->next;
}

int main() {
    int n;
    cin >> n;
    auto A = new ListNode(-1);
    auto cur = A;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cur->next = new ListNode(x);
        cur = cur->next;
    }
    cin >> n;
    auto B = new ListNode(-1);
    cur = B;
    for (int i = 0; i < n; i++) {
        int y;
        cin >> y;
        cur->next = new ListNode(y);
        cur = cur->next;
    }
    auto ret = helper(A->next, B->next);
    while (ret) {
        cout << ret->val << " ";
        ret = ret->next;
    }
    cout << endl;
    return 0;
}
```
## 反转链表
``` cpp
#include <iostream>

using namespace std;
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};


ListNode* helper(ListNode* root) {
    ListNode* ret = new ListNode(-1);
    while (root) {
        auto temp = root->next;
        root->next = ret->next;
        ret->next = root;
        root = temp;
    }
    return ret->next;
}

int main() {
    int n;
    cin >> n;
    ListNode* A = new ListNode(-1);
    ListNode* cur = A;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cur->next = new ListNode(x);
        cur = cur->next;
    }

    auto ret = helper(A->next);
    while (ret) {
        cout << ret->val << " ";
        ret = ret->next;
    }
    return 0;
}
```

## Partion and Reverse List
```
１->2->3->4->5->6->7
空间复杂度Ｏ(1)转换成：
1->7->2->6->3->5->4
```

``` cpp
#include <iostream>


using namespace std;
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};

ListNode* helper(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* fast = head, *slow = head;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    ListNode* head2 = slow->next;
    slow->next = NULL;
    ListNode* ret = new ListNode(-1);
    ListNode* cur = ret;
    while (head2) {
        auto temp = head2->next;
        head2->next = cur->next;
        cur->next = head2;
        cur = cur->next;
        head2 = temp;
    }
    head2 = ret->next;
    ret->next = NULL;

    cur = ret;
    while (head) {
        cur->next = head;
        head = head->next;
        cur = cur->next;
        if (head2) {
            cur->next = head2;
            head2 = head2->next;
            cur = cur->next;
        }
    }
    return ret->next;
}


int main() {
    int n;
    cin >> n;
    ListNode* head = new ListNode(-1);
    auto cur = head;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cur->next = new ListNode(x);
        cur = cur->next;
    }

    auto ret = helper(head->next);
    while (ret) {
        cout << ret->val << " ";
        ret = ret->next;
    }
    cout << endl;
    return 0;
}
```

## 单链表排序
``` cpp
// 快排版本
#include <iostream>

using namespace std;
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(NULL) {}
};

ListNode* helper(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* A = new ListNode(-1);
    ListNode* B = new ListNode(-1);
    ListNode* cur_A = A, *cur_B = B;
    auto base = head;
    head = head->next;
    while (head) {
        if (head->val < base->val) {
            cur_A->next = head;
            cur_A = head;
        }
        else {
            cur_B->next = head;
            cur_B = head;
        }
        head = head->next;
    }
    cur_A->next = NULL;
    cur_B->next = NULL;
    A->next = helper(A->next);
    B->next = helper(B->next);
    cur_A = A;
    while (cur_A->next) {
        cur_A = cur_A->next;
    }
    cur_A->next = base;
    base->next = B->next;
    return A->next;
}


int main() {
    int n;
    cin >> n;
    
    ListNode* head = new ListNode(-1);
    ListNode* cur = head;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cur->next = new ListNode(x);
        cur = cur->next;
    }
    
    auto ret = helper(head->next);
    while (ret) {
        cout << ret->val << " ";
        ret = ret->next;
    }
    cout << endl;
    return 0;
}
```

## 从一个数组里取m个数，能否和为n
``` cpp
// 类似于zeros and ones
#include <iostream>
#include <vector>

using namespace std;

bool helper(vector<int>& nums, int m, int s) {
    int n = nums.size();
    if (n < m) return false;
    vector<vector<bool>> dp(m + 1, vector<bool>(s + 1, false));
    dp[0][0] = true;

    for (auto num : nums) {
        for (int i = m; i >= 1; i --) {
            for (int j = s; j >= num; j--) {
                dp[i][j] = dp[i][j] || dp[i - 1][j - num];
            }
        }
    }
    return dp[m][s];
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    int m, s;
    cin >> m >> s;
    cout << helper(nums, m, s) << endl;
    return 0;
}
```

## 最大子串和
求出数组中最大的子串和，并求出子串
``` cpp
#include <iostream>
#include <vector>
#include <climits>

using namespace std;

vector<int> helper(vector<int>& nums) {
    int n = nums.size();
    int ans = 0;
    vector<int> temp;
    vector<int> cur;
    int ret = INT_MIN;
    for (auto i : nums) {
        ans += i;
        if (ans < 0) {
            ans = 0;
            cur = vector<int>({});
        }
        else {
            cur.push_back(i);
            if (ret < ans) {
                ret = ans;
                temp = cur;
            }
        }
    }
    return temp;
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

## 字符串拼接最大值
一堆数字如123，324，56怎么拼接得到的值最大。

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

bool cmp(const int& a, const int& b) {
    string x = to_string(a);
    string y = to_string(b);
    return x + y > y + x;
}

string helper(vector<int>& nums) {
    if (nums.empty()) return "";
    sort(nums.begin(), nums.end(), cmp);
    string ret;
    for (auto i : nums) {
        ret += to_string(i);
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

## 左边最大值
找出数组中每个数字左边部分（包括自己）最大的数字，然后返回结果数组

``` cpp
#include <iostream>
#include <vector>
#include <climits>
#include <math.h>
using namespace std;

vector<int> helper(vector<int>& nums) {
    int n = nums.size();
    vector<int> ret(n, -1);
    int limit = INT_MIN;
    for (int i = 0; i < n; i++) {
        limit = max(limit, nums[i]);
        ret[i] = limit;
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
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

## Search in Rotated Sorted Array
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int helper(vector<int>& nums, int target) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] == nums[right]) {
            right --;
            continue;
        }
        if (nums[mid] > nums[right]) {
            if (nums[mid] > target && nums[right] < target) {
                right = mid - 1;
            }
            else left = mid + 1;
        }
        else {
            if (nums[mid] < target && nums[right] > target) {
                left = mid + 1;
            }
            else right = mid - 1;
        }
    }
    return -1;
}

int main() {
    int n;
    cin >> n;
    
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    int target;
    cin >> target;
    cout << helper(nums, target)<< endl;
    return 0;
}

```

## 数组变化
一个数组，里面的元素全部初始为0，有以下两种操作：
- 指定一个元素+1          
- 所有的*2   

问到达一个数组目标值得最小操作步数。
``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>

using namespace std;

int helper(vector<int>& nums) {
    int ret = 0;
    int limit = INT_MAX;
    for (auto i : nums) {
        if (!i) continue;
        else {
            limit = min(limit, i);
            ret ++;
        }
    }
    int ans = 1;
    while (ans * 2 <= limit) ret ++, ans *= 2;
    for (auto i : nums) {
        if (i) ret += i - ans;
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

## 顺时针打印数组
``` cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

vector<int> helper(vector<vector<int>>& nums) {
    int m = nums.size(), n = nums[0].size();
    int dir = 0;
    int a[4] = {0, 1, 0, -1}, b[4] = {1, 0, -1, 0};
    
    int x = 0, y = 0;
    vector<int> ret;
    while (ret.size() < m * n) {
        if (nums[x][y] != INT_MAX) {
            ret.push_back(nums[x][y]);
            nums[x][y] = INT_MAX;
        }
        int X = x + a[dir], Y = y + b[dir];
        if (X < m && X >= 0 && Y >= 0 && Y < n && nums[X][Y] != INT_MAX) {  
            x = X, y = Y;
        }
        else dir = (dir + 1) % 4;
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
    for (auto i : helper(nums)) {
        cout << i << " ";
    } 
    cout << endl;
    return 0;
}
```

## K个升序数组归并
``` cpp
#include <iostream>
#include <vector>
#include <utility>
using namespace std;

void heapfy(vector<pair<int, int>>& ans, vector<vector<int>>& nums, int index, int max) {
    int left = index * 2 + 1;
    int right = left + 1;
    int smallest = index;
    if (left < max && nums[ans[left].first][ans[left].second] < nums[ans[smallest].first][ans[smallest].second]) {
        smallest = left;
    }
    if (right < max && nums[ans[right].first][ans[right].second] < nums[ans[smallest].first][ans[smallest].second]) {
        smallest = right;
    }
    if (smallest != index) {
        swap(ans[index], ans[smallest]);
        heapfy(ans, nums, smallest, max);
    }
}

vector<int> helper(vector<vector<int>>& nums) {
    int n = nums.size();
    vector<pair<int, int>> ans;
    for (int i = 0; i < n; i++) {
        if (!nums[i].empty()) ans.push_back({i, 0});
    }
    vector<int> ret;
    while (!ans.empty()) {
        ret.push_back(nums[ans[0].first][ans[0].second]);
        if (ans[0].second + 1 < nums[ans[0].first].size()) {
            ans[0].second ++;
            heapfy(ans, nums, 0, ans.size());
        }
        else {
            ans[0] = ans.back();
            ans.pop_back();
            heapfy(ans, nums, 0, ans.size());
        }
    }
    return ret;
}


int main() {
    int k;
    cin >> k;
    vector<vector<int>> nums;
    for (int i = 0; i < k; i++) {
        int n;
        cin >> n;
        vector<int> ans(n, 0);
        for (int j = 0; j < n; j++) {
            cin >> ans[j];
        }
        nums.push_back(ans);
    }
    for (auto i : helper(nums)) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```
## 二叉树的最近公共祖先
``` cpp
#include <iostream>
#include <vector>

using namespace std;
struct TreeNode {
    string val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(string s) : val(s), left(NULL), right(NULL) {}
};
TreeNode* build(vector<string>& nums, int& index) {
    int n = nums.size();
    if (index >= n) return NULL;
    auto val = nums[index++];
    if (val == "#") return NULL;
    auto ret = new TreeNode(val);
    ret->left = build(nums, index);
    ret->right = build(nums, index);
    return ret;
}

string helper1(TreeNode* root, string a, string b, bool& m) {
    if (!root) return "";
    if (root->val == a || root->val == b) m = true;
    bool ml = false, mr = false;
    string l = helper1(root->left, a, b, ml);
    if (ml) {
        if (m) return root->val;
        m = true;
        string r = helper1(root->right, a, b, mr);
        if (mr) return root->val;
        else return l; 
    }
    else {
        string r = helper1(root->right, a, b, mr);
        if (m && mr) return root->val;
        if (mr) {
            m = mr;
            return r;
        }
        else return "";
    }
}

string helper(vector<string>& nums, string a, string b) {
    int index = 0;
    auto root = build(nums, index);
    bool m = false;
    return helper1(root, a, b, m);
}


int main() {
    int n;
    cin >> n;
    vector<string> nums;
    for (auto i = 0; i < n; i++) {
        string s;
        cin >> s;
        nums.push_back(s);
    }
    string a, b;
    cin >> a >> b;
    cout << helper(nums, a, b) << endl;
    return 0;
}
```

## 两个升序数组，查合并之后的总的中位数

``` cpp
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;
typedef vector<int>::iterator Iter;

int helper1(Iter a, int m, Iter b, int n, int k) {
    if (!n) return a[k - 1];
    if (!m) return b[k - 1];
    if (k == 1) return min(a[0], b[0]);

    int l = min(k / 2, m);
    int r = min(k / 2, n);
    if (a[l - 1] < b[r - 1]) {
        return helper1(a + l, m - l, b, n, k - l);
    }
    else {
        return helper1(a, m, b + r, n - r, k - r);
    }
}

double helper(vector<int>& A, vector<int>& B) {
    int n1 = A.size();
    int n2 = B.size();
    return (
        helper1(A.begin(), n1, B.begin(), n2, (n1 + n2 + 1) / 2) +
        helper1(A.begin(), n1, B.begin(), n2, (n1 + n2 + 2) / 2)
        ) / 2.0;
}

int main() {
    int n;
    cin >> n;
    vector<int> A(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }
    vector<int> B(n, 0);
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> B[i];
    }
    cout << helper(A, B) << endl;
    return 0;
}
```

## LRU
``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include <vector>
#include <utility>

using namespace std;

class LRU {
private:
    unordered_map<int, pair<int, list<int>::iterator>> nums;
    list<int> d;
    int cap;
    void touch(int key) {
        if (nums.find(key) == nums.end()) return;
        d.erase(nums[key].second);
        nums.erase(key);
    }
public:
    LRU(int capacity): cap(capacity) {};
    void add(int key, int val) {
        touch(key);
        d.push_front(key);
        nums[key] = {val, d.begin()};
    }
    int get(int key) {
        if (nums.find(key) == nums.end()) return -1;
        int ret = nums[key].first;
        touch(key);
        d.push_front(key);
        nums[key] = {ret, d.begin()};
        return ret;
    }
};
int main() {
    auto cache = new LRU(2);
    cache->add(1, 1);
    cache->add(2, 2);
    cout << cache->get(1) << endl;       // returns 1
    
    return 0;
}
```

## 带重复的字符串全排列
``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void dfs(vector<vector<int>>& ret, vector<int> nums, int index) {
    int n = nums.size();
    if (n == index) ret.push_back(nums);
    for (int i = index; i < n; i++) {
        if (i != index && nums[i] == nums[index]) continue;
        swap(nums[index], nums[i]);
        dfs(ret, nums, index + 1);
    }
}

vector<vector<int>> helper(vector<int>& nums) {
    vector<vector<int>> ret;
    sort(nums.begin(), nums.end());
    dfs(ret, nums, 0);
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
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}
```

## 自然数排列
0123456791011121314.. 自然数这样顺次排下去，给一个index，找出对应的数字是什么
``` cpp
#include <iostream>
#include <math.h>
using namespace std;
char helper(int n) {
    if (!n) return '0';
    if (n < 10) return '1' + (n - 1);
    
    n -= 9;
    int cnt = 1;
    while (n > (pow(10, cnt + 1) - pow(10, cnt)) * (cnt + 1)) {
        n -= (pow(10, cnt + 1) - pow(10, cnt)) * (cnt + 1);
        cnt ++;
    }
    int ans = pow(10, cnt) + (n - 1) / (cnt + 1);
    return to_string(ans)[(n - 1) % (cnt + 1)];
}
int main() {
    for (int i = 0; i < 200; i++) {
        cout << helper(i);
    }
    cout << endl;
    return 0;
}
```

## House Robber III
[Leetcode 337](https://leetcode.com/problems/house-robber-iii/)
``` cpp
class Solution {
public:
    int helper(TreeNode* root, int& l, int& r) {
        if (!root) return 0;
        int ll = 0, lr = 0, rl = 0, rr = 0;
        l = max(0, helper(root->left, ll, lr));
        r = max(0, helper(root->right, rl, rr));
        return max(l + r, root->val + ll + lr + rl + rr);
    }
    int rob(TreeNode* root) {
        int l, r;
        return helper(root, l, r);
    }
};
```

## Minimum Window Substring
[Leetcode 76](https://leetcode.com/problems/minimum-window-substring/)
``` cpp
class Solution {
public:
    string minWindow(string s, string t) {
        if (t.empty()) return "";
        unordered_map<char, int> m;
        for (auto i : t) {
            m[i] ++;
        }
        int ans = m.size();
        int n = s.size();
        int l = 0, r = 0;
        int len = INT_MAX;
        string ret;
        while (r <= n && l <= r) {
            if (ans > 0 && r < n) {
                auto key = s[r++];
                if (m.find(key) != m.end()) {
                    m[key] --;
                    if (m[key] == 0) ans--;
                }
                
            }
            else {
                auto key = s[l++];
                if (m.find(key) != m.end()) {
                    m[key] ++;
                    if (m[key] == 1) ans ++;
                }
                
            }
            if (!ans && r - l < len) {
                len = r - l;
                ret = s.substr(l, len);
            }
        }
        return ret;
    }
};
```


## 变色龙
``` 
Description
    在一个美丽的小岛上住着一群变色龙：其中有X只变色龙是红色的，Y只变色龙是绿色的，Z只变色龙是蓝色的。
    每个时刻会有两只不同颜色的变色龙相遇，相遇后他们会同时变成第三种颜色。比如，如果一只红色的变色龙和一只蓝色的变色龙相遇了，他们就会同时变成绿色的变色龙，如果一只绿色的变色龙和一只蓝色的变色龙相遇了，他们就会同时变成红色的变色龙，等等。
    那么最后是否有可能所有的变色龙都是同一种颜色呢？
 
Input
    输入的第一行包含一个整数T (1 <= T <= 100)，表示接下来一共有T组测试数据。
    每组数据占一行，包含三个整数X, Y, Z (1 <= X, Y, Z <= 109)，含义同上。
 
Output
    对于每组测试数据，如果最后有可能所有的变色龙都是同一种颜色，用一行输出“Yes”（不包括引号），否则输出“No”（不包括引号）。
 
Sample Input
4
1 1 1
1 2 3
7 1 2
3 7 5
Sample Output
Yes
No
Yes
No
HINT
```

``` cpp
#include <iostream>
#include <math.h>
using namespace std;

string helper(int x, int y, int z) {
    if (x == y || y == z) return "YES";
    if (abs(x - y) % 3 == 0 || abs(x - z) % 3 == 0 || abs(y - z) % 3 == 0) return "YES";
    else return "NO";
}

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        int x, y, z;
        cin >> x >> y >> z;
        cout << helper(x, y, z) << endl;
    }
    return 0;
}
```

## 平方根
``` cpp
#include <iostream>
#include <math.h>
#include <climits>
using namespace std;

int helper1(int n) {
    int ret = n;
    while (ret * ret > n) {
        ret = (ret + n / ret) / 2;
    }
    return ret;
}
int helper2(int n) {
    int left = 1, right = n;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (mid + 1 > n / (mid + 1) && mid <= n / mid) return mid;
        if (mid < n / mid) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    int n;
    cin >> n;
    cout << helper1(n) << endl;
    cout << helper2(n) << endl;
    return 0;
}
```

## 短网址系统
设计一个短网址系统？短网址生成策略？短网址和长网址的映射关系如何表示？存网址的数据库表太大了怎么办？Sharding后如何分别以长网址或短网址为主key搜索？你觉得这个系统追求的是时间效率还是空间节省？那冗余存储的牺牲值不值得？

```cpp
// 代码待补充
```
## 大文件判断重复判断
两个大文件，4g内存，判断两个文件里想同的url
``` cpp
/*
三种思路供参考：
1. 运用bitmap，类似于布隆过滤器
2. 运用hash值，类似于布隆过滤器，有可能出现判断不准确的情况
3. 分桶成小文件，然后查询过程中扫每个文件
*/
```

## 无向图的最小环
```
输入：
n个点，t条边
输出：
最短环的路径
```

```cpp
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_set>
#include <algorithm>
using namespace std;


vector<int> helper(vector<pair<int, int>>& nums, int n) {
    vector<unordered_set<int>> adj(n);
    for (auto i : nums) {
        if (i.first == i.second) continue;
        adj[i.first].insert(i.second);
        adj[i.second].insert(i.first);
    }

    vector<vector<int>> cur;
    cur.push_back({0});

    while (true) {
        vector<vector<int>> next;
        for (auto i : cur) {
            for (auto j : adj[i.back()]) {
                if (i.size() > 2 && j == 0) {
                    i.push_back(0);
                    return i;
                }
                if (find(i.begin(), i.end(), j) == i.end()) {
                    auto temp = i;
                    temp.push_back(j);
                    next.push_back(temp);
                }
                
            }
        }
        if (next.empty()) break;
        cur = next;
    }
    return vector<int>();
}

int main() {
    int n;
    cin >> n;
    int t;
    cin >> t;
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
/*
6 7
0 1
1 2
1 3
2 3
2 4
4 5
5 0
*/
```