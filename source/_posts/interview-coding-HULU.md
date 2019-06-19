---
title: 【面经】HULU部分编程算法题
tags:
  - 算法
  - 总结
  - 面经
date: 2019-04-21 22:37:33
---


> 本文收集网络上面试部分编程面经以供复习

<!-- more -->


## Alien Language(待付费)
> 验证地址[Leetcode 269](https://leetcode.com/problems/alien-dictionary)
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

## Days of Our Lives
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/days-of-our-lives/0)
``` cpp
#include <iostream>
#include <vector>

using namespace std;

vector<long long> helper(long long n, long long k) {
    long long ans = n / 7;
    vector<long long> ret(7, ans);
    int c = n % 7;
    for (int i = 0; i < c; i++) {
        ret[(k - 1 + i) % 7] ++;
    }
    return ret;
}


int main() {
	//code
	int t;
	cin >> t;
	for (int i = 0; i < t; i++) {
	    long long n, k;
	    cin >> n >> k;
	    auto ans = helper(n, k);
	    for (int j = 0; j < 6; j++) {
	        cout << ans[j] << " ";
	    }
	    cout << ans[6] << endl;
	}
	return 0;
}
```

## Connect Nodes at Same Level
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1)
``` cpp
void connect(Node *p)
{
   // Your Code Here
   if (!p) return;
   Node* pre = p;
   pre->nextRight = NULL;
   while (true) {
       while (pre && !pre->left && !pre->right) pre = pre->nextRight;
       if (!pre) return;
       auto temp = pre->left ? pre->left : pre->right;
       auto cur = temp;
       while (pre) {
           if (pre->left) cur->nextRight = pre->left, cur = cur->nextRight;
           if (pre->right) cur->nextRight = pre->right, cur = cur->nextRight;
           pre = pre->nextRight;
       }
       cur->nextRight = NULL;
       pre = temp;
   }
}
```

## Maximum difference between node and its ancestor
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/maximum-difference-between-node-and-its-ancestor/1)

``` cpp
void helper(TreeNode* root, int local_min, int local_max, int& ret) {
    if (!root) return;
    if (local_min != INT_MAX) ret = max(ret, root->val - local_min);
    if (local_max != INT_MAX) ret = max(ret, local_max - root->val);
    local_min = min(local_min, root->val);
    local_max = max(local_max, root->val);
    helper(root, local_min, local_max, ret);
}

int maxDiff(TreeNode* root) {
    // Your code here 
    if (!root) return INT_MIN;
    int ret = INT_MIN;
    helper(root, INT_MAX, INT_MIN, ret);
    return ret;
}
```

## Vertical Order Traversal of a Binary Tree
> 验证地址[Leetcode 987](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool static cmp(const pair<int, int>& a, const pair<int, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
    }
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<vector<int>> ret;
        if (!root) return ret;
        vector<pair<TreeNode*, pair<int, int>>> cur;
        cur.push_back({root, {0, 0}});
        map<int, vector<pair<int, int>>> m;
        while (true) {
            vector<pair<TreeNode*, pair<int, int>>> next;
            for (auto i : cur) {
                m[i.second.first].push_back({i.second.second, i.first->val});
                if (i.first->left) 
                    next.push_back({i.first->left, {i.second.first - 1, i.second.second + 1}});
                if (i.first->right) 
                    next.push_back({i.first->right, {i.second.first + 1, i.second.second + 1}});
            }
            if (next.empty()) break;
            cur = next;
        }
        for (auto i : m) {
            sort(i.second.begin(), i.second.end(), cmp);
            vector<int> temp;
            for (auto j : i.second) {
                temp.push_back(j.second);
            }
            ret.push_back(temp);
        }
        return ret;
    }
};
```

## Top View of Binary Tree
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/top-view-of-binary-tree/1)

``` cpp
void topView(struct Node *root)
{
    // Your code here
    unordered_map<int, int> m;
    if (!root) return;
    vector<pair<Node*, int>> cur;
    cur.push_back({root, 0});
    while (true) {
        vector<pair<Node*, int>> next;
        for (auto i : cur) {
            if (m.find(i.second) == m.end()) {
                cout << i.first->data << " ";
                m[i.second] = i.first->data;
            }
                
            if (i.first->left) {
                next.push_back({i.first->left, i.second - 1});
            }
            if (i.first->right) {
                next.push_back({i.first->right, i.second + 1});
            }
        }
        if (next.empty()) break;
        cur = next;
    }
    
}
```

## Smallest window in a string containing all the characters of another string
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/smallest-window-in-a-string-containing-all-the-characters-of-another-string/0)

``` cpp
#include <iostream>
#include <unordered_map>
#include <climits>
using namespace std;

string helper(string s, string p) {
    unordered_map<char, int> m;
    for (auto i : p) m[i] ++;
    int ans = m.size();
    int n= s.size();
    int i = 0, j = 0;
    int len = INT_MAX;
    string ret = "-1";
    while (i <= j && j <= n) {
        if (j < n && ans > 0) {
            if (m.find(s[j]) != m.end()) {
                m[s[j]] --;
                if (!m[s[j]]) ans --;
            }
            j++;
            
        }
        else {
            if (m.find(s[i]) != m.end()) {
                if (!m[s[i]]) ans ++;
                m[s[i]] ++;
            }
            i++;
        }
        if (!ans) {
            if (j - i < len) {
                
                len = j - i;
                ret = s.substr(i, len);
            }
        }
    }
    return ret;
}

int main() {
	//code
	int t;
	cin >> t;
	for (auto i = 0; i < t; i++) {
	    string s, p;
	    cin >> s >> p;
	    cout << helper(s, p) << endl;
	}
	return 0;
}
```

## Simplify the directory path (Unix like)
> 题目参考自[geeksforgeeks](https://www.geeksforgeeks.org/simplify-directory-path-unix-like/)
```
"/a/./"   --> means stay at the current directory 'a'
"/a/b/.." --> means jump to the parent directory
              from 'b' to 'a'
"////"    --> consecutive multiple '/' are a  valid  
              path, they are equivalent to single "/".

Input : /home/
Output : /home

Input : /a/./b/../../c/
Output : /c

Input : /a/..
Output : /

Input : /a/../
Ouput : /

Input : /../../../../../a
Ouput : /a

Input : /a/./b/./c/./d/
Ouput : /a/b/c/d

Input : /a/../.././../../.
Ouput : /

Input : /a//b//c//////d
Ouput : /a/b/c/d
```


``` cpp
// 栈版本
#include <iostream>
#include <vector>
#include <stack>

using namespace std;

string helper(string s) {
    if (s.empty()) return "/";
    if (s.back() != '/') s.push_back('/');

    stack<string> st;
    string ans;
    for (auto i : s) {
        if (i == '/') {
            if (ans == "" || ans == ".") {
                ans = "";
            }
            else if (ans == "..") {
                if (!st.empty()) st.pop();
                ans = "";
            }
            else {
                st.push(ans);
                ans = "";
            }
        }
        else {
            ans.push_back(i);
        }
    }
    string ret;
    while (!st.empty()) {
        ret = "/" + st.top() + ret;
        st.pop();
    }
    return ret.empty() ? "/" : ret;
}

int main() {
    while (true) {
        string s;
        cin >> s;
        cout << helper(s) << endl;
    }
    return 0;
}
```

``` cpp
// 原地压缩版
#include <iostream>


using namespace std;

int helper(string& s) {
    int n = s.size();
    int index = 0;
    string ans;
    for (int i = 0; i <= n; i++) {
        if (i == n || s[i] == '/') {
            if (ans == ".") {
                while (index > 0 && s[index - 1] != '/') index--;
                index --;
            }
            if (ans == "..") {
                while (index > 0 && s[index - 1] != '/') index--;
                if (index) index --;
                while (index > 0 && s[index - 1] != '/') index--;
                if (index) index --;
            }
            
            ans = "";
        }
        else {
            s[index++] = '/';
            while (i < n && s[i] != '/') {
                ans.push_back(s[i]);
                s[index ++] = s[i++];
            }
            i --;
        }
    }
    if (!index) {
        s = "/";
        return 1;
    }
    return index;
}

int main() {
    while (true) {
        string s;
        cin >> s;
        int len = helper(s);
        cout << s.substr(0, len) << endl;
    }
    return 0;
}
```

## 全排列
### 打印数组的全排列

``` cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;
void dfs(string s, int index, vector<string>& ret) {
    int n = s.size();
    if (index == n - 1) {
        ret.push_back(s);
        return;
    }

    for (int i = index; i < n; i++) {
        if (index != i && s[i] == s[index]) continue;
        swap(s[index], s[i]);
        dfs(s, index + 1, ret);
    }

}

vector<string> helper(string s) {
    vector<string> ret;
    if (s.empty()) return ret;
    sort(s.begin(), s.end());
    dfs(s, 0, ret);
    return ret;
}

int main() {
    string s;
    cin >> s;
    for (auto i : helper(s)) {
        cout << i << endl;
    }
    return 0;
}
```


### 打印n个字符串的全排列第K个值（全排列个数不会溢出情况）
``` cpp
#include <iostream>
#include <vector>

using namespace std;
string helper(int n, int k) {
    vector<long long> fac(n, 1);
    vector<int> nums(n, 0);
    for (int i = 1; i < n; i++) {
        fac[i] = fac[i - 1] * i;
    }
    for (int i = 0; i < n; i++) {
        nums[i] = i + 1;
    }
    string ret;
    for (int i = 0; i < n; i++) {
        int index = (k - 1) / fac[n - i - 1];
        ret += to_string(nums[index]);
        nums.erase(nums.begin() + index);
        k -= index * fac[n - i - 1];
    }
    return ret;
}

int main() {
    int n, k;
    cin >> n >> k;
    cout << helper(n, k) << endl;
    return 0;
}
```

### 打印n个字符串的全排列第K个值（全排列个数会溢出情况）
> [Leetcode 31](https://leetcode.com/problems/next-permutation/)
> 参考全排列的算法，也可以参考[next permutation](https://www.cnblogs.com/grandyang/p/4428207.html)

``` cpp
// 此题的做题思路是全排列
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void next_permutation(vector<int>& nums) {
    int n = nums.size();
    int index = n - 2;
    // 全排列中的全是升序，那么恢复的时候一定index之后是降序
    while (index >= 0 && nums[index] >= nums[index + 1]) index --;
    if (index >= 0) {
        // 找到升序中的第一个小于该值的第一个值，也就是swap的位置
        int i = n - 1;
        while (i > index && nums[i] <= nums[index]) i --;
        swap(nums[i], nums[index]);
    }
    // 如果已经逆序了，就变成了顺序
    reverse(nums.begin() + index + 1, nums.end());
}

void helper(vector<int>& nums, int k) {
    int n = nums.size();
    
    for (int i = 0; i < k; i++) {
        for (auto j : nums) cout << j << " ";
        cout << endl;
        next_permutation(nums);
    }
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n);
    for (int i = 0; i < n; i++) {
        nums[i] = i + 1;
    }
    helper(nums, k);
    return 0;
}
```




## ransom note
> 验证地址[Leetcode 383](https://leetcode.com/problems/ransom-note/)

``` cpp
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> m;
        for (auto i : magazine) m[i]++;
        for (auto i : ransomNote) {
            if (-- m[i] < 0) return false;
        }
        return true;
    }
};
```

## logger rate limiter(待付费)
> 验证地址[Leetcode 359](https://leetcode.com/problems/logger-rate-limiter/)

``` cpp
class Logger {
private:
    unordered_map<string, int> m;
public:
    Logger() {}
    bool shouldPrintMessage(int timestamp, string message) {
        bool ret = true;
        if (m.find(message) != m.end() && m[message] >= timestamp - 10) {
            ret = false;
        } 
        m[message] = timestamp;
        return ret;
    }
};
```

## Snakes and Ladders
> 验证地址[Leetcode 909](https://leetcode.com/problems/snakes-and-ladders/)
``` cpp
class Solution {
public:
    int snakesAndLadders(vector<vector<int>>& board) {
        unordered_set<int> m;
        int ret = 0, n = board.size();
        unordered_set<int> cur({1});
        while (true) {
            unordered_set<int> next;
            for (auto i : cur) {
                m.insert(i);
                if (i == n * n) return ret;
                for (int j = 1; j <= 6; j++) {
                    int index = j + i;
                    if (index > n * n) continue;
                    int a = (index - 1) / n, b = (index - 1) % n;
                    int temp = board[n - a - 1][a % 2 ? n - 1 - b : b];
                    if (temp > 0) index = temp;
                    if (m.find(index) == m.end()) next.insert(index);
                }
            }
            if (next.empty()) break;
            cur = next;
            ret ++;
        }
        return -1;
    }
};
```


## Spirally traversing a matrix
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/spirally-traversing-a-matrix/0)
``` cpp
#include <iostream>
#include <vector>

using namespace std;
void helper(vector<vector<int>>& nums) {
    if (nums.empty() || nums[0].empty()) return;
    int m = nums.size(), n = nums[0].size();
    int a[4] = {0, 1, 0, -1}, b[4] = {1, 0, -1, 0};
    int i = 0, j = 0;
    int cnt = 0;
    int dir = 0;
    while (cnt < m * n) {
        cout << nums[i][j] << " ";
        nums[i][j] = -1;
        cnt ++;
        i += a[dir];
        j += b[dir];
        if (i < m && i >= 0 && j < n && j >= 0 && nums[i][j] >= 0);
        else {
            i -= a[dir];
            j -= b[dir];
            dir = (dir + 1) % 4;
            i += a[dir];
            j += b[dir];
        }
    }
    cout << endl;
}

int main() {
	int t;
	cin >> t;
	for (int i = 0; i < t; i++) {
	    int m, n;
	    cin >> m >> n;
	    vector<vector<int>> nums(m, vector<int>(n, 0));
	    for (int i = 0; i < m; i++) {
	        for (int j = 0; j < n; j++) {
	            cin >> nums[i][j];
	        }
	    }
	    helper(nums);
	}
	return 0;
}
```

## Encode and Decode TinyURL
> 验证地址[Leetcode 535](https://leetcode.com/problems/encode-and-decode-tinyurl/)

``` cpp
// 思路：
// 0-9a-zA-Z一共是62个字母，所以全局有一个cnt表示已经有多少个网址，
// 然后将这个cnt变成62进制数，每一位对应到62个字母中的其中一个，就是编码过程，然后存下来
// 因为每一个网址都是唯一的自增id，所以肯定能保证在O(1)时间里面生成唯一短网址
// 解码过程就是将提取下来的数
class Solution {
class Solution {
public:
    unordered_map<char, int> m;
    unordered_map<long long, string> ans;
    long long id;
    Solution() {
        id = 0;
        int index = 0;
        for (char i = 'a'; i <= 'z'; i++) m[i] = index ++;
        for (char i = 'A'; i <= 'Z'; i++) m[i] = index ++;
        for (char i = '0'; i <= '9'; i++) m[i] = index ++;
    }
    
    // Encodes a URL to a shortened URL.
    string encode(string longUrl) {
        string ret;
        ans[id] = longUrl;
        long long cnt = id;
        while (cnt) {
            ret.push_back(m[cnt % 62]);
            cnt /= 62;
        }
        id ++;
        return ret;
    }

    // Decodes a shortened URL to its original URL.
    string decode(string shortUrl) {
        long long ret = 0;
        for (auto i : shortUrl) {
            ret = ret * 62 + m[i];
        }
        return ans[ret];
    }
};

// Your Solution object will be instantiated and called as such:
// Solution solution;
// solution.decode(solution.encode(url));
```

## XML转化成Tree/JSON
``` cpp
#include <iostream>
#include <vector>


using namespace std;

class Node {
public:
    string key;
    string val;
    vector<Node*> child;
    Node(string x): key(x), val("") {}
};

Node* convert(string s, int& index) {
    string key;
    int n = s.size();
    // if (index >= n || s[index] != '<') throw "Format Error";
    while (index < n && s[index] != '<') index ++;
    index ++;
    while (index < n && s[index] != '>') key.push_back(s[index ++]);
    index ++;
    Node* ret = new Node(key);
    cout << key << endl;
    string val;
    while (index < n - 1 && !(s[index] == '<' && s[index + 1] == '/')) {
        while (index < n && s[index] == ' ') index ++;
        if (s[index] == '<') {
            index --;
            ret->child.push_back(convert(s, index));
        }
        else {
            val.push_back(s[index ++]);
        }
    }
    cout << val << endl;
    while (index < n && s[index] != '>') index ++;
    index ++; 
    if (ret->child.empty()) ret->val = val;
    return ret;

}

string dfs(Node* root) {
    if (!root) return "";
    string ret = "{";
    ret += " " + root->key + ": ";
    if (root->child.empty()) ret += "\"" + root->val + "\"";
    else {
        for (auto i : root->child) {
            ret += dfs(i) + ", ";
        }
        ret.pop_back();
        ret.pop_back();
    }
    ret += " }";
    return ret;
}

string helper(string s) {
    int n = s.size();
    int index = 0;
    while (index < n && s[index] == ' ') index++;
    if (index == n) return "";
    Node* root = convert(s, index);
    return dfs(root);
}

int main() {
    string s = "<note><plus> <to>George</to></plus> <from>John</from><heading>Reminder</heading><body>Don't forget the meeting!</body></note>";
    cout << helper(s) << endl;;
    return 0;
}
```

## bitcoin trading
```
Question:
You know the daily prices of Bitcoin (BTC), and you traveled back to 1 year ago. 
The BTC exchange was highly regulated and you can only do one of (a) BUY, (b) HOLD, and (c) SELL, per day. 
You can trade one BTC per day.
You can not sell BTC that you don't own - no shorts or derivatives.
You have unlimited cash
Maximize your profit for this year
```

``` cpp
#include <iostream>
#include <vector>
#include <stack>
#include <utility>

using namespace std;

int helper(vector<int>& nums) {
    stack<pair<int, int>> s;
    for (auto i : nums) {
        if (s.empty()) s.push({i, i});
        else if (s.top().second > i) s.push({i, i});
        else {
            int x = s.top().first;
            while (!s.empty() && s.top().second >= x) s.pop();
            s.push({x, i});
        }
    }

    int ret = 0;
    while (!s.empty()) {
        ret += s.top().second - s.top().first;
        s.pop();
    }
    return ret;
}

int main()  {
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

## letter association(reduced hangman)
```
Problem Statement
Given a lexicon of words, determine the most commonly associated letters.

For example given abc, bcd, cde we should end up with:
a:b,c (each with a value of 1)
b:c (value of 2)
c:b,d (each with value of 2)
d:c (value of 2)
e:c,d (each with a value of 1)
```

``` cpp
// 用词向量表示频率
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

void helper(vector<string>& nums) {
    int n = nums.size();
    vector<vector<bool>> m(26, vector<bool>(n, false));
    for (int i = 0; i < n; i++) {
        for (auto j : nums[i]) {
            m[j - 'a'][i] = m[j - 'a'][i] || true;
        }
    }
    for (auto i = 0; i < 26; i++) {
        vector<int> cur;
        int max = 0;
        for (int j = 0; j < 26; j++) {
            if (j == i) continue;
            int ans = 0;
            for (int k = 0; k < n; k++) {
                ans += (int)(m[i][k] & m[j][k]);
            }
            if (ans == max) cur.push_back(j);
            else if (ans > max) {
                max = ans;
                cur = {j};
            }
        }
        if (!max) continue;
        cout << (char)('a' + i) << ":";
        for (auto j : cur) {
            cout << (char)(j + 'a') << " ";
        }
        cout << endl;
    }
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
    helper(nums);
    return 0;
}
```

## random select from hash table
```
Problem
Design a hash table which allows random select, put, delete, get, and contain in most efficient way" (which is all operations in O(1)).

Use Case
This is used when we need to add unique elements to the hash map, but then allow randomly picking an element to dispatch. 
```

``` cpp
// 加一个反向索引
#include <iostream>
#include <unordered_map>
#include <utility>
#include <cstdlib>
#include <time.h>
#include <cstring>
using namespace std;

class MyTable {
private:
    unordered_map<int, string> id2key;
    unordered_map<string, pair<int, string>> nums;
    int cnt;
public:
    MyTable(): cnt(0) {}

    string rand_select() {
        if (!cnt) throw "Empty!";
        srand(time(NULL));
        int index = rand() % cnt;
        return nums[id2key[index]].second;
    }

    void put(string key, string val) {
        if (nums.find(key) == nums.end()) {
            id2key[cnt] = key;
            nums[key] = {cnt, val};
            cnt ++;
        }
        else {
            nums[key] = {nums[key].first, val};
        }
    }

    string get(string key) {
        if (nums.find(key) == nums.end()) throw "Not exisit";
        return nums[key].second;
    }
    
    void Delete(string key) {
        cout << "delete: " << key << endl;
        if (nums.find(key) == nums.end() || nums.empty()) throw "Not exisit";
        int index = nums[key].first;
        nums.erase(key);
        key = id2key[cnt - 1];
        nums[key] = {index, nums[key].second};
        id2key[index] = key;
        cnt --;
    }
    int size() {
        return cnt;
    }
};

int main() {
    auto inst = new MyTable();
    for (int i = 0; i < 6; i++) {
        inst->put(to_string(i), to_string(i + 1));
    }
    for (auto i = 0; i < 7; i++) {
        cout << inst->rand_select() << endl;
        system("pause");
        inst->Delete(to_string(i));
    }
    return 0;
}
````


## UTF-8 Validation
> 验证地址[Leetcode 393](https://leetcode.com/problems/utf-8-validation/)

``` cpp
class Solution {
public:
    bool validUtf8(vector<int>& data) {
        int cnt = 0;
        for (auto i : data) {
            if (!cnt) {
                if ((i >> 5) == 0b110) cnt = 1;
                else if ((i >> 4) == 0b1110) cnt = 2;
                else if ((i >> 3) == 0b11110) cnt = 3;
                else if (i >> 7) return false;
            }
            else {
                if ((i >> 6) != 0b10) return false;
                cnt --;
            }
        }
        return cnt == 0;
    }
};
```

## lower/upper bound
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int lower_bound(vector<int>& nums, int c) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < c) left = mid + 1;
        else right = mid; 
    }
    return left;
}

int upper_bound(vector<int>& nums, int c) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] <= c) left = mid + 1;
        else right = mid;
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

    int t;
    cin >> t;
    for (int i = 0; i < t; i++) {
        int c;
        cin >> c;
        cout << lower_bound(nums, c) << endl;
        cout << upper_bound(nums, c)<< endl;
    }

    return 0;
}
```

## 多路归并
数组归并
``` cpp
#include <iostream>
#include <vector>
#include <utility>

using namespace std;

void heapfy(vector<pair<int, int>>& ans, vector<vector<int>>& nums, int index, int max) {
    int left = index * 2 + 1;
    int right = left + 1;
    int smallest = index;
    if (left < max && nums[ans[left].first][ans[left].second] < nums[ans[smallest].first][ans[smallest].second]) smallest = left;
    if (right < max && nums[ans[right].first][ans[right].second] < nums[ans[smallest].first][ans[smallest].second]) smallest = right;
    if (smallest != index) {
        swap(ans[smallest], ans[index]);
        heapfy(ans, nums, smallest, max);
    }
}

vector<int> helper(vector<vector<int>>& nums) {
    vector<int> ret;
    vector<pair<int, int>> ans;
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        if (!nums[i].empty()) ans.push_back({i, 0});
    }
    for (int i = 0; i < ans.size(); i++) heapfy(ans, nums, i, ans.size());
    while (!ans.empty()) {
        auto cur = ans[0];
        ret.push_back(nums[cur.first][cur.second]);
        if (cur.second + 1 < nums[cur.first].size()) ans[0] = {cur.first, cur.second + 1};
        else {
            ans[0] = ans.back();
            ans.pop_back();
        }
        if (!ans.empty()) heapfy(ans, nums, 0, ans.size());
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<vector<int>> nums;
    for (int i = 0; i < n; i++) {
        int t;
        cin >> t;
        vector<int> temp(t, 0);
        for (int j = 0; j < t; j++) {
            cin >> temp[j];
        }
        nums.push_back(temp);
    }
    for (auto i : helper(nums)) {
        cout << i << endl;
    }
    return 0;
}
```

链表
``` cpp
#include <iostream>
#include <vector>
using namespace std;
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(NULL) {}
};
void heapfy(vector<ListNode*>& nums, int index, int max) {
    int left = index * 2 + 1;
    int right = left + 1;
    int smallest = index;
    if (left < max && nums[left]->val < nums[smallest]->val) smallest = left;
    if (right < max && nums[right]->val < nums[smallest]->val) smallest = right;
    if (smallest != index) {
        swap(nums[smallest], nums[index]);
        heapfy(nums, smallest, max);
    }
}

ListNode* helper(vector<ListNode*>& nums) {
    ListNode* ret = new ListNode(-1);
    auto cur = ret;
    for (int i = 0; i < nums.size() / 2; i ++) heapfy(nums, i, nums.size());
    while (!nums.empty()) {
        cur->next = nums[0];
        cur = cur->next;
        if (cur->next) {
            nums[0] = cur->next;
        }
        else {
            nums[0] = nums.back();
            nums.pop_back();
        }
        heapfy(nums, 0, nums.size());
    }
    cur->next = NULL;
    return ret->next;
}

int main() {
    int n;
    cin >> n;
    vector<ListNode*> nums;
    for (auto i = 0; i < n; i++) {
        auto ret = new ListNode(-1);
        auto cur = ret;
        int t;
        cin >> t;
        for (int j = 0; j < t; j++) {
            int c;
            cin >> c;
            cur->next = new ListNode(c);
            cur = cur->next;
        }
        nums.push_back(ret->next);
    }
    auto cur = helper(nums);
    while (cur) {
        cout << cur->val << endl;
        cur = cur->next;
    }
    return 0;
}
```

## Find Peak Element
> [Leetcode 162](https://leetcode.com/problems/find-peak-element/)
``` cpp
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        if (nums.empty()) return -1;
        int n = nums.size();
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) left = mid + 1;
            else right = mid;
        }
        return left;
    }
};
```

## 差值的绝对值第K大
题目：一个数组，任意两个数存在差值，求差值绝对值第K小是多少
> [Leetcode 719](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)
``` cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int count(vector<int>& nums, int target) {
    int n = nums.size();
    int left = 0, right = 1;
    int ret = 0;
    while (right < n) {
        while (left < right && nums[right] - nums[left] > target) left++;
        if (left < right) {
            ret += right - left;
        }
        right ++;
    }
    return ret;
}

int helper(vector<int>& nums, int k) {
    int n = nums.size();
    if (k > n * (n - 1) / 2) return -1;
    sort(nums.begin(), nums.end());
    int left = 0, right = nums.back() - nums[0];
    while (left < right) {
        int mid = left + (right - left) / 2;
        // 这种根据left是否需要+1判断使用upper_bound还是lower_bound
        // 这里需要在前面的个数小于自己的的时候left = mid + 1，所以前面求的小于的个数是包含mid的个数
        int cnt = count(nums, mid);
        if (cnt < k) left = mid + 1;
        else right = mid;
    }
    return left;
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    for (int i = 1; i <= k; i++) {
        cout << helper(nums, i) << endl;
    }
    return 0;
}
```

求第K大：

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
int search(vector<int>& nums, int target) {
    // 双指针，判断有序数组差值大于target的个数
    int n = nums.size();
    int left = 0, right = 0;
    int cnt = 0;
    while (left <= right) {
        while (right < n && nums[right] - nums[left] <= target) right ++;
        if (right < n) cnt += n - right; // right之后的和left相减都大于target
        left ++;
    }
    return cnt;
}

int helper(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    int left = 0, right = nums.back() - nums[0];
    while (left < right) {
        int mid = left + (right - left) / 2;
        int cnt = search(nums, mid);
        cout << mid << " " << cnt << endl;
        // 因为需要将left + 1，所以求得是比mid大于等于的个数
        if (cnt >= k) left = mid + 1;
        else right = mid;
    }
    return left;
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

## 字典序第K大(*)
> [Leetcode 440](https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/)

``` cpp
class Solution {
public:
    int findKthNumber(int n, int k)
    {
        int result = 1;
        for(--k; k > 0; )
        {
            // calculate #|{result, result*, result**, result***, ...}|
            int count = 0;
            for (long long first = static_cast<long long>(result), last = first + 1;
                first <= n; // the interval is not empty
                first *= 10, last *= 10) // increase a digit
            {
                // valid interval = [first, last) union [first, n]
                count += static_cast<int>((min(n + 1LL, last) - first)); // add the length of interval
            }
            
            if (k >= count)
            {   // skip {result, result*, result**, result***, ...}
                // increase the current prefix
                ++result;
                k -= count;
            }
            else
            {   // not able to skip all of {result, result*, result**, result***, ...}
                // search more detailedly
                result *= 10;
                --k;
            }
        }
        return result;
    }
};
```

## Kth Smallest Number in Multiplication Table(*)
> [Leetcode 668](https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/)

``` cpp
class Solution {
public:
    int findKthNumber(int m, int n, int k) {
        int lo = 1, hi = m*n;//[lo, hi)
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = 0,  j = m;
            for(int i = 1; i <= n; i++) {
                while(j >=1 && i*j > mid) j--;
                count += (j);
            }
            if(count < k) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};
```

## 将0移到最前面
[1,2,3,0,0,9]这样的数组，把0移到最前面，此题类似于剑指offer偶数在奇数之前。三种解法。
``` cpp
#include <iostream>
#include <vector>

using namespace std;

vector<int> helper1(vector<int> nums) {
    // 保证相对顺序且空间O(1)
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        for (int j = n - 1; j > i; j--) {
            if (nums[j] == 0 && nums[j - 1] != 0) {
                swap(nums[j - 1], nums[j]);
            }
        }
    }
    return nums;
}

vector<int> helper2(vector<int> nums) {
    // 不要求相对顺序时间O(n)
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left < right) {
        while (left < right && !nums[left]) left ++;
        while (left < right && nums[right]) right --;
        if (left < right) {
            swap(nums[left], nums[right]);
            left ++, right --;
        }
    }
    return nums;
}

vector<int> helper3(vector<int> nums) {
    // 保证相对顺序时间O(n)
    int cnt = 0, n = nums.size();
    for (auto i : nums) {
        if (!i) cnt ++;
    }
    vector<int> ret(n, 0);
    int l = 0, r = 0;
    for (auto i : nums) {
        if (i) ret[cnt + r] = i, r ++;
        else ret[l] = i, l ++;
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
    for (auto i : helper1(nums)) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i : helper2(nums)) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i : helper3(nums)) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```


## Sliding Window Maximum
> [Leetcode 239](https://leetcode.com/problems/sliding-window-maximum/)

``` cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        deque<int> q;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (!q.empty() && nums[q.back()] <= nums[i]) q.pop_back();
            q.push_back(i);
            while (!q.empty() && q.front() <= i - k) q.pop_front();
            if (i >= k - 1) ret.push_back(nums[q.front()]);
        }
        return ret;
    }
};
```

## 质数异或组合
给一个不重复正数集合，如果一个子数组按位异或得到的结果是质数，那么就满足条件，求满足条件的子数组个数。例如[2, 4, 7]，符合条件的有[2] [7] [2,7] [4,7],然后数的范围是小于5000，然后数组长度范围是小于5000。

``` cpp
#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
using namespace std;

int helper(vector<int>& nums) {
    int n = nums.size();
    vector<int> fac(5000, true);
    fac[1] = false;
    for (int i = 2; i <= 5000; i++) {
        if (!fac[i]) continue;
        for (int j = 2; j * i <= 5000; j++) {
            fac[i * j] = false;
        }
    }

    unordered_map<int, vector<unordered_set<int>>> dp;
    for (auto i : nums) {
        dp[i] = {{i}};
    }

    for (int i = 2; i <= 5000; i++) {
        for (int j : nums) {
            // 判重复
            if ((j ^ i) < j && dp.find(i ^ j) != dp.end()) {
                auto temp = dp[i ^ j];
                for (auto k : temp) {
                    if (k.find(j) != k.end()) continue;
                    k.insert(j);
                    dp[i].push_back(k);
                }
            }
        }
    }

    int ret = 0;
    for (auto i : dp) {
        if (!fac[i.first]) continue;
        cout << i.first << ":" << endl;
        for (auto j : i.second) {
            for (auto k : j) {
                cout << k << " ";
            }
            ret ++;
            cout << endl;
        }
    }
    cout << endl;
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
/*
3
2 4 9

3
2 4 7

4
3 2 4 9
*/
```

## 包含所有字符的最短子串
给定一个字符串，求它的最短子串，使得原字符串中所有出现过的字符都在这个子串中
``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <climits>
#include <math.h>

using namespace std;

string helper(string s, string p) {
    unordered_map<char, int> m;
    for (auto i : p) m[i] ++;

    int ans = m.size();
    string ret;
    int n = s.size();
    int len = INT_MAX;
    int l = 0, r = 0;
    while (l <= r && r <= n) {
        if (r < n && ans) {
            auto c = s[r ++];
            if (m.find(c) != m.end()) {
                m[c] --;
                if (!m[c]) ans --;
            }
            
        }
        else {
            auto c = s[l ++];
            if (m.find(c) != m.end()) {
                m[c] ++;
                if (m[c] == 1) ans ++;
            }
            
        }
        if (!ans) {
            if (r - l < len) {
                len = r - l;
                ret = s.substr(l, len);
            }
        }
    }
    return ret;
}

int main() {
    string s, p;
    cin >> s >> p;
    cout << helper(s, p) << endl;
    return 0;
}
```

## Rotated Sorted Array求旋转了多少位
``` cpp
#include <vector>
#include <iostream>

using namespace std;

int helper(vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return n;
    int l = 0, r = n - 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] == nums[r]) r --;
        else if (nums[mid] < nums[mid + 1]) r = mid;
        else l = mid + 1;
    }
    return l;
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

## merge intervals
> [Leetcode 56](https://leetcode.com/problems/merge-intervals/)

``` cpp
class Solution {
public:
    bool static cmp(const vector<int>& a, const vector<int>& b) {
        return a[0] == b[0] ? a[1] < b[1] : a[0] < b[0];
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ret;
        sort(intervals.begin(), intervals.end(), cmp);
        for (auto i : intervals) {
            if (ret.empty()) ret.push_back(i);
            else if (ret.back()[1] >= i[0]) ret.back()[1] = max(ret.back()[1], i[1]);
            else ret.push_back(i);
        }
        return ret;
    }
};
```

## 折线图的在Y轴上覆盖最多的区间
一个折线图，例如[1,2,1,2,1,2,3],求左闭右开覆盖最多的区间是多少，例如此例子就是[1, 2)。
``` cpp
// 此例子类似于会议室那道题，求在会议室中最多的人数是多少
// 但是不同于会议室那道题，这道题是求区间，所以不能用前缀和
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>
using namespace std;

vector<int> helper(vector<int>& nums) {
    int start = INT_MAX;
    int end = INT_MIN;
    for (int i : nums) {
        start = min(start, i);
        end = max(end, i);
    }
    int len = end - start + 1;
    vector<int> ans(len, 0);
    for (int i = 0; i < nums.size() - 1; i++) {
        if (nums[i] < nums[i + 1]) {
            ans[nums[i] - start] ++;
            ans[nums[i + 1] - start] --;
        }
        else {
            ans[nums[i] - start] --;
            ans[nums[i + 1] - start] ++;
        }
    }
    int cnt = 0;
    int mx = INT_MIN;
    vector<int> ret(2, start);
    vector<int> cur(2, start);

    for (int i = 0; i < len; i++) {
        cnt += ans[i];
        if (!cnt) {
            cnt = 0;
            cur = {i + 1 + start, i + 1 + start};
        }
        else {
            // 此部分是用于将右侧开空间或延伸，如果小于0就不会进入与mx的比较
            if (ans[i] <= 0) {
                cur[1] = i + 1 + start;
            }
            else {
                cur = {i + start, i + start + 1};
            }
            // 必须要有等于，否则无法向后延伸，比如[1,3,1,3,1,4]就会返回[1,2)
            if (cnt >= mx) {
                mx = cnt;
                ret = cur;
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

如果是两边都是闭区间，则需要将右边界+1，此时可能会选择点
``` cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>
using namespace std;

vector<int> helper(vector<int>& nums) {
    int start = INT_MAX;
    int end = INT_MIN;
    for (int i : nums) {
        start = min(start, i);
        end = max(end, i);
    }
    int len = end - start + 1;
    vector<int> ans(len + 1, 0);
    for (int i = 0; i < nums.size() - 1; i++) {
        if (nums[i] < nums[i + 1]) {
            ans[nums[i] - start] ++;
            ans[nums[i + 1] - start + 1] --;
        }
        else {
            ans[nums[i] - start + 1] --;
            ans[nums[i + 1] - start] ++;
        }
    }
    int cnt = 0;
    int mx = INT_MIN;
    vector<int> ret(2, start);
    vector<int> cur(2, start);

    for (int i = 0; i < len; i++) {
        cnt += ans[i];
        if (!cnt) {
            cnt = 0;
            cur = {i + 1 + start, i + 1 + start};
        }
        else {
            if (ans[i] > 0) {
                cur = {i + start, i + start};
            }
            else {
                cur[1] = i + start;
            }
            if (cnt >= mx) {
                mx = cnt;
                ret = cur;
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

/*
7
1 3 1 3 1 3 4

7 
3 1 3 1 3 1 4

*/
```


## HDU 5776 sum
> [HDU 5576](https://blog.csdn.net/elbadaernu/article/details/78728718)
给定n个整数，是否存在一个子数组（下标连续）的和对m取模等于0 


思路：假设sum[i]表示前i个数的前缀和，如果存在sum[i]%m==sum[j]%m，那么（sum[j]-sum[i]）%m==0。

``` cpp
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

bool helper(vector<int>& nums, int m) {
    int n = nums.size();
    unordered_map<int, int> hash;

    int ans = 0;
    for (auto i : nums) {
        ans += i;
        if (hash.find(ans % m) != hash.end()) {
            cout << ans << " " << hash[ans % m] << endl;
            return true;
        }
        hash[ans % m] = i;
    }
    return false;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    cout << helper(nums, m) << endl;
    return 0;
}
```


## 矩阵填数
> [hihocoder 1480](https://www.cnblogs.com/hua-dong/p/8453923.html)
复习完钩子定理和卡特兰数更新。。。