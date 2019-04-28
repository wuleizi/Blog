---
title: 【面经】HotStar部分编程算法题
date: 2019-04-21 22:37:33
tags: [算法, 总结, 面经]
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
int helper(Node* root, int& ret) {
    if (!root) return INT_MAX;
    int ans = INT_MAX;
    if (root->left) ans = min(ans, helper(root->left, ret));
    if (root->right) ans = min(ans, helper(root->right, ret));
    if (ans == INT_MAX) return root->data;
    ret = max(ret, root->data - ans);
    return min(root->data, ans);
}
int maxDiff(Node* root)
{
    // Your code here 
    if (!root) return INT_MIN;
    int ret = INT_MIN;
    helper(root, ret);
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
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace std;
void next_permutation(string& s) {
    int n = s.size();
    int i = n - 2;
    while (i >= 0 && s[i] >= s[i + 1]) i--;
    if (i >= 0) {
        int j = n - 1;
        while (s[j] <= s[i]) j --;
        swap(s[j], s[i]);
    }
    reverse(s.begin() + i + 1, s.end());
    // 如果reverse放在外面，则会无限循环，当整体已经处于降序的情况下，reverse会变成升序，此时就会变成全排列的第一个
}

string helper(int n, int k) {
    string ret;
    for (int i = 1; i <= n; i++) {
        ret = ret + to_string(i);;
    }
    for (int i = 0; i < k - 1; i++) {
        next_permutation(ret);
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
private:
    unordered_map<string, string> m;
    long long cnt;
    unordered_map<char, int> dict;
public:
    Solution(): cnt(0) {
        int index = 0;
        for (char i = '0'; i <= '9'; i++) dict[i] = index++;
        for (char i = 'a'; i <= 'z'; i++) dict[i] = index++;
        for (char i = 'A'; i <= 'Z'; i++) dict[i] = index++;
    }
    
    // Encodes a URL to a shortened URL.
    string encode(string longUrl) {
        cnt ++;
        long long ans = cnt;
        string ret;
        while (ans) {
            ret.push_back(dict[ans % 62]);
            ans /= 62;
        }
        m[ret] = longUrl;
        return ret;
    }

    // Decodes a shortened URL to its original URL.
    string decode(string shortUrl) {
        if (m.find(shortUrl) == m.end()) return "";
        else return m[shortUrl];
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

> 参考[Leetcode ]