#include <bits/stdc++.h>
using namespace std;
typedef  pair<int,int> pi ;
#define if1(x) for(int i =1 ;i<=x;i++)
#define if0(x) for(int i = 0;i<x;i++)
#define jf0(x) for(int j = 0;j<x;j++)
#define jf1(x) for(int j = 1;j<=x;j++)
// 定义一个模板类 Bag，作为所有容器类的基类
template <typename T>
class Bag {
public:
    virtual void Add(const T& item) = 0; // 添加一个元素
    virtual void Remove() = 0; // 移除一个元素
    virtual T Get() const = 0; // 获取一个元素
    virtual void Put(const T& item) = 0; // 放置一个元素
    virtual void MakeEmpty() = 0; // 清空容器
    virtual bool Full() const = 0; // 判断容器是否满
    virtual int Length() const = 0; // 获取容器长度
    virtual void SaveToFile(const std::string& filename) const = 0; // 保存到文件
    virtual void LoadFromFile(const std::string& filename) = 0; // 从文件加载
    virtual ~Bag() {} // 虚析构函数
};
 
// 定义一个模板类 Stack，继承自 Bag
template <typename T>
class Stack : public Bag<T> {
private:
    std::vector<T> data; // 使用 vector 存储数据
public:
    void Add(const T& item) override {
        data.push_back(item); // 添加元素到栈顶
    }
    void Remove() override {
        if (!data.empty()) {
            data.pop_back(); // 移除栈顶元素
        }
    }
    T Get() const override {
        if (!data.empty()) {
            return data.back(); // 获取栈顶元素
        }
        throw std::out_of_range("栈为空！！！"); // 栈为空时抛出异常
    }
    void Put(const T& item) override {
        if (!data.empty()) {
            data.back() = item; // 替换栈顶元素
        } else {
            throw std::out_of_range("栈为空！！！"); // 栈为空时抛出异常
        }
    }
    void MakeEmpty() override {
        data.clear(); // 清空栈
    }
    bool Full() const override {
        return false; // 栈可以动态增长，不会满
    }
    int Length() const override {
        return data.size(); // 返回栈的长度
    }
    void SaveToFile(const std::string& filename) const override {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& item : data) {
                file << item << std::endl; // 将数据写入文件
            }
        }
    }

    void LoadFromFile(const std::string& filename) override {
        std::ifstream file(filename);
        T item;
        data.clear();
        while (file >> item) {
            data.push_back(item); // 从文件读取数据
        }
    }
};
// 定义一个模板类 Queue，继承自 Bag
template <typename T>
class Queue : public Bag<T> {
private:
    std::vector<T> data; // 使用 vector 存储数据
public:
    void Add(const T& item) override {
        data.push_back(item); // 添加元素到队尾
    }
    void Remove() override {
        if (!data.empty()) {
            data.erase(data.begin()); // 移除队首元素
        }
    }

    T Get() const override {
        if (!data.empty()) {
            return data.front(); // 获取队首元素
        }
        throw std::out_of_range("队列为空！！！！"); // 队列为空时抛出异常
    }

    void Put(const T& item) override {
        if (!data.empty()) {
            data.front() = item; // 替换队首元素
        } else {
            throw std::out_of_range("队列为空！！！！"); // 队列为空时抛出异常
        }
    }

    void MakeEmpty() override {
        data.clear(); // 清空队列
    }

    bool Full() const override {
        return false; // 队列可以动态增长，不会满
    }

    int Length() const override {
        return data.size(); // 返回队列的长度
    }

    void SaveToFile(const std::string& filename) const override {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& item : data) {
                file << item << std::endl; // 将数据写入文件
            }
        }
    }

    void LoadFromFile(const std::string& filename) override {
        std::ifstream file(filename);
        T item;
        data.clear();
        while (file >> item) {
            data.push_back(item); // 从文件读取数据
        }
    }
};

void init1(){
    //栈的初始化
    Stack<int> stack;
    if1(10) stack.Add(i);
    stack.SaveToFile("stack.txt"); // 保存栈到文件
    //队列初始化
    Queue<int> queue;
    if1(10)queue.Add(i);
    queue.SaveToFile("queue.txt"); // 保存队列到文件
}
void showop1(){
    init1();//调用
    Stack<int> stack2;
    stack2.LoadFromFile("stack.txt"); // 从文件加载栈
    while (stack2.Length() > 0) {
        std::cout << stack2.Get() << " "; // 输出栈顶元素
        stack2.Remove(); // 移除栈顶元素
    }
    std::cout << std::endl;
    Queue<int> queue2;
    queue2.LoadFromFile("queue.txt"); // 从文件加载队列
    while (queue2.Length() > 0) {
        std::cout << queue2.Get() << " "; // 输出队首元素
        queue2.Remove(); // 移除队首元素
    }
    std::cout << std::endl;
}

// 定义模板类 TPoint
template<typename T>
class TPoint {
public:
    T x, y;
    TPoint(T x = 0, T y = 0) : x(x), y(y) {}

    // 向量加法
    TPoint operator+(const TPoint& p) const {
        return TPoint(x + p.x, y + p.y);
    }

    // 向量减法
    TPoint operator-(const TPoint& p) const {
        return TPoint(x - p.x, y - p.y);
    }

    // 向量加法赋值
    TPoint& operator+=(const TPoint& p) {
        x += p.x;
        y += p.y;
        return *this;
    }

    // 向量减法赋值
    TPoint& operator-=(const TPoint& p) {
        x -= p.x;
        y -= p.y;
        return *this;
    }

    // 点乘
    T dot(const TPoint& p) const {
        return x * p.x + y * p.y;
    }

    // 叉乘
    T cross(const TPoint& p) const {
        return x * p.y - y * p.x;
    }

    // 向量长度
    T length() const {
        return std::sqrt(x * x + y * y);
    }

    // 角度（弧度）
    T angle() const {
        return std::atan2(y, x);
    }
};

// 定义模板类 TMath 继承自 TPoint
template<typename T>
class TMath : public TPoint<T> {
public:
    // 计算三角形面积（通过三个点）
    static T triangleArea(const TPoint<T>& a, const TPoint<T>& b, const TPoint<T>& c) {
        return std::abs((b - a).cross(c - a)) / 2.0;
    }

    // 计算圆形面积（通过半径）
    static T circleArea(T radius) {
        return M_PI * radius * radius;
    }

    // 判断两直线的位置关系
    static std::string lineRelation(const TPoint<T>& a1, const TPoint<T>& a2, const TPoint<T>& b1, const TPoint<T>& b2) {
        TPoint<T> da = a2 - a1;
        TPoint<T> db = b2 - b1;
        T crossProduct = da.cross(db);

        if (std::abs(crossProduct) < 1e-10) { // 平行或重合
            T dotProduct = (b1 - a1).dot(da);
            if (std::abs(dotProduct) < 1e-10) {
                return "这两条直线重合了！！！！"; // 重合
            } else {
                return "这两条直线是平行的"; // 平行
            }
        } else {
            return "这两条直线是相交的"; // 相交
        }
    }

    // 计算两个向量之间的夹角余弦值（通过四个点）
    static T cosineAngle(const TPoint<T>& a, const TPoint<T>& b, const TPoint<T>& c, const TPoint<T>& d) {
        TPoint<T> ab = b - a;
        TPoint<T> cd = d - c;
        T dotProduct = ab.dot(cd);
        T lengthProduct = ab.length() * cd.length();
        return dotProduct / lengthProduct;
    }

    // 计算三个点构成的两个向量之间的夹角余弦值（通过三个点）
    static T cosineAngle(const TPoint<T>& a, const TPoint<T>& b, const TPoint<T>& c) {
        TPoint<T> ab = b - a;
        TPoint<T> ac = c - a;
        T dotProduct = ab.dot(ac);
        T lengthProduct = ab.length() * ac.length();
        return dotProduct / lengthProduct;
    }

    // 计算点到直线的距离（通过三个点）
    static T pointToLineDistance(const TPoint<T>& a, const TPoint<T>& b, const TPoint<T>& c) {
        TPoint<T> ab = b - a;
        TPoint<T> ac = c - a;
        return std::abs(ab.cross(ac)) / ab.length();
    }

    // 计算点到线段的距离（通过三个点）
    static T pointToSegmentDistance(const TPoint<T>& a, const TPoint<T>& b, const TPoint<T>& c) {
        TPoint<T> ab = b - a;
        TPoint<T> ac = c - a;
        TPoint<T> bc = c - b;

        T e = ac.dot(ab);
        if (e <= 0) {//叉乘小于0，余弦值为负数，角度大于90
            return ac.length();
        }

        T f = ab.dot(ab);
        if (e >= f) {
            return bc.length();
        }

        return std::sqrt(ac.dot(ac) - (e * e / f));
    }
};

void showop2()
{
    TPoint<double> p1(0, 0), p2(1, 0), p3(0, 1), p4(1, 1);
    TPoint<double> p5(2, 3);

    // 计算三角形面积
    double area = TMath<double>::triangleArea(p1, p2, p3);
    std::cout << "三角形面积: " << area << std::endl;

    // 计算圆形面积
    double radius = 5.0;
    double circleArea = TMath<double>::circleArea(radius);
    std::cout << "圆形面积: " << circleArea << std::endl;

    // 判断两直线的位置关系
    TPoint<double> l1a(0, 0), l1b(1, 1);
    TPoint<double> l2a(0, 1), l2b(1, 2);
    std::string relation = TMath<double>::lineRelation(l1a, l1b, l2a, l2b);
    std::cout << "直线关系: " << relation << std::endl;

    // 计算两个向量之间的夹角余弦值（通过四个点）
    double cosineABCD = TMath<double>::cosineAngle(p1, p2, p3, p4);
    std::cout << "向量 (p1->p2) 和 (p3->p4) 之间夹角的余弦值: " << cosineABCD << std::endl;

    // 计算三个点构成的两个向量之间的夹角余弦值（通过三个点）
    double cosineABC = TMath<double>::cosineAngle(p1, p2, p3);
    std::cout << "向量 (p1->p2) 和 (p1->p3) 之间夹角的余弦值: " << cosineABC << std::endl;

    // 计算点到直线的距离
    double distanceToLine = TMath<double>::pointToLineDistance(p3, p2, p1);
    std::cout << "点 p1 到直线 (p2, p3) 的距离: " << distanceToLine << std::endl;

    // 计算点到线段的距离
    p3 = {2,-1};
    double distanceToSegment = TMath<double>::pointToSegmentDistance(p3, p2, p1);
    std::cout << "点 p1 到线段 (p2, p3) 的距离: " << distanceToSegment << std::endl;

    // 测试向量加法赋值和减法赋值
    p5 += p1;
    std::cout << "p5 += p1 后: (" << p5.x << ", " << p5.y << ")" << std::endl;

    p5 -= p2;
    std::cout << "p5 -= p2 后: (" << p5.x << ", " << p5.y << ")" << std::endl;

}


template<typename T>
class SegmentTree {
private:
    struct Node {
        int l, r;
        T sum;
        T add;
    };

    vector<Node> tr;
    const int N;

    void pushup(int u){
    int sl = 0,sr = 0;
    sl = (tr[u<<1].r-tr[u<<1].l+1)*tr[u<<1].add;
    sr = (tr[u<<1|1].r-tr[u<<1|1].l+1)*tr[u<<1|1].add;
    tr[u].sum = tr[u<<1].sum +tr[u<<1|1].sum+sl+sr;
    }
    void pushdown(int u) {
        if (tr[u].add == 0) return;
        tr[u].sum += (tr[u].r - tr[u].l + 1) * tr[u].add;
        tr[u << 1].add += tr[u].add;
        tr[u << 1 | 1].add += tr[u].add;
        tr[u].add = 0;
    }

    // 建树操作
    void build(int u, int l, int r, const vector<T>& w) {
        tr[u] = {l, r, 0, 0};
        if (l == r) {
            tr[u].sum = w[l];
            return;
        }
        int mid = (l + r) >> 1;
        build(u << 1, l, mid, w);
        build(u << 1 | 1, mid + 1, r, w);
        pushup(u);
    }

    // 区间修改操作
    void modify(int u, int l, int r, T x) {
        if (tr[u].l >= l && tr[u].r <= r) {
            if (tr[u].l == tr[u].r) tr[u].sum += x;
            else tr[u].add += x;
            return;
        }
        pushdown(u);
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (mid >= l) modify(u << 1, l, r, x);
        if (r > mid) modify(u << 1 | 1, l, r, x);
        pushup(u);
    }

    // 区间查询操作
    T query(int u, int l, int r) {
        if (tr[u].l >= l && tr[u].r <= r) {
            return tr[u].sum + tr[u].add * (tr[u].r - tr[u].l + 1);
        }
        pushdown(u);
        int mid = (tr[u].l + tr[u].r) >> 1;
        T v = 0;
        if (mid >= l) v += query(u << 1, l, r);
        if (r > mid) v += query(u << 1 | 1, l, r);
        return v;
    }

public:
    SegmentTree(int size) : N(size) {
        tr.resize(4 * N);
    }

    void build(const vector<T>& data) {
        build(1, 0, N - 1, data);
    }

    void modify(int l, int r, T x) {
        modify(1, l, r, x);
    }

    T query(int l, int r) {
        return query(1, l, r);
    }
};

class HLD {//树链剖分， Heavy-light Decomposition
public:
    int n, m;
    vector<int> a;//记录初始状态的节点的权值。
    vector<vector<int>> adj; // 邻接表
    vector<int> id, nw, top, fa, dep, sz, son;
    int idx;
    SegmentTree<int> segTree;

    // 深度优先搜索1，用于计算每个节点的深度、父节点、子树大小和重儿子
    void dfs1(int u, int f, int d) {
        dep[u] = d, fa[u] = f, sz[u] = 1;
        for (int j : adj[u]) {
            if (j == f) continue;
            dfs1(j, u, d + 1);
            sz[u] += sz[j];
            if (sz[son[u]] < sz[j]) son[u] = j;
        }
    }

    // 深度优先搜索2，用于分配节点的链顶、位置和重链剖分
    void dfs2(int u, int t) {
        id[u] = ++idx, nw[idx] = a[u], top[u] = t;
        if (!son[u]) return;
        dfs2(son[u], t);
        for (int j : adj[u]) {
            if (j == fa[u] || j == son[u]) continue;
            dfs2(j, j);
        }
    }

public:
    HLD(int n, const vector<int>& values) : n(n), a(values), segTree(n) {
        adj.resize(n + 1);
        id.resize(n + 1);
        nw.resize(n + 1);
        top.resize(n + 1);
        fa.resize(n + 1);
        dep.resize(n + 1);
        sz.resize(n + 1);
        son.resize(n + 1, 0);
        idx = 0;
    }

    // 添加边
    void add_edge(int a, int b) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // 预处理，进行深度优先搜索和建树
    void preprocess() {
        dfs1(1, -1, 1);
        dfs2(1, 1);
        vector<int> data(n + 1);
        for (int i = 1; i <= n; ++i) data[i] = nw[i];
        segTree.build(data);
    }

    // 查询两个节点的最近公共祖先
    int LCA(int x, int y) {
        while (top[x] != top[y]) {
            if (dep[top[x]] < dep[top[y]]) swap(x, y);
            x = fa[top[x]];
        }
        return dep[x] < dep[y] ? x : y;
    }

    // 更新路径上的节点
    void update_path(int u, int v, int k) {
        while (top[u] != top[v]) {
            if (dep[top[u]] < dep[top[v]]) swap(u, v);
            segTree.modify(id[top[u]], id[u], k);
            u = fa[top[u]];
        }
        if (dep[u] < dep[v]) swap(u, v);
        segTree.modify(id[v], id[u], k);
    }

    // 查询路径上的节点和
    int query_path(int u, int v) {
        int res = 0;
        while (top[u] != top[v]) {
            if (dep[top[u]] < dep[top[v]]) swap(u, v);
            res += segTree.query(id[top[u]], id[u]);
            u = fa[top[u]];
        }
        if (dep[u] < dep[v]) swap(u, v);
        res += segTree.query(id[v], id[u]);
        return res;
    }

    // 更新子树上的节点
    void update_tree(int u, int k) {
        segTree.modify(id[u], id[u] + sz[u] - 1, k);
    }

    // 查询子树上的节点和
    int query_tree(int u) {
        return segTree.query(id[u], id[u] + sz[u] - 1);
    }
};

void showop3()
{
    int n = 5;
    vector<int> values = {0, 1, 2, 3, 4, 5}; // 0号索引未使用
    HLD tree(n, values);
    tree.add_edge(1, 2);
    tree.add_edge(1, 3);
    tree.add_edge(2, 4);
    tree.add_edge(2, 5);
    tree.preprocess();
    cout << "从节点 4 到节点 5 的路径权重和: " << tree.query_path(4, 5) << endl;
    tree.update_path(4, 5, 1);
    cout << "更新后从节点 4 到节点 5 的路径权重和: " << tree.query_path(4, 5) << endl;
    cout << "以节点 2 为根的子树和: " << tree.query_tree(2) << endl;
    tree.update_tree(2, 1);
    cout << "更新后以节点 2 为根的子树和: " << tree.query_tree(2) << endl;
}
signed main() {
    showop1();
    showop2();
    showop3();
    return 0;
}
