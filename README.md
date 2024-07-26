# cpp_goods_tudy
cq某cpp程序设计实践小众项目_（关于类的实现和拓展），满绩（因为按照老师要求做的）

首先，这个是我们老师的期末要求，我是在完成这个的基础之上做了一些拓展
	试建立一个继承结构，以栈、队列为派生类，建立它们的抽象基类-Bag类，写出各个类的声明及定义，并实现如下功能：
	统一命名各派生类的插入操作为Add，删除操作为Remove。
	统一命名各派生类的存取操作为Get和Put。
	统一命名各派生类的初始化操作为MakeEmpty，判空操作为Full，计数操作为Length。
	要求能将一个栈或队列的内容存入一个文件中，并可从一个文件中读入一个栈或队列

CSDN链接
https://blog.csdn.net/qq_51677674/article/details/140727600?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22140727600%22%2C%22source%22%3A%22qq_51677674%22%7D



系统应实现的主要功能
一：Queue 和 Stack 类的实现

流程图：
 ![图片](https://github.com/user-attachments/assets/cfd9f7cf-32bb-4095-92ce-882d61264aec)

1. Bag 抽象基类
Bag 是一个抽象基类，定义了一系列纯虚函数，这些函数必须由继承它的类实现。它提供了以下接口：
•	Add(const T& item) 添加一个元素
•	Remove() 移除一个元素
•	Get() const 获取一个元素
•	Put(const T& item) 放置一个元素
•	MakeEmpty() 清空容器
•	Full() const 判断容器是否满
•	Length() const 获取容器长度
•	SaveToFile(const stdstring& filename) const 保存到文件
•	LoadFromFile(const stdstring& filename) 从文件加载
•	虚析构函数：确保正确销毁派生类对象

2. Stack 类
Stack 类继承自 Bag，并实现了栈的功能。主要特点包括：
•	使用 stdvectorT 存储数据
•	Add(const T& item) 向栈顶添加元素
•	Remove() 移除栈顶元素
•	Get() const 获取栈顶元素
•	Put(const T& item) 替换栈顶元素
•	MakeEmpty() 清空栈
•	Full() const 栈可以动态增长，不会满
•	Length() const 返回栈的长度
•	SaveToFile(const stdstring& filename) const 将栈保存到文件
•	LoadFromFile(const stdstring& filename) 从文件加载栈
3. Queue 类
Queue 类也继承自 Bag，并实现了队列的功能。主要特点包括：
•	使用 stdvectorT 存储数据
•	Add(const T& item) 向队尾添加元素
•	Remove() 移除队首元素
•	Get() const 获取队首元素
•	Put(const T& item) 替换队首元素
•	MakeEmpty() 清空队列
•	Full() const 队列可以动态增长，不会满
•	Length() const 返回队列的长度
•	SaveToFile(const stdstring& filename) const 将队列保存到文件
•	LoadFromFile(const stdstring& filename) 从文件加载队列
4. init1 函数
init1 函数用于初始化栈和队列，并将它们保存到文件中：
•	创建 Stackint 对象，添加 10 个整数（假设 if1(10) 是一个循环）
•	将栈保存到 stack.txt
•	创建 Queueint 对象，添加 10 个整数
•	将队列保存到 queue.txt
5. showop1 函数
showop1 函数用于展示从文件加载栈和队列的操作：
•	调用 init1 函数初始化数据
•	创建 Stackint 对象 stack2，从 stack.txt 加载数据，并输出栈顶元素直到栈为空
•	创建 Queueint 对象 queue2，从 queue.txt 加载数据，并输出队首元素直到队列为空
二：Point 类和计算几何方法类的实现
1.	TPoint 模板类
流程图：
 ![图片](https://github.com/user-attachments/assets/75994af1-6d29-4718-a20c-0dc9f391f36c)

TPoint 类是一个通用的二维点或向量类，提供了以下功能：
•	构造函数：初始化二维点的坐标。
•	向量加法：重载 + 运算符，实现两个向量的加法。
•	向量减法：重载 - 运算符，实现两个向量的减法。
•	向量加法赋值：重载 += 运算符，实现向量加法并赋值。
•	向量减法赋值：重载 -= 运算符，实现向量减法并赋值。
•	点乘：计算两个向量的点积。
•	叉乘：计算两个向量的叉积。
•	向量长度：计算向量的模长。
•	角度（弧度）：计算向量与 x 轴的夹角（弧度）。
2. TMath 模板类
TMath 类继承自 TPoint，提供了一些常见的几何计算功能：
•	计算三角形面积：通过三个点计算三角形的面积。
•	计算圆形面积：通过半径计算圆形的面积。
•	判断两直线的位置关系：通过四个点判断两条直线是平行、重合还是相交。
•	计算两个向量之间的夹角余弦值：通过四个点或三个点计算两个向量之间夹角的余弦值。
•	计算点到直线的距离：通过三个点计算点到直线的距离。
•	计算点到线段的距离：通过三个点计算点到线段的距离。
3. showop2 函数
showop2 对 TPoint 和 TMath 类进行一些几何计算操作的功能展示：
•	计算三角形面积：通过三个点计算并输出三角形的面积。
•	计算圆形面积：通过半径计算并输出圆形的面积。
•	判断两直线的位置关系：通过四个点判断并输出两条直线的位置关系。
•	计算两个向量之间的夹角余弦值：通过四个点或三个点计算并输出两个向量之间夹角的余弦值。
•	计算点到直线的距离：通过三个点计算并输出点到直线的距离。
•	计算点到线段的距离：通过三个点计算并输出点到线段的距离。
•	测试向量加法赋值和减法赋值：展示了向量加法赋值和减法赋值的操作结果。
三：线段树和树剖类的实现

流程图
![图片](https://github.com/user-attachments/assets/0f192dac-0266-48b0-ac58-1913fc9ca5b4)

 
1. SegmentTree 模板类
SegmentTree 类是一个通用的线段树实现，支持区间更新和区间查询操作。主要功能包括：
•	构造函数：初始化线段树的大小。
•	建树操作：根据输入数据构建线段树。
•	区间修改操作：对指定区间进行加法更新。
•	区间查询操作：查询指定区间的和。
2. HLD 类
HLD 类实现了树链剖分，用于处理树上的路径和子树的区间更新和查询操作。主要功能包括：
•	构造函数：初始化树的大小和节点的初始值。
•	添加边：构建树的邻接表。
•	预处理：进行两次深度优先搜索，计算每个节点的深度、父节点、子树大小、重儿子，以及进行重链剖分并建立线段树。
•	查询最近公共祖先（LCA）：查询两个节点的最近公共祖先。
•	更新路径上的节点：对路径上的节点进行区间加法更新。
•	查询路径上的节点和：查询路径上的节点和。
•	更新子树上的节点：对以某个节点为根的子树进行区间加法更新。
•	查询子树上的节点和：查询以某个节点为根的子树的节点和。
详细功能及其作用
1. SegmentTree 类
•	pushup：更新父节点的值，确保父节点的值等于其子节点的和。
•	pushdown：将懒惰标记传递给子节点，确保在查询或更新时子节点的值是最新的。
•	build：递归建树，将输入数据构建为线段树。
•	modify：递归更新指定区间的值，支持懒惰标记。
•	query：递归查询指定区间的和，支持懒惰标记。
2. HLD 类
•	dfs1：第一次深度优先搜索，计算每个节点的深度、父节点、子树大小和重儿子。
•	dfs2：第二次深度优先搜索，进行重链剖分，分配节点的链顶和位置。
•	LCA：计算两个节点的最近公共祖先。
•	update_path：更新路径上的节点值，使用线段树的区间更新操作。
•	query_path：查询路径上的节点和，使用线段树的区间查询操作。
•	update_tree：更新子树上的节点值，使用线段树的区间更新操作。
•	query_tree：查询子树上的节点和，使用线段树的区间查询操作。
