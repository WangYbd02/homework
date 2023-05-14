//Huffman_Tree 1.0
//默认节点权值为非负整数

#include <iostream>

using namespace std;


typedef int Type;  //节点权值类型


class Node  //节点
{
public:
	Type weight;
	Node* left;
	Node* right;
};


class MinHeap;  //小顶堆的声明(友元类)

class Stack;  //栈的声明


class Huffman_Tree
{
	friend class MinHeap;
private:
	Node* root;

public:

	Huffman_Tree()
	{
		root = new Node;
		this->root->weight = -1;
		this->root->left = this->root->right = NULL;
	}
	Huffman_Tree(Type weightt)
	{
		root = new Node;
		this->root->weight = weightt;
		this->root->left = this->root->right = NULL;
	}

	Node* operator+(Huffman_Tree& t1)  //重载+运算符便于后续操作
	{
		Node* pn = new Node;
		pn->left = this->root;
		pn->right = t1.root;
		pn->weight = this->root->weight + t1.root->weight;

		return pn;
	}

	MinHeap WeightHeap(Type weightt[], Type Sentry, int N);  //依据根节点权值构造小顶堆
	Huffman_Tree& Construct(MinHeap Minheap);  //构造哈夫曼树
	Node* GetRoot();  //返回根节点的指针
	void WPL(Node* n, Type& sum, int len);  //计算WPL
	void PreorderTraversal(Node* n);  //先序遍历
	void Huffman_Code(Node* n, Stack* stack);  //哈夫曼编码

};


class MinHeap
{
	friend class Huffman_Tree;
private:
	Huffman_Tree* heap;
	int Size;
	int Capacity;

public:

	MinHeap(int Capacity, Type Sentry)
	{
		this->Capacity = Capacity;
		this->Size = 0;
		heap = new Huffman_Tree[Capacity + 1];
		this->heap[0].root->weight = Sentry;
		for (int i = 0; i <= Capacity; i++)
		{
			this->heap[i].root->left = this->heap[i].root->right = NULL;
		}
	}
	~MinHeap() {};

	int GetSize();  //返回堆中元素数量
	int GetCapacity();  //返回堆容量
	bool Insert(Huffman_Tree t);  //插入元素
	Huffman_Tree& GetTop();  //返回堆顶元素
	void DeleteHeap();  //清空堆中元素
};


class Stack
{
private:
	int* stack;
	int Size;
	int Capacity;
public:
	Stack(int N)
	{
		stack = new int[N];
		for (int i = 0; i < N; i++)
		{
			stack[i] = -1;
		}
		this->Capacity = N;
		this->Size = 0;
	}
	~Stack()
	{
		this->DeleteStack();
	}

	int GetSize();  //返回栈中元素的数量
	int GetCapacity();  //返回栈的容量
	bool Push(int num);  //插入元素
	bool Pop();  //返回元素
	void Output();  //输出栈中所有元素
	void DeleteStack();  //清空栈中元素
};

int Stack::GetSize()
{
	return this->Size;
}

int Stack::GetCapacity()
{
	return this->Capacity;
}

bool Stack::Push(int num)
{
	if (this->Size == this->Capacity)
	{
		cout << "栈满" << endl;
		return false;
	}

	stack[Size] = num;
	Size++;
	return true;
}

bool Stack::Pop()
{
	if (this->Size == 0)
	{
		cout << "栈空" << endl;
		return false;
	}

	stack[Size] = -1;
	Size--;
	return true;
}

void Stack::Output()
{
	for (int i = 0; stack[i] != -1; i++)
	{
		cout << stack[i];
	}
}

void Stack::DeleteStack()
{
	if (stack == NULL)
		return;

	delete[]stack;
	stack = NULL;
	return;
}

int MinHeap::GetSize()
{
	return this->Size;
}

int MinHeap::GetCapacity()
{
	return this->Capacity;
}

bool MinHeap::Insert(Huffman_Tree t)
{
	if (this->Size == this->Capacity)
	{
		cout << "堆满" << endl;
		return false;
	}

	heap[++Size] = t;  //尾插
	if (this->Size == 1)  //仅有一个元素时，不需要进行调整
		return true;

	int i = Size / 2;
	int j = Size;
	while (t.root->weight < heap[i].root->weight)  //插入元素后进行调整，使该结构仍为堆
	{
		heap[j] = heap[i];
		j = i;
		i = i / 2;
	}
	heap[j] = t;
	return true;
}

Huffman_Tree& MinHeap::GetTop()
{
	if (this->Size == 0)
	{
		Huffman_Tree fail_tree;
		cout << "堆空" << endl;
		return fail_tree;
	}

	Huffman_Tree Top;
	Top = heap[1];
	if (this->Size == 1)  //只有一个元素时，不需要调整
	{
		this->Size--;
		return Top;
	}

	int i = 2;
	while (1)
	{
		if (i + 1 <= this->Size)  //判断heap[i/2]有无右孩子
		{
			if (heap[i].root->weight > heap[i + 1].root->weight)
				i++;
		}

		if (heap[Size].root->weight > heap[i].root->weight)
			heap[i / 2] = heap[i];
		else
		{
			heap[i / 2] = heap[Size];
			break;
		}
		i = i * 2;
		if (i > this->Size)
		{
			heap[i / 2] = heap[Size];
			break;
		}
	}
	this->Size--;

	return Top;
}

void MinHeap::DeleteHeap()
{
	if (heap == NULL)
		return;

	delete[]heap;
	heap = NULL;
	return;
}

MinHeap Huffman_Tree::WeightHeap(Type weightt[], Type Sentry, int N)
{
	MinHeap Minheap(N, Sentry);
	Huffman_Tree* pTree = new Huffman_Tree[N];
	for (int i = 0; i < N; i++)
	{
		pTree[i].root->weight = weightt[i];
		pTree[i].root->left = pTree[i].root->right = NULL;
	}
	for (int j = 0; j < N; j++)
	{
		Minheap.Insert(pTree[j]);
	}

	delete[]pTree;

	return Minheap;
}

Huffman_Tree& Huffman_Tree::Construct(MinHeap Minheap)
{
	Huffman_Tree* pTree;
	Huffman_Tree t1, t2;
	int size = Minheap.GetSize() - 1;
	for (int i = 0; i < size; i++)
	{
		pTree = new Huffman_Tree;
		t1 = Minheap.GetTop();
		t2 = Minheap.GetTop();
		pTree->root = t1 + t2;
		Minheap.Insert(*pTree);
	}

	Huffman_Tree Finished_Tree = Minheap.GetTop();

	Minheap.DeleteHeap();
	return Finished_Tree;
}

Node* Huffman_Tree::GetRoot()
{
	return this->root;
}

void Huffman_Tree::WPL(Node* n, Type& sum, int len)
{
	if (n != NULL)
	{
		if (n->left == NULL && n->right == NULL)
			sum = sum + len * n->weight;
		if (n->left != NULL)
			WPL(n->left, sum, len + 1);
		if (n->right != NULL)
			WPL(n->right, sum, len + 1);
	}
	return;
}

void Huffman_Tree::PreorderTraversal(Node* n)
{
	if (n != NULL)
	{
		cout << n->weight;
		if (n->left != NULL || n->right != NULL)
			cout << "(";
		if (n->left != NULL)
		{
			PreorderTraversal(n->left);
			if (n->right != NULL)
				cout << ",";
			else
				cout << ")";
		}
		if (n->right != NULL)
		{
			PreorderTraversal(n->right);
			cout << ")";
		}
	}
	return;
}

void Huffman_Tree::Huffman_Code(Node* n, Stack* stack)
{
	if (n != NULL)
	{
		if (n->left == NULL && n->right == NULL)
		{
			stack->Output();
			cout << " ";
		}
		if (n->left != NULL)
		{
			stack->Push(0);
			Huffman_Code(n->left, stack);
		}
		if (n->right != NULL)
		{
			stack->Push(1);
			Huffman_Code(n->right, stack);
		}
	}
	if (stack->GetSize() != 0)
		stack->Pop();

	return;
}



void IM1()
{

	int N;  //节点数量
	cout << "输入节点数量:\n";
	cin >> N;
	Type* weight = new Type[N];
	Type weightt = 0;
	cout << "输入各节点权值:\n";
	for (int i = 0; i < N; i++)
	{
		cin >> weightt;  //输入节点权值
		weight[i] = weightt;
	}

	//构造哈夫曼树
	Huffman_Tree t;
	MinHeap Minheap = t.WeightHeap(weight, -1, N);
	t = t.Construct(Minheap);

	//计算WPL
	Type sum = 0;
	int len = 0;
	t.WPL(t.GetRoot(), sum, len);
	cout << sum << endl;

	//先序遍历
	t.PreorderTraversal(t.GetRoot());
	cout << endl;

	//哈夫曼编码
	Stack stack(N);
	Stack* p_stack = &stack;
	t.Huffman_Code(t.GetRoot(), p_stack);

	return;
}


int main()
{
	IM1();

	return 0;
}
