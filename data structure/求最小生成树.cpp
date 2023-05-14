#include <iostream>

using namespace std;

//顶点下标从1开始


#define MAXWEIGHT 10001


template <class T>
class Edge
{
public:
	int u;
	int v;
	T weight;
};


template <class T>
class Graph;

template <class T>
class MinHeap
{
	friend class Graph<T>;
private:
	Edge<T>* heap;
	int Size;
	int Capacity;
public:

	MinHeap(int Capacity, T Sentry)
	{
		this->Capacity = Capacity;
		this->Size = 0;
		heap = new Edge<T>[Capacity + 1];
		this->heap[0].weight = Sentry;
		for (int i = 0; i <= Capacity; i++)
		{
			heap[i].u = heap[i].v = 0;
		}
	}

	int GetSize();
	int GetCapacity();
	bool Insert(int u1, int v1, T weightt);
	Edge<T>& GetTop();
	void DeleteHeap();
};

template <class T>
int MinHeap<T>::GetSize()
{
	return this->Size;
}

template <class T>
int MinHeap<T>::GetCapacity()
{
	return this->Capacity;
}

template <class T>
bool MinHeap<T>::Insert(int u1, int v1, T weightt)
{
	if (this->Size == this->Capacity)
	{
		cout << "堆满" << endl;
		return false;
	}

	Edge<T> e;
	e.u = u1;
	e.v = v1;
	e.weight = weightt;

	heap[++Size] = e;  //尾插
	if (this->Size == 1)  //仅有一个元素时，不需要进行调整
		return true;

	int i = Size / 2;
	int j = Size;
	while (e.weight < heap[i].weight)  //插入元素后进行调整，使该结构仍为堆
	{
		heap[j] = heap[i];
		j = i;
		i = i / 2;
	}
	heap[j] = e;
	return true;
}

template <class T>
Edge<T>& MinHeap<T>::GetTop()
{
	if (this->Size == 0)
	{
		Edge<T> fail_e;
		cout << "堆空" << endl;
		return fail_e;
	}

	Edge<T> Top;
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
			if (heap[i].weight > heap[i + 1].weight)
				i++;
		}

		if (heap[Size].weight > heap[i].weight)
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

template <class T>
void MinHeap<T>::DeleteHeap()
{
	if (heap == NULL)
		return;

	delete[]heap;
	heap = NULL;
	return;
}


template <class T>
class Graph
{
private:

	int Size;
	T** matrix;

public:

	Graph(int Size)
	{
		matrix = new T*[Size];
		for (int i = 0; i < Size; i++)
		{
			matrix[i] = new T[Size];
		}
		this->Size = Size;
		for (int j = 0; j < Size; j++)
		{
			for (int k = 0; k < Size; k++)
			{
				matrix[j][k] = MAXWEIGHT;
			}
		}
	}
	~Graph()
	{
		for (int i = 0; i < this->Size; i++)
		{
			delete[]matrix[i];
		}
		delete[]matrix;
	}


	bool Insert(int u, int v, T weight);  //插入边
	bool ChangeWeight(int u, int v, T weight);  //修改已存在边的权值
	T ShowWeight(int u, int v); //查看已存在边的权值
	Edge<T>* MinTree_Edge(int N);  //求一棵最小生成树
	T MinWeight(Edge<T> edge[], int N);  //最小生成树的权值
};

template <class T>
bool Graph<T>::Insert(int u, int v, T weight)
{
	if (u > this->Size || v > this->Size)
	{
		cout << "越界" << endl;
		return false;
	}
	if (matrix[u - 1][v - 1] != MAXWEIGHT)
	{
		cout << "边已存在" << endl;
		return false;
	}

	matrix[u - 1][v - 1] = weight;
	matrix[v - 1][u - 1] = weight;
	return true;
}

template <class T>
bool Graph<T>::ChangeWeight(int u, int v, T weight)
{
	if (matrix[u - 1][v - 1] != MAXWEIGHT)
	{
		cout << "边不存在" << endl;
		return false;
	}

	matrix[u - 1][v - 1] = weight;
	matrix[u - 1][v - 1] = weight;
	return true;
}

template <class T>
T Graph<T>::ShowWeight(int u, int v)
{
	return matrix[u - 1][v - 1];
}

template <class T>
Edge<T>* Graph<T>::MinTree_Edge(int N)
{
	Edge<T>* edge = new Edge<T>[N - 1];
	MinHeap<T>* Minheap = new MinHeap<T>(N * (N - 1) / 2 + 1, -1);
	Edge<T> e;
	int* visited = new int[N];
	for (int x = 0; x < N; x++)
	{
		visited[x] = x;
	}
	int i = 0;
	int j = 0;
	int t = 0;
	int k = 0;
	int m = 0;
	int h = 0;
	int e_v = 0;

	for (i = 0; i < N; i++)
	{
		for (j = i + 1; j < N; j++)
		{
			if (matrix[i][j] != MAXWEIGHT)
			{
				Minheap->Insert(i, j, matrix[i][j]);
				k++;
			}
			else
				continue;
		}
	}
	for (m = 0; m < k; m++)
	{
		e = Minheap->GetTop();
		if (visited[e.u] == visited[e.v])
		{
			continue;
		}
		edge[t] = e;
		e_v = visited[e.v];
		for (h = 0; h < N; h++)
		{
			if (visited[h] == e_v)
				visited[h] = visited[e.u];
		}
		t++;
		if (t == N - 1)
			break;
	}

	return edge;

}

template <class T>
T Graph<T>::MinWeight(Edge<T> edge[], int N)
{
	T min = 0;
	for (int i = 0; i < N - 1; i++)
	{
		min = min + edge[i].weight;
	}

	return min;
}



void IM1()
{
	int N;
	int M;
	cin >> N;
	cin >> M;

	//初始化一个N阶空图
	Graph<int> graph(N);
	
	//建图
	int i = 0;
	int flag = 1;
	int u = 0;
	int v = 0;
	int weight = MAXWEIGHT;
	while (i < M)
	{
		cin >> u;
		cin >> v;
		cin >> weight;
		flag = graph.Insert(u, v, weight);
		if (flag == 1)
			i++;
	}

	Edge<int>* edge = graph.MinTree_Edge(N);
	int min = graph.MinWeight(edge, N);

	cout << min << endl;

	return;
}

int main()
{
	IM1();

	return 0;
}