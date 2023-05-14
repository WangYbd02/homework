#include <iostream>

using namespace std;

template <class T>
class MaxHeap
{
private:
	T* heap;
	int Size;
	int Capacity;

public:
	MaxHeap(int N,T top)
	{
		heap = new T[N + 1];
		heap[0] = top;
		this->Size = 0;
		this->Capacity = N;
	}
	~MaxHeap() { delete[]heap; }

	bool Insert(T ele);
	void BuildHeap(T a[], int N);
	T GetTop();
};

template <class T>
bool MaxHeap<T>::Insert(T ele)
{
	if (this->Size == this->Capacity)
	{
		cout << "堆满" << endl;
		return false;
	}

	heap[++Size] = ele;
	if (this->Size == 1)
		return true;

	int i = Size;
	while (ele > heap[i / 2] && i > 1)
	{
		heap[i] = heap[i / 2];
		i = i / 2;
	}
	heap[i] = ele;
	return true;
}

template<class T>
void MaxHeap<T>::BuildHeap(T a[], int N)
{
	for (int i = 0; i < N; i++)
	{
		Insert(a[i]);
	}
	return;
}

template <class T>
T MaxHeap<T>::GetTop()
{
	if (this->Size == 0)
	{
		cout << "堆空" << endl;
		return heap[0];
	}
	
	T Top = heap[1];
	if (this->Size == 1)
	{
		this->Size--;
		return Top;
	}

	int i = 2;
	while (1)
	{
		if (i + 1 <= this->Size)
		{
			if (heap[i] < heap[i + 1])
				i++;
		}
		if (heap[Size] <= heap[i])
		{
			heap[i / 2] = heap[i];
			i = i * 2;
		}
		else
		{
			heap[i / 2] = heap[Size];
			break;
		}
		if (i > this->Size)
		{
			heap[i / 2] = heap[Size];
			break;
		}
	}
	this->Size--;

	return Top;
}


void Merge(int a[], int temp[], int L, int R, int R_End)
{
	int t = L;
	int L_End = R - 1;
	int num = R_End - L + 1;

	while (L <= L_End && R <= R_End)
	{
		if (a[L] <= a[R])
		{
			temp[t] = a[L];
			t++; L++;
		}
		else
		{
			temp[t] = a[R];
			t++; R++;
		}
	}

	while (L <= L_End)
	{
		temp[t++] = a[L++];
	}
	while (R <= R_End)
	{
		temp[t++] = a[R++];
	}

	for (int i = 0; i <= num; i++)
		a[R_End - i] = temp[R_End - i];

	return;
}

void Msort(int a[], int temp[], int L, int R_End)
{
	int s;
	if (L < R_End)
	{
		s = (L + R_End) / 2;
		Msort(a, temp, L, s);
		Msort(a, temp, s + 1, R_End);
		Merge(a, temp, L, s + 1, R_End);
	}
	return;
}


void MergeSort(int a[], int N)
{
	int* temp = new int[N];
	Msort(a, temp, 0, N - 1);
	delete[]temp;
	return;
}

void HeapSort(int a[], int N)
{
	MaxHeap<int> Maxheap(N, 0);
	Maxheap.BuildHeap(a, N);
	for (int i = 0; i < N; i++)
	{
		a[i] = Maxheap.GetTop();
	}
	return;
}

int main()
{
	int N;
	cin >> N;
	int* a = new int[N];
	int* b = new int[N];
	int* c = new int[N];

	for (int i = 0; i < N; i++)
	{
		cin >> a[i];
	}
	for (int j = 0; j < N; j++)
	{
		c[j] = b[j] = a[j];
	}

	MergeSort(b, N);
	for (int x = 0; x < N;x++)
	{
		cout << b[x];
		if (x != N - 1)
			cout << " ";
	}
	cout << endl;

	HeapSort(c, N);
	for (int y = 0; y < N; y++)
	{
		cout << c[y];
		if (y != N - 1)
			cout << " ";
	}

	return 0;
}