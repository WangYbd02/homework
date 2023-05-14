#include <iostream>

using namespace std;


template <class T>
class Node
{
public:

	T data;
	Node* next;
};

template <class T>
class Queue
{
private:

	Node<T>* front; //front指向队列的第一个元素
	Node<T>* rear;  //rear指向队列的最后一个元素

public:

	Queue()
	{
		front = new Node<T>;
		rear = new Node<T>;
		front = rear = NULL;
	}
	~Queue()
	{
		this->DeleteAll();
	}

	bool IfEmpty();  //判断队列是否为空
	int QueueLen();  //返回队列长度
	bool Enqueue(T Ele);  //入队
	bool Dequeue();  //出队
	void Show();  //输出队列中所有元素
	void DeleteAll();  //清空队列
};

template <class T>
bool Queue<T>::IfEmpty()
{
	if (front == NULL)
		return false;

	return true;
}

template <class T>
int Queue<T>::QueueLen()
{
	Node<T>* p = new Node<T>;
	int cpt = 0;
	p = front;

	while (p !=	NULL)
	{
		cpt++;
		p = p->next;
	}

	delete p;
	return cpt;
}

template <class T>
bool Queue<T>::Enqueue(T Ele)  //error
{
	Node<T>* s = new Node<T>;
	s->data = Ele;
	s->next = NULL;
	if (IfEmpty() == 0)
	{
		front = s;
		rear = s;
	}
	else
	{
		rear->next = s;
		rear = s;
	}
	return true;
}

template <class T>
bool Queue<T>::Dequeue()
{
	if (!IfEmpty())
	{
		cout << "队列为空" << endl;
		return false;
	}
	else
	{
		Node<T>* s = new Node<T>;
		s = front;
		front = front->next;
		delete s;
		return true;
	}
}

template <class T>
void Queue<T>::Show()
{
	Node<T>* p = new Node<T>;
	p = front;

	while (p != NULL)
	{
		cout << p->data;
		if (p->next != NULL)
			cout << " ";

		p = p->next;
	}
	delete p;
}

template <class T>
void Queue<T>::DeleteAll()
{
	if (IfEmpty() == 0)
		return;

	Node<T>* p = new Node<T>;
	p = front;

	while (p != NULL)
	{
		Node<T>* s = new Node<T>;
		s = p;
		p = p->next;
		delete s;
	}

	delete p;
	return;
}


void IM1()
{
	int N;
	int k;
	Queue<int>queue;
	cin >> N;
	for (int i = 0; i < N; i++)
	{
		cin >> k;
		queue.Enqueue(k);
	}

	queue.Dequeue();
	queue.Dequeue();
	queue.Enqueue(11);
	queue.Enqueue(12);

	queue.Show();

	return;
}

int main()
{
	IM1();

	return 0;
}
