#include <iostream>
#include <string>

using namespace std;


template <class T>
class Node
{
public:
	T data;
	Node* next;

	Node() {};
	Node(T& a)
	{
		this->data = a;
	}
};

template <class T>
class Stack:public Node<T>
{
private:
	
	Node<T>* top;  //用top->next指向栈顶元素

public:

	Stack()
	{
		top = new Node<T>;
		top->next = NULL;
	}

	bool IfEmpty();  //判断栈是否为空
	int StackSize();  //返回栈的长度
	bool Push(T node);  //入栈
	bool Pop();  //出栈
	T& GetTop();  //返回栈顶元素
	void Show();  //输出栈中所有元素
	void DeleteAll();  //清空栈中所有元素

	~Stack()
	{
		if (top->next != NULL)
		{
			this->DeleteAll();
		}
		delete top;
	}
};


template <class T>
bool Stack<T>::IfEmpty()
{
	if (top->next == NULL)
		return false;

	return true;
}

template <class T>
int Stack<T>::StackSize()
{
	Node<T>* p = new Node<T>;
	p = top->next;
	int cpt = 0;

	while (p)
	{
		p = p->next;
		cpt++;
	}
	delete p;

	return cpt;
}

template <class T>
bool Stack<T>::Push(T node)
{
	Node<T>* s = new Node<T>;

	if (!IfEmpty())
	{
		s->data = node;
		s->next = NULL;
		top->next = s;
		return true;
	}

	s->data = node;
	s->next = top->next;
	top->next = s;
	return true;
}

template <class T>
bool Stack<T>::Pop()
{
	if (!IfEmpty())
		return false;

	Node<T>* s = new Node<T>;
	s = top->next;
	top->next = top->next->next;
	delete s;
	return true;
}

template <class T>
T& Stack<T>::GetTop()
{
	if (!IfEmpty())
	{
		cout << "栈为空" << endl;
		return;
	}
	return top->next->data;
}

template <class T>
void Stack<T>::Show()
{
	Node<T>* p = new Node<T>;
	p = top->next;
	while (p)
	{
		cout << p->data << endl;
		p = p->next;
	}
	delete p;
	return;
}

template <class T>
void Stack<T>::DeleteAll()
{
	Node<T>* p = new Node<T>;
	p = top->next;
	while (p)
	{
		Node<T>* s = new Node<T>;
		s = p;
		p = p->next;
		delete s;
	}
	delete p;
}

void IM1()
{
	Stack<char> stack;
	string str1;
	int flag = 1;
	getline(cin, str1);
	for (int i = 0; i < str1.size(); i++)
	{
		if (str1[i] == '(')
			stack.Push(str1[i]);

		if (str1[i] == ')')
		{
			flag = stack.Pop();
			if (flag == 0)
			{
				cout << "括号不匹配！" << endl;
				return;
			}
		}
	}
	if (stack.IfEmpty() != 0)
	{
		cout << "括号不匹配！" << endl;
		return;
	}
	cout << "括号匹配！" << endl;
	return;
}


int main()
{
	IM1();

	return 0;
}