//多项式的输出形式为：
//	该多项式共有k项.
//	(co_0)x(ex_0) + ... + (co_(k-1))x(ex_(k-1))
//其中，co_i为系数，ex_i为指数，且对于任意 i < j ，都有ex_i > ex_j

//只能支持指数为正整数的情况，系数可通过修改T以支持double,float,int类型


#include <iostream>
#include <string>

using namespace std;


template <class T>  //幂函数(该函数限制了多项式的项的指数只能取正整数)
T pow(T a, int b)
{
	T s = a;
	for (int i = 1; i < b; i++)
	{
		s = s * a;
	}

	return s;
}



//多项式的项
template <class T>
class Term
{
public:
	T Coefficient;  //系数
	int Exponent;  //指数
	Term* next;  //指向多项式的下一项

	Term() {};
	Term(T co, int ex)
	{
		this->Coefficient = co;
		this->Exponent = ex;
		this->next = NULL;
	}
};


//多项式
template <class T>
class Polynomial
{
private:
	Term<T>* head;  //头指针
	Term<T>* rear;  //尾指针，指向最后一项，便于一些操作的进行

public:

	Polynomial()
	{
		head = new Term<T>;
		head->Exponent = 0;  //指示多项式的项数
		head->next = rear = NULL;
	}

	Polynomial& operator+(Polynomial& Po1);  //重载+运算符实现多项式相加
	Polynomial& operator-(Polynomial& Po1);  //重载-运算符实现多项式相减

	void Insert(T co, int ex);  //插入项
	Term<T>& GetMax();  //返回最高次项
	Term<T>& GetMin();  //返回最低次项


	void Input(T Co[], int Ex[], int N);  //输入多项式
	void Output();  //输出多项式
	Polynomial& Plus(Polynomial& Po1, Polynomial& Po2);  //多项式相加
	Polynomial& Minus(Polynomial& Po1, Polynomial& Po2);  //多项式相减
	T Assign(T t);  //计算多项式在点x = t 处的值
};


template <class T>
Polynomial<T>& Polynomial<T>::operator+(Polynomial& Po1)
{
	Polynomial<T> Po;
	Term<T>* p1 = new Term<T>;  //指向第一个多项式的项
	Term<T>* p2 = new Term<T>;  //指向第二个多项式的项
	p1 = this->head->next;
	p2 = Po1.head->next;
	
	while (p1 != NULL && p2 != NULL)
	{
		if (p1->Exponent > p2->Exponent)
		{
			Po.Insert(p1->Coefficient, p1->Exponent);
			p1 = p1->next;
		}

		else if (p1->Exponent == p2->Exponent)
		{
			if ((p1->Coefficient) + (p2->Coefficient) != 0)  //同次项相加为0则删除
				Po.Insert((p1->Coefficient) + (p2->Coefficient), p1->Exponent);

			p1 = p1->next;
			p2 = p2->next;
		}

		else
		{
			Po.Insert(p2->Coefficient, p2->Exponent);
			p2 = p2->next;
		}
	}

	if (p1 == NULL)
	{
		while (p2 != NULL)
		{
			Po.Insert(p2->Coefficient, p2->Exponent);
			p2 = p2->next;
		}
	}

	else if (p2 == NULL)
	{
		while (p1 != NULL)
		{
			Po.Insert(p1->Coefficient, p1->Exponent);
			p1 = p1->next;
		}
	}

	delete p1;
	delete p2;

	return Po;
}


template <class T>
Polynomial<T>& Polynomial<T>::operator-(Polynomial& Po1)
{
	Polynomial<T> Po;
	Term<T>* p1 = new Term<T>;  //指向第一个多项式的项
	Term<T>* p2 = new Term<T>;  //指向第二个多项式的项
	p1 = this->head->next;
	p2 = Po1.head->next;

	while (p1 != NULL && p2 != NULL)
	{
		if (p1->Exponent > p2->Exponent)
		{
			Po.Insert(p1->Coefficient, p1->Exponent);
			p1 = p1->next;
		}

		else if (p1->Exponent == p2->Exponent)
		{
			if ((p1->Coefficient) - (p2->Coefficient) != 0)  //同次项相减为0则删除
				Po.Insert((p1->Coefficient) - (p2->Coefficient), p1->Exponent);

			p1 = p1->next;
			p2 = p2->next;
		}

		else
		{
			Po.Insert(-(p2->Coefficient), p2->Exponent);
			p2 = p2->next;
		}
	}

	if (p1 == NULL)
	{
		while (p2 != NULL)
		{
			Po.Insert(-(p2->Coefficient), p2->Exponent);
			p2 = p2->next;
		}
	}

	else if (p2 == NULL)
	{
		while (p1 != NULL)
		{
			Po.Insert(p1->Coefficient, p1->Exponent);
			p1 = p1->next;
		}
	}

	delete p1;
	delete p2;

	return Po;
}


template <class T>
void Polynomial<T>::Insert(T co, int ex)
{
	if (co == 0)
		return;

	Term<T>* s = new Term<T>;
	s->Coefficient = co;
	s->Exponent = ex;
	s->next = NULL;

	if (this->head->next == NULL)  //多项式为空时，直接插入
	{
		this->head->next = rear = s;
		this->head->Exponent++;
		return;
	}

	Term<T>* p = new Term<T>;
	p = this->head->next;
	if (ex > p->Exponent)  //新插入元素的指数比原多项式第一项的指数大
	{
		s->next = p;
		this->head->next = s;
		this->head->Exponent++;
		return;
	}

	if (ex < this->rear->Exponent)  //新插入元素的指数比原多项式最后一项指数小
	{
		rear->next = s;
		rear = s;
		this->head->Exponent++;
		return;
	}

	while (p != NULL)
	{
		if (ex == p->Exponent)
		{
			p->Coefficient = p->Coefficient + co;
			delete s;
			return;
		}
		else if (ex<p->Exponent && ex>p->next->Exponent)
		{
			s->next = p->next;
			p->next = s;
			this->head->Exponent++;
			return;
		}
		else
		{
			p = p->next;
		}
	}

}


template <class T>
Term<T>& Polynomial<T>::GetMax()
{
	Term<T> t;
	t.Coefficient = this->head->Coefficient;
	t.Exponent = this->head->Exponent;
	t.next = this->head->next;
	return t;
}

template <class T>
Term<T>& Polynomial<T>::GetMin()
{
	Term<T> t;
	t.Coefficient = this->rear->Coefficient;
	t.Exponent = this->rear->Exponent;
	t.next = this->rear->next;
	return t;
}

template <class T>
void Polynomial<T>::Input(T Co[], int Ex[], int N)
{
	for (int i = 0; i < N; i++)
	{
		Insert(Co[i], Ex[i]);
	}
}


template <class T>
void Polynomial<T>::Output()
{
	int k = this->head->Exponent;
	if (k == 0)
	{
		cout << "多项式为0" << endl;
		return;
	}

	Term<T>* p = new Term<T>;
	p = this->head->next;

	cout << "该多项式共有" << k << "项." << endl;
	while (p != NULL)
	{
		cout << p->Coefficient << "x(" << p->Exponent << ")";
		p = p->next;
		if (p != NULL)
		{
			if (p->Coefficient > 0)
				cout << "+";
		}
	}
	cout << endl;

	return;
}


template <class T>
Polynomial<T>& Polynomial<T>::Plus(Polynomial<T>& Po1, Polynomial<T>& Po2)
{
	Polynomial<T> Po;
	Po = Po1 + Po2;
	return Po;
}

template <class T>
Polynomial<T>& Polynomial<T>::Minus(Polynomial<T>& Po1, Polynomial<T>& Po2)
{
	Polynomial<T> Po;
	Po = Po1 - Po2;
	return Po;
}


template <class T>
T Polynomial<T>::Assign(T t)
{
	T sum = t;
	sum = sum - t;
	Term<T>* p = new Term<T>;
	p = this->head->next;

	while (p != NULL)
	{
		sum = sum + (p->Coefficient) * pow(t, (p->Exponent));
		p = p->next;
	}

	return sum;
}

void IM1()
{
	int N1;
	cout << "输入第一个多项式:\n项数: ";
	cin >> N1;

	int* Co1 = new int[N1];
	int* Ex1 = new int[N1];

	for (int i = 0; i < N1; i++)
	{
		cin >> Co1[i];
		cin >> Ex1[i];
	}

	Polynomial<int> Po1;
	Po1.Input(Co1, Ex1, N1);
	cout << "第一个多项式:\n";
	Po1.Output();


	int N2;
	cout << "输入第二个多项式:\n项数: ";
	cin >> N2;

	int* Co2 = new int[N2];
	int* Ex2 = new int[N2];

	for (int j = 0; j < N2; j++)
	{
		cin >> Co2[j];
		cin >> Ex2[j];
	}

	Polynomial<int> Po2;
	Po2.Input(Co2, Ex2, N2);
	cout << "第二个多项式:\n";
	Po2.Output();


	Polynomial<int> Po_sum;

	//两种方式均可实现多项式相加
//	Po_sum = Po1 + Po2;  
	Po_sum = Po_sum.Plus(Po1, Po2);

	cout << "两个多项式相加为:\n";
	Po_sum.Output();


	Polynomial<int> Po_differ;

	//两种方式均可实现多项式相减
	Po_differ = Po1 - Po2;
//	Po_differ = Po_differ.Minus(Po1, Po2);

	cout << "两个多项式相减为:\n";
	Po_differ.Output();


	int t;
	cout << "为多项式赋值:\n	x = ";
	cin >> t;

	cout << "第一个多项式的值为: ";
	cout << Po1.Assign(t) << endl;

	cout << "第二个多项式的值为: ";
	cout << Po2.Assign(t) << endl;

	cout << "两个多项式相加的值为: ";
	cout << Po_sum.Assign(t) << endl;

	cout << "两个多项式相减的值为:  ";
	cout << Po_differ.Assign(t) << endl;


	delete[]Co1;
	delete[]Ex1;
	delete[]Co2;
	delete[]Ex2;

	return;
}



int main()
{
	IM1();

	return 0;
}