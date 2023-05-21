/// 埃米尔特插值多项式构造

#pragma once
#include <iostream>


template <class T>
class Hermite
{
private:
	T x0, x1; // x
	T f0, f1; // f(x)
	T df0, df1; // f'(x)
	T* a; // 各项的系数 a[i]为x的i次方项的系数
public:
	Hermite(T x0, T x1, T f0, T f1, T df0, T df1) {
		this->x0 = x0;
		this->x1 = x1;
		this->f0 = f0;
		this->f1 = f1;
		this->df0 = df0;
		this->df1 = df1;
		this->a = NULL;
	}
	~Hermite() {
		if (this->a != NULL)
			delete[]this->a;
		return;
	}

public:
	T pow(T x, int a); // pow(x, a) = x^a
	void dsolve(); // 求解,将插值多项式中各项的系数保存到动态数组a中
	T calculation(T x); // 将x代入求得的插值多项式中,返回函数值 需要先调用dsolve()
	void printExpression(); // 输出求得的插值多项式 需要先调用dsolve()
};

template<class T>
T Hermite<T>::pow(T x, int a)
{
	T m = x;
	m = m / x;
	for (int i = 0; i < a; i++)
		m = m * x;
	return m;
}

template<class T>
void Hermite<T>::dsolve()
{
	this->a = new T[4];
	a[0] = f0 * pow(x1, 2) * (x1 - 3 * x0) / pow(x1 - x0, 3) \
		+ f1 * pow(x0, 2) * (x0 - 3 * x1) / pow(x0 - x1, 3) \
		- (df0 * x1 + df1 * x0) * x0 * x1 / pow(x0 - x1, 2);
	a[1] = 6 * x0 * x1 * (f0 - f1) / pow(x1 - x0, 3) \
		+ (df0 * x1 * (x1 + 2 * x0) + df1 * x0 * (x0 + 2 * x1)) / pow(x1 - x0, 2);
	a[2] = -3 * (x0 + x1) * (f0 - f1) / pow(x1 - x0, 3) \
		- (df0 * (x0 + 2 * x1) + df1 * (x1 + 2 * x0)) / pow(x1 - x0, 2);
	a[3] = 2 * (f0 - f1) / pow(x1 - x0, 3) + (df0 + df1) / pow(x1 - x0, 2);
	return;
}

template<class T>
T Hermite<T>::calculation(T x)
{
	return a[3] * pow(x, 3) + a[2] * pow(x, 2) + a[1] * x + a[0];
}

template<class T>
void Hermite<T>::printExpression()
{
	if (a[3] != 0)
		std::cout << a[3] << "x^3";
	if (a[2] > 0)
		std::cout << "+" << a[2] << "x^2";
	else if (a[2] < 0)
		std::cout << a[2] << "x^2";
	if (a[1] > 0)
		std::cout << "+" << a[1] << "x";
	else if (a[1] < 0)
		std::cout << a[1] << "x";
	if (a[0] > 0)
		std::cout << "+" << a[0];
	else if (a[0] < 0)
		std::cout << a[0];
	std::cout << "  x∈[" << x0 << ", " << x1 << "]\n";
}