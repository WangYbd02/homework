/// 三次样条插值函数构造

#pragma once
#include <iostream>
#include <string>
#include "Hermite_Interpolation.hpp"
#include "Elimination_with_Maximal_Column_Pivoting.hpp"


template <class T>
class CubicSplineInterpolation
{
private:
	int num; // 采样点数量
	T* x; // x
	T* f; // f(x)
	T* h; // h(i) = x(i+1) - x(i)
	T* lambda; // λ λ(i) = h(i+1) / (h(i+1)+h(i))
	T* miu; // μ μ(i) = h(i) / (h(i+1)+h(i))
	T* m;  //解向量m 即包含f'(x(i)) i = 0, 1, ..., num 的数组
	int cond;  //边界条件类型
	T cond1, cond2;  //边界条件
public:
	CubicSplineInterpolation(int num, T* x, T* f) {
		this->num = num;
		this->x = new T[num];
		this->f = new T[num];
		for (int i = 0; i < num; i++) {
			this->x[i] = *(x + i);
			this->f[i] = *(f + i);
		}
		this->h = new T[num - 1];
		for (int j = 0; j < num - 1; j++)
			this->h[j] = this->x[j + 1] - this->x[j];
		this->lambda = new T[num - 2];
		this->miu = new T[num - 2];
		for (int k = 0; k < num - 2; k++) {
			this->lambda[k] = this->h[k + 1] / (this->h[k + 1] + this->h[k]);
			this->miu[k] = this->h[k] / (this->h[k + 1] + this->h[k]);
		}
		this->m = NULL;
		// 若不进行边界条件类型的选择,则默认的边界条件为自然边界条件
		this->cond = 2;
		this->cond1 = this->cond2 = 0;
	}
	~CubicSplineInterpolation() {
		if (this->x != NULL)
			delete[]this->x;
		if (this->f != NULL)
			delete[]this->f;
		if (this->h != NULL)
			delete[]this->h;
		if (this->lambda != NULL)
			delete[]this->lambda;
		if (this->miu != NULL)
			delete[]this->miu;
		if (this->m != NULL)
			delete[]this->m;
		return;
	}

public:
	void selectBoundaryCondition(); // 选择边界条件
	void dsolve(); // 求解,得到解向量m
	void printm(); // 输出解向量 需要先调用dsolve()
	void printExpressions(); // 输出表达式 需要先调用dsolve()
	T calculation(T x); // 将x代入求得的插值多项式中,返回函数值 需要先调用dsolve()
};

// condi == "1"  第一边界条件 cond == 1
// condi == "2"  第二边界条件 cond == 2
// default  自然边界条件 cond == 2
// condi为其他值  无效选择,调用默认值
template<class T>
void CubicSplineInterpolation<T>::selectBoundaryCondition()
{
	std::string condi;
	std::cout << "please select the boundary condition:\n";
	std::cout << "1.First boundary condition\n2.Second boundary condition\n";
	std::cin.clear();
	std::cin.ignore();
	std::getline(std::cin, condi);
	if (condi == "1") {
		this->cond = 1;
		T value1, value2;
		std::cout << "f\'(x0)=";
		std::cin >> value1;
		std::cout << "f\'(x" << this->num - 1 << ")=";
		std::cin >> value2;
		this->cond1 = value1;
		this->cond2 = value2;
		return;
	} else if (condi == "2") {
		this->cond = 2;
		T value1, value2;
		std::cout << "f\"(x0)=";
		std::cin >> value1;
		std::cout << "f\"(x" << this->num - 1 << ")=";
		std::cin >> value2;
		this->cond1 = value1;
		this->cond2 = value2;
		return;
	} else {
		this->cond = 2;
		this->cond1 = this->cond2 = 0;
		return;
	}
}

template <class T>
void CubicSplineInterpolation<T>::dsolve()
{
	// 第一边界条件
	if (this->cond == 1) 
	{
		int size = this->num - 2; // 系数矩阵的大小
		// 初始化 A = zeros(size, size);
		T** A = new T * [size];  // n-1阶系数矩阵
		for (int i = 0; i < size; i++)
			A[i] = new T[size];
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				A[j][k] = 0;
			}
		}
		// 为A的第一行赋值
		A[0][0] = 2;
		A[0][1] = T(this->miu[0]);
		// 为A的最后一行赋值
		A[size - 1][size - 2] = this->lambda[size - 1];
		A[size - 1][size - 1] = 2;
		// 为A的其他行赋值
		for (int q = 1; q < size; q++) {
			A[q][q - 1] = this->lambda[q];
			A[q][q] = 2;
			A[q][q + 1] = this->miu[q];
		}

		// 初始化向量b并为其赋值
		T* b = new T[size];
		for (int t = 0; t < size; t++) {
			// 对应于课本的公式d(i) = μ(i)*(y(i+1)-y(i))/h(i) + λ(i)*(y(i)-y(i-1))/h(i-1)
			b[t] = (this->miu[t] * (this->f[t + 2] - this->f[t + 1]) / this->h[t + 1] \
				+ this->lambda[t] * (this->f[t + 1] - this->f[t]) / this->h[t]) * 3;
		}
		b[0] -= this->lambda[0] * cond1; // d(1) - λ(1)*f'(x0) 
		b[size - 1] -= this->miu[size - 1] * cond2; // d(n-1) - μ(n-1)*f'(xn)

		// 调用Elimination求解,得到解向量m
		Elimination<T> s(size, A, b);
		s.dsolve();
		T* p = s.getx();
		this->m = new T[this->num];
		this->m[0] = cond1;
		for (int z = 0; z < size; z++)
			this->m[z + 1] = *(p + z);
		this->m[size + 1] = cond2;

		// 释放内存
		for (int d = 0; d < size - 1; d++)
			delete[]A[d];
		delete[]A;
		delete[]b;

		return;
	}
	// 第二边界条件
	else if (this->cond == 2) 
	{
		int size = this->num; // 系数矩阵的大小
		// 初始化 A = zeros(size, size);
		T** A = new T * [size];  // n+1阶系数矩阵
		for (int i = 0; i < size; i++)
			A[i] = new T[size];
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				A[j][k] = 0;
			}
		}
		// 为A的第一行赋值
		A[0][0] = 2;
		A[0][1] = 1;
		// 为A的最后一行赋值
		A[size - 1][size - 2] = 1;
		A[size - 1][size - 1] = 2;
		// 为A的其他行赋值
		for (int q = 1; q < size - 1; q++) {
			A[q][q - 1] = this->lambda[q - 1];
			A[q][q] = 2;
			A[q][q + 1] = this->miu[q - 1];
		}

		// 初始化向量b并为其赋值
		T* b = new T[size];
		b[0] = 3 * (this->f[1] - this->f[0]) / this->h[0] - this->h[0] * this->cond1 / 2;
		for (int t = 0; t < size - 2; t++) {
			// 对应于课本的公式d(i) = μ(i)*(y(i+1)-y(i))/h(i) + λ(i)*(y(i)-y(i-1))/h(i-1)
			b[t + 1] = (this->miu[t] * (this->f[t + 2] - this->f[t + 1]) / this->h[t + 1] \
				+ this->lambda[t] * (this->f[t + 1] - this->f[t]) / this->h[t]) * 3;
		}
		b[size - 1] = 3 * (this->f[size - 1] - this->f[size - 2]) / this->h[size - 2] + this->h[size - 2] * this->cond2 / 2;

		// 调用Elimination求解m
		Elimination<T> s(size, A, b);
		s.dsolve();
		T* p = s.getx();
		this->m = new T[this->num];
		for (int z = 0; z < size; z++)
			this->m[z] = *(p + z);

		// 释放内存
		for (int d = 0; d < size; d++)
			delete[]A[d];
		delete[]A;
		delete[]b;

		return;
	}
}

template<class T>
void CubicSplineInterpolation<T>::printm()
{
	for (int i = 0; i < this->num; i++) {
		std::cout << *(this->m + i) << " ";
	}
	return;
}

template<class T>
void CubicSplineInterpolation<T>::printExpressions()
{
	for (int i = 0; i < this->num - 1; i++)
	{
		Hermite<T> hermite(x[i], x[i + 1], f[i], f[i + 1], m[i], m[i + 1]);
		hermite.dsolve();
		std::cout << "S" << i << "(x) = ";
		hermite.printExpression();
	}

	return;
}

template<class T>
T CubicSplineInterpolation<T>::calculation(T x)
{
	if (x < this->x[0] || x > this->x[this->num - 1]) {
		std::cout << "out of extension.\n";
		return 0;
	}
	int i = 0;
	while (x > this->x[i]) i++;
	Hermite<T> hermite(x[i - 1], x[i], f[i - 1], f[i], m[i - 1], m[i]);
	hermite.dsolve();
	T res = hermite.calculation(x);
	return res;
}