/// 列主元高斯消去法

#pragma once
#include <iostream>


template <class T>
class Elimination
{
private:
	int size; // 方阵的大小
	T** A_b; // 增广矩阵
	T* x; // 解向量
public:
	Elimination(int size, T** A, T* b) {
		this->size = size;
		this->A_b = new T * [size];
		this->x = NULL;
		for (int i = 0; i < size; i++) {
			A_b[i] = new T[size + 1];
		}
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				A_b[j][k] = *(*(A + j) + k);
			}
		}
		for (int m = 0; m < size; m++) {
			A_b[m][size] = *(b + m);
		}
	}
	~Elimination() {
		if (this->A_b != NULL) {
			for (int i = 0; i < size; i++) {
				delete[]this->A_b[i];
			}
			delete[]this->A_b;
		}
		if (this->x != NULL)
			delete[]this->x;
		return;
	}

public:
	void dsolve(); // 求解
	/*
	* 如果出现下述情况:某一列(除了最后两列)在消元操作前,从当前行到最后一行全为0,(即给定方阵的行列式为0),函数会崩溃 
	*/
	T* getx(); // 返回解向量 需要先调用dsolve()
	void printx(); // 输出解向量 需要先调用dsolve()
};

template <class T>
void Elimination<T>::dsolve()
{
	T** a_b; // 增广矩阵A_b的拷贝
	int size = this->size;
	a_b = new T * [size];
	for (int i1 = 0; i1 < size; i1++) {
		a_b[i1] = new T[size + 1];
	}
	for (int i2 = 0; i2 < size; i2++) {
		for (int i3 = 0; i3 <= size; i3++) {
			a_b[i2][i3] = A_b[i2][i3];
		}
	}

	// 获得行阶梯形矩阵
	int max_row = 0;
	T* temp = NULL;
	T beta;
	for (int i4 = 0; i4 < size - 1; i4++)
	{
		max_row = i4; // 记录当前列中绝对值最大的数所在行
		for (int i5 = i4 + 1; i5 < size; i5++) { // 找到当前列中绝对值最大的数所在行
			if (abs(a_b[i5][i4]) > abs(a_b[max_row][i4]))
				max_row = i5;
		}
		if (max_row != i4) { // 交换行
			temp = a_b[i4];
			a_b[i4] = a_b[max_row];
			a_b[max_row] = temp;
		}
		for (int i6 = i4 + 1; i6 < size; i6++) { //得到行阶梯形矩阵
			beta = a_b[i6][i4];
			for (int i7 = i4; i7 <= size; i7++) {
				a_b[i6][i7] -= beta * a_b[i4][i7] / a_b[i4][i4];
			}
		}
	}

	// 回代求解
	this->x = new T[size];
	for (int i8 = size - 1; i8 >= 0; i8--) {
		this->x[i8] = a_b[i8][size];
		for (int i9 = size - 1; i9 > i8; i9--)
			this->x[i8] -= a_b[i8][i9] * this->x[i9];
		this->x[i8] = this->x[i8] / a_b[i8][i8];
	}
	
	// 释放内存
	for (int i9 = 0; i9 < size; i9++)
		delete[]a_b[i9];
	delete[]a_b;
	
	return;
}

template<class T>
T* Elimination<T>::getx()
{
	return this->x;
}

template <class T>
void Elimination<T>::printx()
{
	std::cout << "x =";
	for (int i = 0; i < this->size; i++)
		std::cout << " " << this->x[i];
	std::cout << std::endl;
	return;
}