/// ����Ԫ��˹��ȥ��

#pragma once
#include <iostream>


template <class T>
class Elimination
{
private:
	int size; // ����Ĵ�С
	T** A_b; // �������
	T* x; // ������
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
	void dsolve(); // ���
	/*
	* ��������������:ĳһ��(�����������)����Ԫ����ǰ,�ӵ�ǰ�е����һ��ȫΪ0,(���������������ʽΪ0),��������� 
	*/
	T* getx(); // ���ؽ����� ��Ҫ�ȵ���dsolve()
	void printx(); // ��������� ��Ҫ�ȵ���dsolve()
};

template <class T>
void Elimination<T>::dsolve()
{
	T** a_b; // �������A_b�Ŀ���
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

	// ����н����ξ���
	int max_row = 0;
	T* temp = NULL;
	T beta;
	for (int i4 = 0; i4 < size - 1; i4++)
	{
		max_row = i4; // ��¼��ǰ���о���ֵ������������
		for (int i5 = i4 + 1; i5 < size; i5++) { // �ҵ���ǰ���о���ֵ������������
			if (abs(a_b[i5][i4]) > abs(a_b[max_row][i4]))
				max_row = i5;
		}
		if (max_row != i4) { // ������
			temp = a_b[i4];
			a_b[i4] = a_b[max_row];
			a_b[max_row] = temp;
		}
		for (int i6 = i4 + 1; i6 < size; i6++) { //�õ��н����ξ���
			beta = a_b[i6][i4];
			for (int i7 = i4; i7 <= size; i7++) {
				a_b[i6][i7] -= beta * a_b[i4][i7] / a_b[i4][i4];
			}
		}
	}

	// �ش����
	this->x = new T[size];
	for (int i8 = size - 1; i8 >= 0; i8--) {
		this->x[i8] = a_b[i8][size];
		for (int i9 = size - 1; i9 > i8; i9--)
			this->x[i8] -= a_b[i8][i9] * this->x[i9];
		this->x[i8] = this->x[i8] / a_b[i8][i8];
	}
	
	// �ͷ��ڴ�
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