#include <iostream>

using namespace std;

//{5, 88, 12, 56, 71, 28, 33, 43, 93, 17}
//除留取余法 p = 13

class Hash
{
private:
	int* hash;
	int Size;
	int Capacity;
	int p;  //所选素数
	
public:
	Hash(int Capacity, int p) {
		hash = new int[Capacity];
		for (int i = 0; i < Capacity; i++)
		{
			hash[i] = -1;
		}
		this->Capacity = Capacity;
		this->p = p;
		this->Size = 0;
	}
	~Hash() { delete hash; }

	bool linear_detection(int ele);
	int linear_search(int ele);
	bool squared_detection(int ele);
	int squared_search(int ele);
	void show();
	float linear_ASL(int num[], int N);
	float squared_ASL(int num[], int N);
};

bool Hash::linear_detection(int ele)
{
	if (this->Size == this->Capacity){
		cout << "表满" << endl;
		return false;
	}
	int pos = ele % p;
	while (true)
	{
		if (hash[pos] == -1){
			hash[pos] = ele;
			return true;
		}

		else{
			pos = (pos + 1) % p;
		}
	}
	return false;
}

int Hash::linear_search(int ele)
{
	int pos = ele % p;
	int count = 1;
	int flag = 0;
	while (count <= this->Capacity)
	{
		if (hash[pos] == ele) {
			flag = 1;
			break;
		}
		else if (hash[pos] == -1)
			break;
		else{
			pos = (pos + 1) % p;
			count++;
		}
	}
	if (flag == 1)
		return count;
	else
		return 0;
}

bool Hash::squared_detection(int ele)
{
	if (this->Size == this->Capacity){
		cout << "表满" << endl;
		return false;
	}
	int pos = ele % p;
	int t = 1;
	int k = 0;
	while (true)
	{
		if (hash[pos + k] == -1) {
			hash[pos + k] = ele;
			return true;
		}
		else {
			if (t % 2 == 1)
				k = ((t + 1) / 2) * ((t + 1) / 2);
			else
				k = -(t / 2) * (t / 2);
			t++;
		}
	}
	return false;
}

int Hash::squared_search(int ele)
{
	int pos = ele % p;
	int count = 1;
	int flag = 0;
	int t = 1;
	int k = 0;
	while (count <= this->Capacity)
	{
		if (hash[pos + k] == ele) {
			flag = 1;
			break;
		}
		else if (hash[pos] == -1)
			break;
		else {
			if (t % 2 == 1)
				k = ((t + 1) / 2) * ((t + 1) / 2);
			else
				k = -(t / 2) * (t / 2);
			t++;
			count++;
		}
	}
	if (flag == 1)
		return count;
	else
		return 0;
}

void Hash::show()
{
	for (int i = 0; i < this->Capacity; i++)
	{
		cout << hash[i] << " ";
	}
	return;
}

float Hash::linear_ASL(int num[], int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + linear_search(num[i]);
	}
	float asl = sum / N;
	return asl;
}

float Hash::squared_ASL(int num[], int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + squared_search(num[i]);
	}
	float asl = sum / N;
	return asl;
}


typedef struct node
{
	int data;
	node* next;
};

class Hash_l
{
private:
	node* hash;
	int Size;
	int Capacity;
	int p;  //所选素数

public:
	Hash_l(int Capacity, int p) {
		hash = new node[Capacity];
		for (int i = 0; i < Capacity; i++)
		{
			hash[i].data = 0;
			hash[i].next = NULL;
		}
		this->Capacity = Capacity;
		this->p = p;
		this->Size = 0;
	}
	~Hash_l() { delete hash; }

	bool linked_detection(int ele);
	int linked_search(int ele);
	void show();
	float ASL(int num[], int N);
};

bool Hash_l::linked_detection(int ele)
{
	int pos = ele % p;
	hash[pos].data++;
	node* p = &hash[pos];
	while (p->next != NULL)
	{			
		p = p->next;
	}
	node* s = new node;
	s->data = ele;
	s->next = NULL;
	p->next = s;
	return true;
}

int Hash_l::linked_search(int ele)
{
	int pos = ele % p;
	int count = 1;
	int flag = 0;
	if (hash[pos].data == 0)
		return 0;
	node* p = hash[pos].next;
	while (p != NULL)
	{
		if (p->data == ele){
			flag = 1;
			break;
		}
		else
		{
			p = p->next;
			count++;
		}
	}
	if (flag == 1)
		return count;
	else
		return 0;
}

void Hash_l::show()
{
	node* p;
	for (int i = 0; i < this->Capacity; i++)
	{
		p = hash[i].next;
		cout << i << ": ";
		while (p!= NULL)
		{
			cout << p->data << " ";
			p = p->next;
		}
		if (i != this->Capacity - 1)
			cout << endl;
	}
	return;
}

float Hash_l::ASL(int num[], int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + linked_search(num[i]);
	}
	float asl = sum / N;
	return asl;
}


int main()
{
	int num[10] = { 5,88,12,56,71,28,33,43,93,17 };

	Hash hash1(13, 13);
	for (int i = 0; i < 10; i++)
	{
		hash1.linear_detection(num[i]);
	}
	cout << "对于线性探测：" << endl;
	hash1.show();
	cout << endl;
	cout << "ASL = " << hash1.linear_ASL(num, 10) << endl;

	Hash hash2(13, 13);
	for (int j = 0; j < 10; j++)
	{
		hash2.squared_detection(num[j]);
	}
	cout << "对于平方探测：" << endl;
	hash2.show();
	cout << endl;
	cout << "ASL = " << hash2.squared_ASL(num, 10) << endl;

	Hash_l hash3(13, 13);
	for (int k = 0; k < 10; k++)
	{
		hash3.linked_detection(num[k]);
	}
	cout << "对于链地址法：" << endl;
	hash3.show();
	cout << endl;
	cout << "ASL = " << hash3.ASL(num, 10) << endl;
	

	return 0;
}