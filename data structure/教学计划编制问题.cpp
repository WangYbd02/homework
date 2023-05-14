#include <iostream>
#include <string>

using namespace std;

typedef struct Course
{
	string Code;  //课程编号
	int credit;  //学分
	int pre;  //先修课程数
	int num;  //输入次序
};

class Queue  //循环队列
{
public:
	Course* c;
	int Capacity;  //队列容量
	int front;  //队头指针
	int rear;  //队尾指针

	Queue(int capacity) {
		c = new Course[capacity + 1];
		this->Capacity = capacity;
		front = rear = 0;
	}
	~Queue() { delete[]c; }

	bool IsEmpty()  //判断队列是否为空
	{
		if (front == rear)
			return true;
		return false;
	}

	int GetSize()  //获取队列中的元素数量
	{
		int size;
		if (rear >= front)
			size = rear - front;
		else
			size = rear + this->Capacity + 1 - front;
		return size;
	}

	void Enqueue(Course ele)  //入队
	{
		if (front == (rear + 1) % (this->Capacity + 1)) {
			cout << "队列已满.\n";
			return;
		}
		c[rear] = ele;
		rear = (rear + 1) % (this->Capacity + 1);
	}

	Course& Dequeue()  //出队
	{
		if (front == rear) {
			Course fail;
			fail.credit = 0;
			return fail;
		}
		Course ct = c[front];
		front = (front + 1) % (this->Capacity + 1);
		return ct;
	}

	Course& GetFront()  //返回队头元素
	{
		if (IsEmpty()) {
			Course fail;
			fail.credit = 0;
			return fail;
		}
		return c[front];
	}
};

class System
{
private:
	Course* course;  //课程信息
	int** graph;  //课程间的关系
	int Term;  //学期总数
	int Credit;  //每学期的学分上限
	int V_num;  //课程总数
	int E_num;  //先修关系总数

public:
	System(int term, int cred, int quantity) {
		this->Term = term;
		this->Credit = cred;
		this->V_num = quantity;
		course = new Course[quantity];
		graph = new int* [quantity];
		for (int i = 0; i < quantity; i++)
			graph[i] = new int[quantity];
		for (int j = 0; j < quantity; j++){
			for (int k = 0; k < quantity; k++){
				graph[j][k] = 0;
			}
		}
	}
	~System() {
		delete[]course;
		for (int i = 0; i < this->V_num; i++)
			delete[]graph[i];
		delete[]graph;
	}

	void Input();  //输入
	bool Avg();  //课程尽量分布均匀
	bool Fro();  //课程尽量集中在前几个学期

};


void System::Input()
{
	int i, j, m, t;  //循环变量

	cout << "请依次输入" << this->V_num << "个课程的编号:\n";
	for (i = 0; i < this->V_num; i++)
		cin >> course[i].Code;

	cout << "请依次输入" << this->V_num << "个课程的学分:\n";
	for (j = 0; j < this->V_num; j++)
		cin >> course[j].credit;

	for (m = 0; m < this->V_num; m++)  //先修课程数置0，添加课程输入次序信息
	{
		course[m].pre = 0;
		course[m].num = m;
	}

	string s1, s2;
	int n1, n2;

	//判断输入的先修关系是否有效
	int flag1 = 0;
	int	flag2 = 0;

	int cnt = 0;  //记录有效边数
	cout << "请输入课程间的先修关系(先修课程在前，格式如:C01  C02)，以-1结束:\n";
	while (true)
	{
		cin >> s1;  //第一个课程编号
		if (s1 == "-1")
			break;
		cin >> s2;  //第二个课程编号
		flag1 = 0;
		flag2 = 0;
		for (t = 0; t < this->V_num; t++)
		{
			if (s1 == course[t].Code) {
				n1 = t;
				flag1 = 1;
			}
			else if (s2 == course[t].Code) {
				n2 = t;
				flag2 = 1;
			}

			if (flag1 == 1 && flag2 == 1)
				break;
		}

		if (flag1 == 1 && flag2 == 1){
			graph[n1][n2] = 1;
			course[n2].pre++;
			cnt++;
		}
		else{
			cout << "无效编号.\n";
		}
	}
	this->E_num = cnt;
}

bool System::Avg()
{

	Queue q(this->V_num);  //创建一个队列
	int i1, i2, i3, i4, i5;  //循环变量
	int avg1 = this->V_num / this->Term;  //平均每学期要学的课程，向下取整
	int avg2 = avg1;
	if (this->V_num > avg1 * this->Term)
		avg2 = avg2 + 1;

	int* Order = new int[this->V_num + this->Term - 1];  //记录排课顺序
	int order = 0;
	for (i1 = 0; i1 < this->V_num + this->Term - 1; i1++)
		Order[i1] = -2;

	int* prev = new int[this->V_num];  //记录每个课程未被安排的先修课程数
	for (i2 = 0; i2 < this->V_num; i2++)
		prev[i2] = course[i2].pre;

	int cnt = 0;  //已排课的数量，用于判断图中是否有环及课程能否在规定学期内排完
	int m_cre = 0;  //记录某一学期的学分总数
	Course temp;  //暂存队头课程信息
	int t = 0;  //暂存队头课程学分信息
	int tool = 0;  //记录某一学期的课程数

	//BFS 拓扑排序
	for (i3 = 0; i3 < this->V_num; i3++)
	{
		if (prev[i3] == 0)
			q.Enqueue(course[i3]);
	}
	while (!q.IsEmpty())
	{
		if (order == this->V_num + this->Term - 1)  //判断课程能否在规定学期内排完
			break;
		if (m_cre > this->Credit || tool == avg2)  //使课程尽量平均分配并保证不超过学分上限
		{
			Order[order++] = -1;
			m_cre = 0;
			tool = 0;
		}

		temp = q.GetFront();
		t = temp.num;
		m_cre = m_cre + course[t].credit;
		if (m_cre > this->Credit)  //判断学分总数是否超过学分上限
			continue;
		q.Dequeue();
		t = temp.num;
		Order[order++] = t;  //出队，表示排课
		cnt++;	tool++;
		for (i4 = 0; i4 < this->V_num; i4++)
		{
			if (graph[t][i4] == 1)  //相邻结点入度减一
			{
				graph[t][i4] = 0;
				prev[i4]--;
				if (prev[i4] == 0)  //相邻结点入度减一
					q.Enqueue(course[i4]);
			}
		}
	}
	if (cnt != this->V_num)
	{
		cout << "按该方法不能排出合适的课表.\n";
		return false;
	}
	
	//输出排课结果
	int k = 1;
	cout << "第1个学期:";
	for (i5 = 0; Order[i5] != -2 && i5 < this->V_num + this->Term - 1; i5++)
	{
		if (Order[i5] == -1)
		{
			k++;
			cout << "\n第" << k << "个学期:";
			continue;
		}
		cout << course[Order[i5]].Code << " ";
	}
	return true;
}

void Sort(Course c[], int N)  //选择排序(小->大)
{
	int i, j, k;
	Course temp;
	for (i = 0; i < N - 1; i++)
	{
		k = i;
		for (j = i + 1; j < N; j++)
		{
			if (c[j].credit > c[k].credit)
				k = j;
		}
		if (k != i)
		{
			temp = c[k];
			c[k] = c[i];
			c[i] = temp;
		}
	}
	return;
}


bool System::Fro()
{
	Queue q(this->V_num);  //创建一个队列
	int i1, i2, i3, i4, i5, i6, i7, i8, i9;  //循环变量

	int* Order = new int[this->V_num + this->Term - 1];  //记录排课顺序
	int order = 0;
	for (i1 = 0; i1 < this->V_num + this->Term - 1; i1++)
		Order[i1] = -2;

	int* prev = new int[this->V_num];  //记录每个课程未被安排的先修课程数
	for (i2 = 0; i2 < this->V_num; i2++)
		prev[i2] = course[i2].pre;

	Course* cTemp = new Course[this->V_num];  //暂存未排且入度为0的课
	Course c_void;
	c_void.credit = 0;
	for (i3 = 0; i3 < this->V_num; i3++)
		cTemp[i3] = c_void;
	int ctemp = 0;

	int cnt = 0;  //已排课的数量，用于判断图中是否有环及课程能否在规定学期内排完
	int m_cre = 0;  //记录某一学期的学分总数
	Course temp;  //暂存队头课程信息
	int t = 0;  //暂存队头课程学分信息

	for (i4 = 0; i4 < this->V_num; i4++)
	{
		if (prev[i4] == 0)
			cTemp[ctemp++] = course[i4];
	}
	Sort(cTemp, ctemp);  //选择排序，将课程按学分从小到大排序
	for (i5 = 0; i5 < ctemp; i5++)
	{
		q.Enqueue(cTemp[i5]);
		cTemp[i5] = c_void;
	}
	ctemp = 0;

	//BFS 拓扑排序
	while (!q.IsEmpty())
	{
		if (order == this->V_num + this->Term - 1)  //判断课程能否在规定学期内排完
			break;
		if (m_cre > this->Credit)  //保证不超过学分上限
		{
			Order[order++] = -1;
			m_cre = 0;

			//一个学期结束，进入下一学期前，进行一次选择排序
			while (!q.IsEmpty())
			{
				cTemp[ctemp++] = q.GetFront();
				q.Dequeue();
			}
			Sort(cTemp, ctemp);
			for (i6 = 0; i6 < ctemp; i6++)
			{
				q.Enqueue(cTemp[i6]);
				cTemp[i6] = c_void;
			}
			ctemp = 0;
		}
		temp = q.GetFront();
		t = temp.num;
		m_cre = m_cre + course[t].credit;
		if (m_cre > this->Credit)  //判断学分总数是否超过学分上限
			continue;

		q.Dequeue();
		Order[order++] = t;  //出队，表示排课
		cnt++;
		for (i7 = 0; i7 < this->V_num; i7++)
		{
			if (graph[t][i7] == 1)  //相邻结点入度减一
			{
				graph[t][i7] = 0;
				prev[i7]--;
				if (prev[i7] == 0)  //入队为0，等待进队
				{
					cTemp[ctemp++] = course[i7];
				}
			}
		}
		Sort(cTemp, ctemp);  //待进队课程排序
		for (i8 = 0; i8 < ctemp; i8++)
		{
			q.Enqueue(cTemp[i8]);
			cTemp[i8] = c_void;
		}
		ctemp = 0;

	}
	if (cnt != this->V_num)
	{
		cout << "按该方法不能排出合适的课表.\n";
		return false;
	}

	int k = 1;
	cout << "第1个学期:";
	for (i9 = 0; Order[i9] != -2 && i9 < this->V_num + this->Term - 1; i9++)
	{
		if (Order[i9] == -1)
		{
			k++;
			cout << "\n第" << k << "个学期:";
			continue;
		}
		cout << course[Order[i9]].Code << " ";
	}
	return true;
}


void Call()
{
	int term, cred, quantity;
	cout << "请输入学期总数:";
	cin >> term;
	cout << "请输入每学期的学分上限:";
	cin >> cred;
	cout << "请输入课程总数:";
	cin >> quantity;

	System s(term, cred, quantity);

	s.Input();

	cout << "请选择:\n\t1.学习负担尽量均匀的编排方式\n\t2.尽量集中在前几个学期的编排方式\n";
	cin.clear();
	cin.ignore();
	string n;
	getline(cin, n);
	int flag = 1;
	while (flag)
	{
		if (n[0] == '1' && n[1] == '\0') {
			flag = 0;
			s.Avg();
		}
		else if (n[0] == '2' && n[1] == '\0') {
			flag = 0;
			s.Fro();
		}
		else {
			cout << "非法选择，请重新选择:\n";
			getline(cin, n);
		}
	}

	return;
}


int main()
{
	Call();
	return 0;
}