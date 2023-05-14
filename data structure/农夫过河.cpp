//该程序只能给出一条安全路径
#include <iostream>
#include <queue>

using namespace std;


class Graph
{
	//用二进制四位数依次表示农夫，白菜，羊，狼的位置
	//0表示在南岸，1表示在北岸
	int** graph;
public:
	int* saferoad;  //安全路径
	int times;  //记录  步骤数 + 1
	int flag;  //判断是否存在安全路径

	Graph()
	{
		graph = new int*[16];
		for (int i = 0; i < 16; i++)
		{
			graph[i] = new int[16];
		}
		for (int x = 0; x < 16; x++)
		{
			for (int y = 0; y < 16; y++)
				graph[x][y] = -1;
		}
		saferoad = new int[16];
		for (int j = 0; j < 16; j++)
			saferoad[j] = -1;
	}
	~Graph()
	{
		delete[]saferoad;
		for (int i = 0; i < 16; i++)
		{
			delete[]graph[i];
		}
		delete[]graph;
	}

	bool IsSafe(int i);  //判断情况是否安全
	bool IsAccessible(int i, int j);  //判断两个步骤是否有可能连续
	bool Isaccessible(int i, int j);  //辅助函数
	void Construct();  //建立有向图图
	bool SafeRoad();  //BFS
	void PrintWay();  //打印步骤
};

bool Graph::IsSafe(int i)
{
	if (i == 3 || i == 12 || (i >= 6 && i <= 9))
		return false;
	else
		return true;
}

bool Graph::Isaccessible(int i, int j)  //i > j
{
	if (i == 7 && (j == 6 || j == 5 || j == 3))
		return true;
	else if (i == 6 && (j == 4 || j == 2))
		return true;
	else if (i == 5 && (j == 4 || j == 1))
		return true;
	else if (i == 4 && j == 0)
		return true;
	else if (i == 3 && (j == 2 || j == 1))
		return true;
	else if (j == 0 && (i == 1 || i == 2))
		return true;
	else
		return false;
}

bool Graph::IsAccessible(int i, int j)
{
	if (i == j)  //不能有环
		return false;
	else if (i - 8 < 0 && j - 8 < 0)  //农夫不能停留
		return false;
	else if (i - 8 >= 0 && j - 8 >= 0)  //农夫不能停留
		return false;

	if (i - 8 >= 0)
	{
		if (i - 8 > j)
		{
			if (Isaccessible(i - 8, j) != 0)
				return true;
			else
				return false;
		}
		else if (i - 8 == j)
			return true;
		else
		{
			if (Isaccessible(j, i - 8) != 0)
			{
				return true;
			}
			else
				return false;
		}
	}
	else
	{
		if (j - 8 > i)
		{
			if (Isaccessible(j - 8, i) != 0)
				return true;
			else
				return false;
		}
		else if (j - 8 == i)
			return true;
		else
		{
			if (Isaccessible(i, j - 8) != 0)
			{
				return true;
			}
			else
				return false;
		}
	}
}

void Graph::Construct()
{
	int i, j;
	for (i = 0; i < 15; i++) //0b1111不能作为始点
	{
		if (IsSafe(i) == 0)
			continue;
		for (j = 1; j < 16; j++)  //0b0000不能作为终点
		{
			if (IsSafe(j) == 0)
				continue;
			if (IsAccessible(i, j) != 0)
				graph[i][j] = 1;
		}
	}
}

bool Graph::SafeRoad()
{
	this->flag = 1;
	int visited[16];
	for (int i = 0; i < 16; i++)
		visited[i] = -1;

	queue<int> q;
	int s = 0;
	int k = 1;

	while (true)
	{
		for (k = 1; k < 16; k++)
		{
			if (graph[s][k] == 1 && visited[k] == -1)  //不能存在回路
			{
				q.push(k);
				visited[k] = s;
			}
		}
		if (graph[s][15] == 1)  //达成目标，跳出
			break;
		if (q.size() != 0)
		{
			s = q.front();
			q.pop();
		}
		else
		{
			this->flag = 0;  //安全路径不存在
			return false;
		}
	}

	int j, t, m;
	int p = 15;
	int cnt = 0;
	int* inverse_road = new int[16];  //从终点往起点找路径

	for (j = 0; j < 16; j++)
	{
		inverse_road[j] = p;
		cnt++;
		if (p == 0)
			break;

		t = visited[p];
		p = t;
	}

	this->times = cnt;
	for (m = 0; m < cnt; m++)
	{
		this->saferoad[m] = inverse_road[cnt - m - 1];
	}

	delete[]inverse_road;

	return true;
}

void Graph::PrintWay()
{
	if (this->flag == 0)
	{
		cout << "不存在安全路径" << endl;
	}
	cout << "假定开始时农夫、白菜、羊、狼都在南岸，目标是使农夫、白菜、羊、狼到达北岸" << endl;
	cout << endl;
	int i = 1;
	int s = 0;
	int sign = 0;
	for (i = 1; i < times; i++)
	{
		s = saferoad[i] - saferoad[i - 1];
		cout << "第" << i << "步: 农夫";
		if (s > 0)
			cout << "从南岸前往北岸，";
		else
			cout << "从北岸前往南岸，";
		if (s == 8 || s == -8)
			cout << "没带任何东西，";
		else if (s == 12 || s == -12)
			cout << "带着白菜，";
		else if (s == 10 || s == -10)
			cout << "带着羊,";
		else if (s == 9 || s == -9)
			cout << "带着狼，";

		sign = 0;
		cout << "到达后，北岸";
		if (saferoad[i] >= 8)
		{
			cout << "有农夫";
			sign = 1;
		}
		if ((saferoad[i] % 8) >= 4)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "白菜";
			sign = 1;
		}
		if ((saferoad[i] % 4) >= 2)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "羊";
			sign = 1;
		}
		if (saferoad[i] % 2 == 1)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "狼";
			sign = 1;
		}
		if (sign == 0)
			cout << "没有任何东西";

		sign = 0;
		cout << ",南岸";
		if (s < 8)
		{
			cout << "有农夫";
			sign = 1;
		}
		if ((saferoad[i] % 8) < 4)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "白菜";
			sign = 1;
		}
		if ((saferoad[i] % 4) < 2)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "羊";
			sign = 1;
		}
		if (saferoad[i] % 2 == 0)
		{
			if (sign == 1)
				cout << "、";
			else
				cout << "有";
			cout << "狼";
			sign = 1;
		}
		if (sign == 0)
			cout << "没有任何东西";
		cout << endl;
	}
	cout << "此时，目标达成" << endl;
}

int main()
{
	Graph G;
	G.Construct();
	G.SafeRoad();
	G.PrintWay();
	return 0;
}