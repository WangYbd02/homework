/// please read README.md before using

#include <iostream>
#include "Cubic_Spline_Interpolation.hpp"


// 功能实现模块
void splineModule()
{
	int num;
	std::cout << "please input the number of dots:"; // 输入采样点的数量
	std::cin >> num;
	double* x = new double[num];  // x
	double* f = new double[num];  // f(x)
	for (int i = 0; i < num; i++)
	{
		std::cout << "x" << i << "=";
		std::cin >> x[i];
		std::cout << "f(x" << i << ")=";
		std::cin >> f[i];
	}
	CubicSplineInterpolation<double> c(num, x, f);
	c.selectBoundaryCondition();
	c.dsolve();
	c.printExpressions();
	return;
}

int main()
{
	splineModule();
	return 0;
}
