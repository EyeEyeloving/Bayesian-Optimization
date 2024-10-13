/**代理模型：
* 提供对未知区域的采样点的预测；它应该足够灵活以能通过微调来模拟真实函数
* （1）关键一：代理模型必须是一个概率模型
* （2）关键二：返回一个均值预测mean prediction和对当前预测的不确定性variance prediction
* 
*/

#pragma once

#include <iostream>
#include <string>

class SurrogateModel
{
public:
	std::string surrogate_model;

public:
	SurrogateModel();

	SurrogateModel(std::string surrogate);

	void fit();

private:
	void fitGaussianProcess();
};