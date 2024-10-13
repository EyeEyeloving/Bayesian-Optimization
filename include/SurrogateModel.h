/**����ģ�ͣ�
* �ṩ��δ֪����Ĳ������Ԥ�⣻��Ӧ���㹻�������ͨ��΢����ģ����ʵ����
* ��1���ؼ�һ������ģ�ͱ�����һ������ģ��
* ��2���ؼ���������һ����ֵԤ��mean prediction�ͶԵ�ǰԤ��Ĳ�ȷ����variance prediction
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