/**��˹����
* �����˲�������Ŀ�꺯��ֵ�ĺ�����ʷֲ��������������㣬���ڸõ��Ŀ�꺯��ֵ���������ʷֲ���
* ������õ����ʵĿ��ֵ�����ɶ��Ǳ��Ŀ�꺯�������õ�������GP���ò��������ʵ����ֵ��Ϊһ���������
* ��ˣ�GP�ڲ�������������Ǻ���ֵ��������ķֲ������ڲ������ȡֵ��Χ�ڣ�GP������Ϊ�Ƕ��Ǳ��Ŀ�꺯���Ĵ���ģ��
* GP��һ�ַǲ���ģ�ͣ�������ͨ��mean function��covariance/kernel function��ȫ����
* kernel function��ѡ��Ӱ��GP��Ŀ�꺯����ģ�⣬�����˶�Ŀ�꺯������������֪ʶ
* GP���ı׶��ǵ��������ϴ�ʱ�ļ��㸴�Ӷ�
*/

#pragma once

#include <Eigen/Dense>

class GaussianProcess
{
private:
	/*�ռ���֪����������ݼ��Ļ�����Ϣ������Ԥ���Ӻ���Ӧ��*/
	int predict_dimension;
	Eigen::MatrixXd predictor;
	int response_dimension;
	Eigen::MatrixXd response;

	/*Gaussian Kernel*/
	double kernel_sigma_gk;
	double kernel_scale_gk;

public:
	GaussianProcess();

	void fit(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& Y_train);

	void predict();

private:
	Eigen::RowVectorXd fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::VectorXd& X2);
	
	Eigen::MatrixXd fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
};

