/** ��Ҷ˹�Ż�
* ����������Ҫ�����
* ��1������ģ�ͣ����ڽ�ģĿ�꺯���ı�Ҷ˹����ģ�ͣ�
* ��2���������������ھ�����һ�������������
* BO�ĺ��ģ�������ǵ�Ŀ��ֻ���ҵ�Ŀ�꺯�������λ�ã���ôȷ���ƽ����ľ�ȷģ���ƺ��Ǻ��˷������ġ�
* ������ֻ����Ŀ�꺯������ֵʱ������Ҫ�Ľ�����ģ�Ͷ�Ŀ�꺯����Ԥ�ڽϵ͵������Ԥ�⾫�ȡ�
* BO������ѧϰ����������ѧϰѡ����һ���ȷ���ĵ���̽������������Ҷ˹�Ż���Ҫ��Explore��ȷ������
*����������ؾ��и߻ƽ��������ע��֪�ľ��и���Ŀ�꺯��Ԥ�������һ��Exploit��֮��ȡ��ƽ�⡣
*/

#pragma once

#include "SurrogateModel.h"
#include "AcquisitionFunc.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>

struct NextPointCandiate
{
    Eigen::VectorXd candiate_predictor;
    Eigen::VectorXd candiate_response;
};

class BayesianOptimization :
    public SurrogateModel, public AcquisitionFunc
{
public:
    /*BayesOpt*/
    std::string surrogate_model;
    std::string acquisition_func;

    /*������Ϣ*/
    int predictor_dimension;
    int response_dimension;

public:
    BayesianOptimization();

    /*����ģ��Ŀ�꺯���Ĵ���ģ�ͺͲ�������*/
    BayesianOptimization(std::string surrogate, std::string acquisition);

    void fit(Eigen::MatrixXd& data_block, int& data_dimension);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension);

    /*������һ����ѡ��*/
    NextPointCandiate findNextPointCandiate(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper);
};

