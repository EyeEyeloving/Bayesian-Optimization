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
#include "AcquisitionFcn.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <Eigen/Dense>

struct PointIncumbent {
    Eigen::VectorXd candiate_predictor;
    Eigen::VectorXd candiate_response;
};

//struct ObjFcnEvaluation {
//    Eigen::VectorXd evaluation_response; // Ŀ�꺯��ֵ
//};

struct Trace {
    std::vector<int> index_trace; // ����ģ�͹��Ƶ�����λ�õ�����
    Eigen::MatrixXd best_estimated_trace; // ����ģ��Ԥ�������Ŀ��ֵ
    Eigen::MatrixXd best_sampling_trace; // ���ڲ���/trial��ʵ������Ŀ��ֵ
};

class BayesianOptimization :
    public SurrogateModel, public AcquisitionFcn
{
public:
    /*BayesOpt*/
    // objective_fcn
    int number_bayesopt = 0;
    int number_evaluation = 0; // ��δ֪�����Ĳ���/��Ŀ�꺯������������
    int number_max_iter = 10;
    std::vector<Trace> trace_table;

    std::string surrogate_model;
    std::string acquisition_func;

    /*������Ϣ*/
    std::vector<double> predictor_domainMin;
    std::vector<double> predictor_domainMax;
    int predictor_dimension;
    Eigen::MatrixXd predictor_set;
    int response_dimension;
    Eigen::MatrixXd response_set;

private: 
    std::queue<Eigen::VectorXd*> init_try_point_queue;
    bool optimization_finished = false;

public:
    BayesianOptimization();

    /*����ģ��Ŀ�꺯���Ĵ���ģ�ͺͲ�������*/
    BayesianOptimization(std::string surrogate, std::string acquisition);

    void fit(Eigen::MatrixXd& data_block, int& data_dimension);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension);

    std::queue<Eigen::VectorXd*> processInitializationData(const int& number_init);

    /*������һ����ѡ��*/
    PointIncumbent findIncumbent();

    std::vector<PointIncumbent> findFBestGlobal(int& num_initial_points, int& num_best_points);

    Eigen::VectorXd findNextInAcquisitionFcn();

    Eigen::VectorXd fitAcquisitionFcn();

    /*ʹ�������ѡ��ִ��Action�����Reward����Evaluation by sampling the objective function*/
    Eigen::VectorXd callObjectiveFcn(const Eigen::VectorXd& candiate_predictor);

    /*������֪���ݼ�*/
    void augmentObservationSet(const Eigen::VectorXd candiate_predictor, const Eigen::VectorXd candiate_response);

    /*ѵ������ģ��*/
    void fitSurrogateModel();

    /*����trace_table����*/
    void updateTrace();

    /*���´���ģ��*/
    void updateSurrogateMdl();
};

