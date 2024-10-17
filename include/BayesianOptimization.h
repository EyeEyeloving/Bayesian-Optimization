/** 贝叶斯优化
* 包含两个主要组件：
* （1）代理模型：用于建模目标函数的贝叶斯概率模型；
* （2）采样函数：用于决定下一步在哪里采样。
* BO的核心：如果我们的目标只是找到目标函数的最高位置，那么确定黄金含量的精确模型似乎是很浪费算力的。
* 即，当只关心目标函数的最值时，不需要改进代理模型对目标函数在预期较低的区域的预测精度。
* BO与主动学习的区别：主动学习选择下一个最不确定的点来探索函数，但贝叶斯优化需要在Explore不确定区域
*（可能意外地具有高黄金含量）与关注已知的具有更高目标函数预测的区域（一种Exploit）之间取得平衡。
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
//    Eigen::VectorXd evaluation_response; // 目标函数值
//};

struct Trace {
    std::vector<int> index_trace; // 代理模型估计的最优位置的索引
    Eigen::MatrixXd best_estimated_trace; // 代理模型预测的最优目标值
    Eigen::MatrixXd best_sampling_trace; // 基于采样/trial的实际最优目标值
};

class BayesianOptimization :
    public SurrogateModel, public AcquisitionFcn
{
public:
    /*BayesOpt*/
    // objective_fcn
    int number_bayesopt = 0;
    int number_evaluation = 0; // 对未知函数的采样/对目标函数的评估次数
    int number_max_iter = 10;
    std::vector<Trace> trace_table;

    std::string surrogate_model;
    std::string acquisition_func;

    /*数据信息*/
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

    /*设置模拟目标函数的代理模型和采样函数*/
    BayesianOptimization(std::string surrogate, std::string acquisition);

    void fit(Eigen::MatrixXd& data_block, int& data_dimension);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension);

    std::queue<Eigen::VectorXd*> processInitializationData(const int& number_init);

    /*采样下一个候选点*/
    PointIncumbent findIncumbent();

    std::vector<PointIncumbent> findFBestGlobal(int& num_initial_points, int& num_best_points);

    Eigen::VectorXd findNextInAcquisitionFcn();

    Eigen::VectorXd fitAcquisitionFcn();

    /*使用这个候选点执行Action并获得Reward，即Evaluation by sampling the objective function*/
    Eigen::VectorXd callObjectiveFcn(const Eigen::VectorXd& candiate_predictor);

    /*扩充已知数据集*/
    void augmentObservationSet(const Eigen::VectorXd candiate_predictor, const Eigen::VectorXd candiate_response);

    /*训练代理模型*/
    void fitSurrogateModel();

    /*更新trace_table属性*/
    void updateTrace();

    /*更新代理模型*/
    void updateSurrogateMdl();
};

