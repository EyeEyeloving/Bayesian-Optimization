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

    /*数据信息*/
    int predictor_dimension;
    int response_dimension;

public:
    BayesianOptimization();

    /*设置模拟目标函数的代理模型和采样函数*/
    BayesianOptimization(std::string surrogate, std::string acquisition);

    void fit(Eigen::MatrixXd& data_block, int& data_dimension);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension);

    /*采样下一个候选点*/
    NextPointCandiate findNextPointCandiate(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper);
};

