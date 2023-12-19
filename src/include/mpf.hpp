#ifndef MPF_H
#define MPF_H

#include "rpf.hpp"

class MultiplicativeRPF : public RandomPlantedForest
{

public:
  // using RandomPlantedForest::calcOptimalSplit;
  MultiplicativeRPF(const NumericMatrix &samples_Y, const NumericMatrix &samples_X, const NumericVector parameters = {1, 50, 30, 10, 0.4, 0, 0, 0, 0, 0, 0.1});
  ~MultiplicativeRPF(){};

private:
  void create_tree_family(std::vector<Leaf> initial_leaves, size_t n) override;
  void fit() override;
  Split calcOptimalSplit(const std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &X,
                         std::multimap<int, std::shared_ptr<DecisionTree>> &possible_splits, TreeFamily &curr_family) override;
  Split calcOptimalSplit2(const std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &X,
                        std::multimap<int, std::shared_ptr<DecisionTree>> &possible_splits, TreeFamily &curr_family) override;
  Split calcOptimalSplitForLeaf(Leaf &leaf, const std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &X, int k);
  std::vector<double> getSamplePoints(const std::vector<std::vector<double>> &X, const std::vector<int> individuals, int k);
};

#endif