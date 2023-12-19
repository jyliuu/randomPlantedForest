#include "mpf.hpp"

// constructor
MultiplicativeRPF::MultiplicativeRPF(const NumericMatrix &samples_Y, const NumericMatrix &samples_X,
                                         const NumericVector parameters)
{

  // Ensure correct Rcpp RNG state
  Rcpp::RNGScope scope;

  // initialize class members
  std::vector<double> pars = to_std_vec(parameters);
  if (pars.size() != 9)
  {
    Rcout << "Wrong number of parameters - set to default." << std::endl;
    this->max_interaction = 1;
    this->n_trees = 50;
    this->n_splits = 30;
    this->split_try = 10;
    this->t_try = 0.4;
    this->purify_forest = 0;
    this->deterministic = 0;
    this->nthreads = 1;
    this->cross_validate = 0;
  }
  else
  {
    this->max_interaction = pars[0];
    this->n_trees = pars[1];
    this->n_splits = pars[2];
    this->split_try = pars[3];
    this->t_try = pars[4];
    this->purify_forest = pars[5];
    this->deterministic = pars[6];
    this->nthreads = pars[7];
    this->cross_validate = pars[8];
  }

  // set data and data related members
  this->set_data(samples_Y, samples_X);
}

// determine optimal split
Split MultiplicativeRPF::calcOptimalSplit(const std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &X,
                                             std::multimap<int, std::shared_ptr<DecisionTree>> &possible_splits, TreeFamily &curr_family)
{

  Split curr_split, min_split;
  curr_split.Y = &Y;
  std::set<int> tree_dims;
  std::vector<double> unique_samples;
  int k;
  unsigned int n = 0;
  double leaf_size, sample_point;

  // sample possible splits
  unsigned int n_candidates = ceil(t_try * possible_splits.size()); // number of candidates that will be considered
  std::vector<int> split_candidates(possible_splits.size());
  std::iota(split_candidates.begin(), split_candidates.end(), 0); // consecutive indices of possible candidates

  if (!deterministic)
  {
    shuffle_vector(split_candidates.begin(), split_candidates.end()); // shuffle for random order
  }

  // consider a fraction of possible splits
  while (n < n_candidates)
  {

    auto candidate = possible_splits.begin();
    std::advance(candidate, split_candidates[n]); // get random split candidate without replacement
    k = candidate->first - 1;                     // split dim of current candidate, converted to index starting at 0
    leaf_size = n_leaves[k];

    // Test if splitting in the current tree w.r.t. the coordinate "k" is an element of candidate tree
    tree_dims = candidate->second->split_dims;
    tree_dims.erase(k + 1);

    std::vector<std::shared_ptr<DecisionTree>> curr_trees;
    if (tree_dims.size() == 0)
      curr_trees.push_back(curr_family[std::set<int>{0}]);
    if (curr_family.find(tree_dims) != curr_family.end())
      curr_trees.push_back(curr_family[tree_dims]);
    if (curr_family.find(candidate->second->split_dims) != curr_family.end())
      curr_trees.push_back(curr_family[candidate->second->split_dims]);

    // go through all trees in current family
    for (auto &curr_tree : curr_trees)
    {

      // skip if tree has no leaves
      if (curr_tree->leaves.size() == 0)
        continue;

      // go through all leaves of current tree
      // TODO: Swap this loop with go through samples loop
      for (auto &leaf : curr_tree->leaves)
      {
        curr_split = calcOptimalSplitForLeaf(leaf, Y, X, k);     
        // update split if squared sum is smaller
        // TODO: save split coordinate and tree index only, not leaf
        if (curr_split.min_sum < min_split.min_sum)
        {
          min_split = curr_split;
          min_split.tree_index = curr_tree;
        }
      }
    }

    ++n;
  }

  return min_split;
}

Split MultiplicativeRPF::calcOptimalSplitForLeaf(Leaf &leaf, const std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &X, const int k) {
  Split curr_split, min_split;
  const double leaf_size = n_leaves[k];

  std::vector<double> tot_sum(value_size, 0);
  // extract sample points according to individuals from X and Y
  
  const std::vector<double> sample_points = getSamplePoints(X, leaf.individuals, k);

        // go through samples
  for (size_t sample_pos = 0; sample_pos < sample_points.size(); ++sample_pos)
        {

          // get samplepoint
    double sample_point = sample_points[sample_pos];

          // clear current split
          {
            curr_split.I_s.clear();
            curr_split.I_b.clear();
            curr_split.I_s.reserve(leaf.individuals.size());
            curr_split.I_b.reserve(leaf.individuals.size());
            curr_split.M_s = std::vector<double>(value_size, 0);
            curr_split.M_b = std::vector<double>(value_size, 0);
          }

          // get samples greater/smaller than samplepoint
          if (sample_pos == 0)
          {
            curr_split.sum_s = std::vector<double>(value_size, 0);
            curr_split.sum_b = std::vector<double>(value_size, 0);

            for (int individual : leaf.individuals)
            {
              if (X[individual][k] < sample_point)
              {
                curr_split.I_s.push_back(individual);
                curr_split.sum_s += Y[individual];
              }
              else
              {
                curr_split.I_b.push_back(individual);
                curr_split.sum_b += Y[individual];
              }
            }

            tot_sum = curr_split.sum_s + curr_split.sum_b;
          }
          else
          {

            for (int individual : leaf.individuals)
            {
              if (X[individual][k] < sample_point)
              {
          if (X[individual][k] >= sample_points[sample_pos - 1])
                {
                  curr_split.sum_s += Y[individual];
                }
                curr_split.I_s.push_back(individual);
              }
              else
              {
                curr_split.I_b.push_back(individual);
              }
            }

            curr_split.sum_b = tot_sum - curr_split.sum_s;
          }

          // accumulate squared mean and get mean
    // TODO: accumulate the sum of error of multiple leaves
          L2_loss(curr_split);

          // update split if squared sum is smaller
    // TODO: save split coordinate and tree index only, not leaf
          if (curr_split.min_sum < min_split.min_sum)
          {
            min_split = curr_split;
            min_split.leaf_index = &leaf;
            min_split.split_coordinate = k + 1;
            min_split.split_point = sample_point;
          }
  }

  return min_split;
        }

std::vector<double> MultiplicativeRPF::getSamplePoints(const std::vector<std::vector<double>> &X, const std::vector<int> individuals, const int k) {
    const double leaf_size = n_leaves[k];

  std::vector<double> unique_samples = std::vector<double>(individuals.size());
  for (unsigned int i = 0; i < individuals.size(); ++i)
  {
    unique_samples[i] = X[individuals[i]][k];
  }
  std::sort(unique_samples.begin(), unique_samples.end());
  unique_samples.erase(std::unique(unique_samples.begin(), unique_samples.end()), unique_samples.end());

  // check if number of sample points is within limit
  if (unique_samples.size() < 2 * leaf_size)
    return std::vector<double>();

  // consider split_try-number of samples
  std::vector<double> samples_points = std::vector<double>(split_try);
  for (size_t i = 0; i < samples_points.size(); ++i)
    samples_points[i] = unique_samples[R::runif(leaf_size, unique_samples.size() - leaf_size)];
  std::sort(samples_points.begin(), samples_points.end());

  return samples_points;
}

  }

  return min_split;
}

void MultiplicativeRPF::create_tree_family(std::vector<Leaf> initial_leaves, size_t n)
{

  TreeFamily curr_family;
  curr_family.insert(std::make_pair(std::set<int>{0}, std::make_shared<DecisionTree>(DecisionTree(std::set<int>{0}, initial_leaves)))); // save tree with one leaf in the beginning
  // store possible splits in map with splitting variable as key and pointer to resulting tree
  std::multimap<int, std::shared_ptr<DecisionTree>> possible_splits;
  for (int feature_dim = 1; feature_dim <= feature_size; ++feature_dim)
  {
    // add pointer to resulting tree with split dimension as key
    curr_family.insert(std::make_pair(std::set<int>{feature_dim}, std::make_shared<DecisionTree>(DecisionTree(std::set<int>{feature_dim}))));
    possible_splits.insert(std::make_pair(feature_dim, curr_family[std::set<int>{feature_dim}]));
  }

  // sample data points with replacement
  int sample_index;
  std::vector<std::vector<double>> samples_X;
  std::vector<std::vector<double>> samples_Y;

  // deterministic
  if (deterministic)
  {
    samples_X = X;
    samples_Y = Y;
    this->t_try = 1;
  }
  else
  {
    samples_X = std::vector<std::vector<double>>(sample_size);
    samples_Y = std::vector<std::vector<double>>(sample_size);

    for (size_t i = 0; i < sample_size; ++i)
    {

      sample_index = R::runif(0, sample_size - 1);
      samples_Y[i] = Y[sample_index];
      samples_X[i] = X[sample_index];
    }
  }

  // modify existing or add new trees through splitting
  Split curr_split;
  for (int split_count = 0; split_count < n_splits; ++split_count)
  {

    // find optimal split
    curr_split = calcOptimalSplit2(samples_Y, samples_X, possible_splits, curr_family);

    // continue only if we get a significant result
    if (!std::isinf(curr_split.min_sum))
    {

      // update possible splits
      for (int feature_dim = 1; feature_dim <= feature_size; ++feature_dim)
      { // consider all possible dimensions

        // ignore dim if same as split coordinate or in dimensions of old tree
        if (feature_dim == curr_split.split_coordinate || curr_split.tree_index->split_dims.count(feature_dim) > 0)
          continue;

        // create union of split coord, feature dim and dimensions of old tree
        std::set<int> curr_dims = curr_split.tree_index->split_dims;
        curr_dims.insert(curr_split.split_coordinate);
        curr_dims.insert(feature_dim);
        curr_dims.erase(0);

        // do not exceed maximum level of interaction
        if (max_interaction >= 0 && curr_dims.size() > (size_t)max_interaction)
          continue;

        // skip if possible_split already exists
        if (possibleExists(feature_dim, possible_splits, curr_dims))
          continue;

        // check if resulting tree already exists in family
        std::shared_ptr<DecisionTree> found_tree = treeExists(curr_dims, curr_family);

        // update possible_splits if not already existing
        if (found_tree)
        { // if yes add pointer
          possible_splits.insert(std::make_pair(feature_dim, found_tree));
        }
        else
        { // if not create new tree
          curr_family.insert(std::make_pair(curr_dims, std::make_shared<DecisionTree>(DecisionTree(curr_dims))));
          possible_splits.insert(std::make_pair(feature_dim, curr_family[curr_dims]));
        }
      }

      // update values of individuals of split interval with mean
      for (int individual : curr_split.leaf_index->individuals)
      { // todo: loop directly over I_s I_b
        if (samples_X[individual][curr_split.split_coordinate - 1] < curr_split.split_point)
        {
          samples_Y[individual] -= curr_split.M_s;
        }
        else
        {
          samples_Y[individual] -= curr_split.M_b;
        }
      }

      // construct new leaves
      Leaf leaf_s, leaf_b;
      {
        leaf_s.individuals = curr_split.I_s;
        leaf_b.individuals = curr_split.I_b;

        leaf_s.value = curr_split.M_s;
        leaf_b.value = curr_split.M_b;

        // initialize interval with split interval
        leaf_s.intervals = curr_split.leaf_index->intervals;
        leaf_b.intervals = curr_split.leaf_index->intervals;

        // interval of leaf with smaller individuals has new upper bound in splitting dimension
        leaf_s.intervals[curr_split.split_coordinate - 1].second = curr_split.split_point;
        // interval of leaf with bigger individuals has new lower bound in splitting dimension
        leaf_b.intervals[curr_split.split_coordinate - 1].first = curr_split.split_point;
      }

      // construct split_dims of resulting tree when splitting in split_coordinate
      std::set<int> resulting_dims = curr_split.tree_index->split_dims;
      resulting_dims.insert(curr_split.split_coordinate);
      resulting_dims.erase(0);

      // check if resulting tree already exists in family
      std::shared_ptr<DecisionTree> found_tree = treeExists(resulting_dims, curr_family);

      // determine which tree is modified
      if (curr_split.tree_index->split_dims.count(curr_split.split_coordinate))
      { // if split variable is already in tree to be split
        // change values
        {
          leaf_s.value += curr_split.leaf_index->value;
          leaf_b.value += curr_split.leaf_index->value;
        }
        *curr_split.leaf_index = leaf_b;                 // replace old interval
        curr_split.tree_index->leaves.push_back(leaf_s); // add new leaf
      }
      else
      {                                       // otherwise
        found_tree->leaves.push_back(leaf_s); // append new leaves
        found_tree->leaves.push_back(leaf_b);
      }
    }
  }

  // remove empty trees & clear individuals of each tree
  auto keys = getKeys(curr_family);
  for (auto &key : keys)
  {
    if (curr_family[key]->leaves.size() == 0)
    {
      curr_family.erase(key);
      continue;
    }
    for (auto &leaf : curr_family[key]->leaves)
    {
      leaf.individuals.clear();
    }
  }

  tree_families[n] = curr_family;
}

// fit forest to new data
void MultiplicativeRPF::fit()
{

  // setup initial set of individuals
  std::vector<int> initial_individuals(sample_size);
  std::iota(initial_individuals.begin(), initial_individuals.end(), 0);

  // initialize intervals with lower and upper bounds
  std::vector<Interval> initial_intervals(feature_size);
  for (int i = 0; i < feature_size; ++i)
    initial_intervals[i] = Interval{lower_bounds[i], upper_bounds[i]};

  // set properties of first leaf
  Leaf initial_leaf;
  {
    initial_leaf.value = std::vector<double>(value_size, 0);
    initial_leaf.individuals = initial_individuals;
    initial_leaf.intervals = initial_intervals;
  }
  std::vector<Leaf> initial_leaves{initial_leaf}; // vector with initial leaf

  // initialize tree families
  this->tree_families = std::vector<TreeFamily>(n_trees);

  // Loop over number of tree families and dispatch threads in batches
  // of nhreads at once
  if (nthreads > 1)
  {
    if (nthreads > std::thread::hardware_concurrency())
    {
      Rcout << "Requested " << nthreads << " threads but only " << std::thread::hardware_concurrency() << " available" << std::endl;
    }
    // Create local thread count to not overwrite nthreads,
    // would get reported wrongly by get_parameters()
    unsigned int current_threads = nthreads;
    for (int n = 0; n < n_trees; n += current_threads)
    {
      if (n >= (n_trees - current_threads + 1))
      {
        current_threads = n_trees % current_threads;
      }

      std::vector<std::thread> threads(current_threads);
      for (int t = 0; t < current_threads; ++t)
      {
        // Rcout << "Dispatching thread " << (n + t + 1) << "/" << n_trees << std::endl;
        threads[t] = std::thread(&MultiplicativeRPF::create_tree_family, this, std::ref(initial_leaves), n + t);
      }
      for (auto &t : threads)
      {
        if (t.joinable())
          t.join();
      }
    }
  }
  else
  {
    for (int n = 0; n < n_trees; ++n)
    {
      create_tree_family(initial_leaves, n);
    }
  }

  // optionally purify tree
  if (purify_forest)
  {
    this->purify_3();
  }
  else
  {
    purified = false;
  }
}
