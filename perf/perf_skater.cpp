//
// Created by Xun Li on 2/5/21.
//

#include <iostream>
#include <vector>
#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/sa/LISA.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_clustering.h>
#include <libgeoda/GenUtils.h>

int main()
{
    GeoDa gda("/Users/xun/github/libgeoda_paper/data/us-sdoh-2014.shp");
    GeoDaWeight* w = gda_queen_weights(&gda, 1, false, false);
    std::vector<double> val = gda.GetNumericCol("ep_pov");

    int nCPUs = 6;
    int permutations = 99999;
    int last_seed_used = 123456789;
    double significance_cutoff = 0.01;

    std::string scale_method = "standardize";
    std::string distance_method = "euclidean";
    std::vector<double> bound_vals;
    double min_bound = 0;
    int seed = 123456789;
    int cpu_threads = 6;
    std::vector<std::vector<double> > data;
    data.push_back(val);

    std::vector<std::vector<int> > cluster_ids = gda_skater(4, w, data, scale_method, distance_method, bound_vals, min_bound, seed, cpu_threads);

    std::vector<int> clusters = GenUtils::flat_2dclusters(w->num_obs, cluster_ids);

    delete w;
}