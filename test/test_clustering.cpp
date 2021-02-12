//
// Created by Xun Li on 2019-06-06.
//

#include <vector>
#include <limits.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_clustering.h>


using namespace testing;

namespace {

    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};

    TEST(CLUSTERING_TEST, SKATER) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::vector<double> bound_vals;
        double bound = 0.0;
        int rand_seed = 123456789;
        int cpu_threads = 6;
        std::vector<std::vector<int> > clst = gda_skater(4, w, data, "standardize", "euclidean", bound_vals, bound, rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double between_ss = gda_betweensumofsquare(clst, data);
        double ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039);

        std::vector<std::vector<int> > clst_6 = gda_skater(6, w, data, "standardize", "euclidean", bound_vals, bound, rand_seed, cpu_threads );
        between_ss = gda_betweensumofsquare(clst_6, data);
        ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.419541653761205);
        delete w;
    }

    TEST(CLUSTERING_TEST, REDCAP) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::string method = "firstorder-singlelinkage";
        std::vector<double> bound_vals;
        double bound = 0.0;
        int rand_seed = 123456789;
        int cpu_threads = 6;
        std::vector<std::vector<int> > clst = gda_redcap(4, w, data, "standardize", method, "euclidean", bound_vals, bound, rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double between_ss = gda_betweensumofsquare(clst, data);
        double ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039); // same as SKATER

        method = "fullorder-completelinkage";
        std::vector<std::vector<int> > clst1 = gda_redcap(4, w, data, "standardize", method, "euclidean", bound_vals, bound, rand_seed, cpu_threads);
        between_ss = gda_betweensumofsquare(clst1, data);
        ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.35109901091774076);

        method = "fullorder-completelinkage";
        std::vector<std::vector<int> > clst2 = gda_redcap(5, w, data, "standardize", method, "euclidean", bound_vals, bound, rand_seed, cpu_threads);
        between_ss = gda_betweensumofsquare(clst2, data);
        ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.39420885009615186);

        method = "fullorder-averagelinkage";
        std::vector<std::vector<int> > clst3 = gda_redcap(4, w, data, "standardize", method, "euclidean", bound_vals, bound, rand_seed, cpu_threads);

        between_ss = gda_betweensumofsquare(clst3, data);
        ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.30578249025454063);

        method = "fullorder-singlelinkage";
        std::vector<std::vector<int> > clst4 = gda_redcap(4, w, data, "standardize", method, "euclidean", bound_vals, bound, rand_seed, cpu_threads);

        between_ss = gda_betweensumofsquare(clst4, data);
        ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.29002543000953057);

        delete w;
    }

    TEST(CLUSTERING_TEST, MAXP_GREEDY) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        int iterations = 99;
        std::vector<std::pair<double, std::vector<double>>> min_bounds;
        std::vector<std::pair<double, std::vector<double>>> max_bounds;
        std::vector<int> init_regions;
        std::string distance_method = "euclidean";
        int rnd_seed = 123456789;
        int cpu_threads = 6;

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        std::vector<std::vector<int> > clst = gda_maxp_greedy(w, data, "standardize", iterations,
                                                              min_bounds, max_bounds, init_regions, distance_method,
                                                              rnd_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double between_ss = gda_betweensumofsquare(clst, data);
        double ratio = between_ss / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.42329309419590377);

        delete w;
    }
}