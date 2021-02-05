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
    const int rand_seed = 123456789;
    const int cpu_threads = 6;

    TEST(CLUSTERING_TEST, SKATER) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::vector<std::vector<int> > clst = gda_skater(4, w, data, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039);

        std::vector<std::vector<int> > clst_6 = gda_skater(6, w, data, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        withinss = gda_withinsumofsquare(clst_6, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.419541653761205);
        delete w;
    }

    TEST(CLUSTERING_TEST, REDCAP) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::string method = "firstorder-singlelinkage";
        std::vector<std::vector<int> > clst = gda_redcap(4, w, data, method, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039); // same as SKATER

        method = "fullorder-wardlinkage";
        std::vector<std::vector<int> > clst1 = gda_redcap(4, w, data, method, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        withinss = gda_withinsumofsquare(clst1, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.379025557025986);

        method = "fullorder-completelinkage";
        std::vector<std::vector<int> > clst2 = gda_redcap(4, w, data, method, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        withinss = gda_withinsumofsquare(clst2, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.35109901091774076);

        method = "fullorder-averagelinkage";
        std::vector<std::vector<int> > clst3 = gda_redcap(4, w, data, method, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        withinss = gda_withinsumofsquare(clst3, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.30578249025454063);

        method = "fullorder-singlelinkage";
        std::vector<std::vector<int> > clst4 = gda_redcap(4, w, data, method, "euclidean", std::vector<double>(), -1,
                rand_seed, cpu_threads);
        withinss = gda_withinsumofsquare(clst4, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.29002543000953057);

        delete w;
    }


    TEST(CLUSTERING_TEST, MAXP_GREEDY) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;

        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        int iterations = 99;
        std::vector<int> init_regions;

        std::vector<std::vector<int> > clst = gda_maxp_greedy(w, data, iterations, min_bounds, max_bounds,
                init_regions, "euclidean", rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.42329309419590377);

        delete w;
    }

    TEST(CLUSTERING_TEST, MAXP_SA) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;

        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        int iterations = 99;
        std::vector<int> init_regions;

        double cooling_rate = 0.85;
        int sa_maxit = 1;
        std::vector<std::vector<int> > clst = gda_maxp_sa(w, data, iterations, cooling_rate, sa_maxit,
                min_bounds, max_bounds, init_regions, "euclidean", rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.46832569616103947);

        delete w;
    }

    TEST(CLUSTERING_TEST, MAXP_TABU) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;

        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        int iterations = 99;
        std::vector<int> init_regions;

        int tabu_length = 10;
        int conv_tabu = 10;
        std::vector<std::vector<int> > clst = gda_maxp_tabu(w, data, iterations, tabu_length, conv_tabu,
                                                          min_bounds, max_bounds, init_regions, "euclidean",
                                                          rand_seed, cpu_threads);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.4893668149272537);

        delete w;
    }


    TEST(CLUSTERING_TEST, AZP_GREEDY) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;
        int p = 5;
        int inits = 0;
        std::vector<int> init_regions;

        std::vector<std::vector<int> > clst = gda_azp_greedy(p, w, data, inits, min_bounds, max_bounds,
                                                             init_regions, "euclidean", rand_seed);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.35985409831772192);

        // using min_bounds
        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        clst = gda_azp_greedy(p, w, data, inits, min_bounds, max_bounds,
                              init_regions, "euclidean", rand_seed);

        totalss = gda_totalsumofsquare(data);
        withinss = gda_withinsumofsquare(clst, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.37015555244290943);

        delete w;
    }

    TEST(CLUSTERING_TEST, AZP_SA) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;


        int p = 5;
        int inits = 0;
        double cooling_rate = 0.85;
        int sa_maxit = 1;
        std::vector<int> init_regions;

        std::vector<std::vector<int> > clst = gda_azp_sa(p, w, data, inits, cooling_rate, sa_maxit, min_bounds,
                max_bounds, init_regions, "euclidean", rand_seed);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.4302644093198012);

        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        clst = gda_azp_sa(p, w, data, inits, cooling_rate, sa_maxit, min_bounds,
                                                         max_bounds, init_regions, "euclidean", rand_seed);
        totalss = gda_totalsumofsquare(data);
        withinss = gda_withinsumofsquare(clst, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.39917732725352167);

        delete w;
    }

    TEST(CLUSTERING_TEST, AZP_TABU) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }

        std::vector<std::pair<double, std::vector<double> > > min_bounds, max_bounds;


        int p = 5;
        int inits = 0;
        int tabu_length = 10;
        int conv_tabu = 10;
        std::vector<int> init_regions;

        std::vector<std::vector<int> > clst = gda_azp_tabu(p, w, data, inits, tabu_length, conv_tabu, min_bounds,
                                                         max_bounds, init_regions, "euclidean", rand_seed);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.42221739641363148);

        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        min_bounds.push_back(std::make_pair(min_bound, bound_vals));

        clst = gda_azp_tabu(p, w, data, inits, tabu_length, conv_tabu, min_bounds,
                            max_bounds, init_regions, "euclidean", rand_seed);
        totalss = gda_totalsumofsquare(data);
        withinss = gda_withinsumofsquare(clst, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.42314746403476627);

        delete w;
    }
}