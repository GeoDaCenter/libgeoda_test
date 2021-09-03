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
#include <libgeoda/geofeature.h>
#include <libgeoda/GenUtils.h>


using namespace testing;

namespace {

    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};

    TEST(CLUSTERING_TEST, MAKE_SPATIAL) {
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
        std::vector<std::vector<int> > clst = gda_skater(6, w, data, "standardize", "euclidean", bound_vals, bound, rand_seed, cpu_threads);
        int num_obs = w->GetNumObs();
        std::vector<int> cids = GenUtils::flat_2dclusters(num_obs, clst);

        std::vector<int> sc_cids = gda_makespatial(cids, w);

        EXPECT_EQ(sc_cids[0], cids[0]);
        EXPECT_EQ(sc_cids[1], cids[1]);
        EXPECT_EQ(sc_cids[2], cids[2]);
        EXPECT_EQ(sc_cids[3], cids[3]);

        delete w;
    }

    TEST(CLUSTERING_TEST, SPATIAL_VALIDATION) {
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
        std::vector<std::vector<int> > clst = gda_skater(6, w, data, "standardize", "euclidean", bound_vals, bound, rand_seed, cpu_threads);


        int num_obs = w->GetNumObs();
        std::vector<int> cids = GenUtils::flat_2dclusters(num_obs, clst);

        gda::ShapeType shape_type = gda::POLYGON;
        ValidationResult sv = gda_spatialvalidation(&gda, cids, w);

        bool spatially_constrained = sv.spatially_constrained;
        Fragmentation frag = sv.fragmentation;
        std::vector<Fragmentation> frags = sv.cluster_fragmentation;
        std::vector<Diameter> diams = sv.cluster_diameter;
        std::vector<Compactness> comps = sv.cluster_compactness;
        std::vector<JoinCountRatio> jcr = sv.joincount_ratio;

        EXPECT_TRUE(spatially_constrained);
        EXPECT_DOUBLE_EQ(frag.fraction, 0);
        EXPECT_DOUBLE_EQ(frag.entropy, 1.5302035777210896);
        EXPECT_DOUBLE_EQ(frag.std_entropy, 0.85402287751287753);
        EXPECT_DOUBLE_EQ(frag.simpson, 0.25619377162629758);
        EXPECT_DOUBLE_EQ(frag.std_simpson, 1.5371626297577856);

        EXPECT_EQ(diams[0].steps, 7);
        EXPECT_DOUBLE_EQ(diams[0].ratio, 0.2413793103448276);
        EXPECT_EQ(diams[1].steps, 7);
        EXPECT_DOUBLE_EQ(diams[1].ratio, 0.25);
        EXPECT_EQ(diams[2].steps, 4);
        EXPECT_DOUBLE_EQ(diams[2].ratio, 0.36363636363636365);
        EXPECT_EQ(diams[3].steps, 3);
        EXPECT_DOUBLE_EQ(diams[3].ratio, 0.375);
        EXPECT_EQ(diams[4].steps, 3);
        EXPECT_DOUBLE_EQ(diams[4].ratio, 0.6);
        EXPECT_EQ(diams[5].steps, 2);
        EXPECT_DOUBLE_EQ(diams[5].ratio, 0.5);

        EXPECT_EQ(comps[0].isoperimeter_quotient, 0.0097723523876562887);
        EXPECT_EQ(comps[0].area, 177914101737.5);
        EXPECT_EQ(comps[0].perimeter, 15125528.512594011);
        EXPECT_EQ(comps[1].isoperimeter_quotient, 0.0099144268567466272);
        EXPECT_EQ(comps[1].area, 164582498646);
        EXPECT_EQ(comps[1].perimeter, 14443184.236951336);
        EXPECT_EQ(comps[2].isoperimeter_quotient, 0.029675044913002577);
        EXPECT_EQ(comps[2].area, 72184135751);
        EXPECT_EQ(comps[2].perimeter, 5528790.326552554);
        EXPECT_EQ(comps[3].isoperimeter_quotient, 0.034800225315536358);
        EXPECT_EQ(comps[3].area, 50339473596);
        EXPECT_EQ(comps[3].perimeter, 4263519.3561323136);
        EXPECT_EQ(comps[4].isoperimeter_quotient, 0.046733291357011458);
        EXPECT_EQ(comps[4].area, 32318674158);
        EXPECT_EQ(comps[4].perimeter, 2947939.1554776165);
        EXPECT_EQ(comps[5].isoperimeter_quotient, 0.035828472477053515);
        EXPECT_EQ(comps[5].area, 27445319943.5);
        EXPECT_EQ(comps[5].perimeter, 3102593.9023676016);

        EXPECT_EQ(jcr[0].ratio, 0.8571428571428571);
        EXPECT_EQ(jcr[1].ratio, 0.89230769230769236);
        EXPECT_EQ(jcr[2].ratio, 0.58461538461538465);
        EXPECT_EQ(jcr[3].ratio, 0.54545454545454541);
        EXPECT_EQ(jcr[4].ratio, 0.38461538461538464);
        EXPECT_EQ(jcr[5].ratio, 0.66666666666666663);
        delete w;
    }

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

        EXPECT_DOUBLE_EQ(ratio, 0.44996710675020168);

        delete w;
    }
}