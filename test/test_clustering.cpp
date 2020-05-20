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
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::vector<std::vector<int> > clst = gda_skater(4, w, data);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039);

        std::vector<std::vector<int> > clst_6 = gda_skater(6, w, data);
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
        std::vector<std::vector<int> > clst = gda_redcap(4, w, data, method);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.31564466593112039); // same as SKATER

        method = "fullorder-completelinkage";
        std::vector<std::vector<int> > clst1 = gda_redcap(4, w, data, method);
        withinss = gda_withinsumofsquare(clst1, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.25712705781933565);

        method = "fullorder-completelinkage";
        std::vector<std::vector<int> > clst2 = gda_redcap(5, w, data, method);
        withinss = gda_withinsumofsquare(clst2, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.32264464163898959);

        method = "fullorder-averagelinkage";
        std::vector<std::vector<int> > clst3 = gda_redcap(4, w, data, method);
        withinss = gda_withinsumofsquare(clst3, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.30578249025454063);

        method = "fullorder-singlelinkage";
        std::vector<std::vector<int> > clst4 = gda_redcap(4, w, data, method);
        withinss = gda_withinsumofsquare(clst4, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.3157257642930576);

        delete w;
    }

    TEST(CLUSTERING_TEST, MAXP_GREEDY) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i=0; i<6; ++i) {
            data.push_back( gda.GetNumericCol(col_names[i]) );
        }
        std::vector<double> bound_vals = gda.GetNumericCol("Pop1831");
        double min_bound = 3236.6700000000001; // 10% of Pop1831

        int initial = 99;
        std::vector<std::vector<int> > clst = gda_maxp(w, data, bound_vals, min_bound, "greedy", initial);
        double totalss = gda_totalsumofsquare(data);
        double withinss = gda_withinsumofsquare(clst, data);
        double ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.50701807973320201);

        initial = 99;
        std::vector<std::vector<int> > clst1 = gda_maxp(w, data, bound_vals, min_bound, "tabu", initial, 85);
        withinss = gda_withinsumofsquare(clst1, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.52602489024178434);

        initial = 999;
        std::vector<std::vector<int> > clst2 = gda_maxp(w, data, bound_vals, min_bound, "greedy", initial);
        withinss = gda_withinsumofsquare(clst2, data);
        ratio = (totalss - withinss) / totalss;

        EXPECT_DOUBLE_EQ(ratio, 0.52602489024178434);

        delete w;
    }
}