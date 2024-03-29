//
// Created by Xun Li on 2019-06-04.
//

#include <limits.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/gda_weights.h>

using namespace testing;

namespace {

    TEST(WEIGHTS_TEST, QUEEN_CREATE) {
        GeoDa gda("../../data/natregimes.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);

        EXPECT_THAT(w->num_obs, 3085);
        EXPECT_THAT(w->GetMinNbrs(), 1);
        EXPECT_THAT(w->GetMaxNbrs(), 14);
        EXPECT_TRUE(w->is_symmetric);
        EXPECT_DOUBLE_EQ(w->GetSparsity(), 0.0019089598070866245);
        EXPECT_DOUBLE_EQ(w->GetMeanNbrs(), 5.8891410048622364);

        delete w;
    }

    TEST(WEIGHTS_TEST, ROOK_CREATE) {
        GeoDa gda("../../data/natregimes.shp");
        GeoDaWeight* w = gda_rook_weights(&gda, 1, false, 0);

        EXPECT_THAT(w->num_obs, 3085);
        EXPECT_THAT(w->GetMinNbrs(), 1);
        EXPECT_THAT(w->GetMaxNbrs(), 13);
        EXPECT_TRUE(w->is_symmetric);
        EXPECT_DOUBLE_EQ(w->GetSparsity(), 0.0018059886153789576);
        EXPECT_DOUBLE_EQ(w->GetMeanNbrs(), 5.571474878444084);

        delete w;
    }

    TEST(WEIGHTS_TEST, KNN_CREATE) {
        GeoDa gda("../../data/natregimes.shp");

        GeoDaWeight* w = gda_knn_weights(&gda, 4, 1.0, false, false, false, "", 0.0, false, false, "");

        EXPECT_FALSE(w->is_symmetric);
        EXPECT_THAT(w->num_obs, 3085);
        EXPECT_THAT(w->GetMinNbrs(), 4);
        EXPECT_THAT(w->GetMaxNbrs(), 4);
        EXPECT_DOUBLE_EQ(w->GetSparsity(), 0.0012965964343598055);
        EXPECT_DOUBLE_EQ(w->GetMeanNbrs(), 4);

        delete w;
    }

    TEST(WEIGHTS_TEST, DIST_CREATE) {
        GeoDa gda("../../data/natregimes.shp");
        double min_thres = gda_min_distthreshold(&gda, false, false);

        EXPECT_NEAR(min_thres, 1.4657759325950015, 1e-6);

        GeoDaWeight* w = gda_distance_weights(&gda, min_thres, "", 1.0, false, false, false, "", false);

        EXPECT_TRUE(w->is_symmetric);
        EXPECT_THAT(w->num_obs, 3085);
        EXPECT_THAT(w->GetMinNbrs(), 1);
        EXPECT_THAT(w->GetMaxNbrs(), 85);
        EXPECT_DOUBLE_EQ(w->GetSparsity(), 0.011939614751148575);
        EXPECT_DOUBLE_EQ(w->GetMeanNbrs(), 36.833711507293351);

        delete w;
    }

    TEST(WEIGHTS_TEST, KERNEL_KNN) {
        GeoDa gda("../../data/natregimes.shp");

        double power = 1;
        bool is_inverse = false;
        bool is_arc = false;
        bool is_mile = true;
        std::string kernel = "triangular";
        double bandwidth  = 0;
        bool adaptive_bandwidth = true;
        bool use_kernel_diagonals = false;
        int k = 15;
        GeoDaWeight* w = gda_knn_weights(&gda, k, power, is_inverse,
                is_arc, is_mile, kernel, bandwidth,
                adaptive_bandwidth, use_kernel_diagonals, "");

        EXPECT_FALSE(w->is_symmetric);
        EXPECT_THAT(w->num_obs, 3085);
        EXPECT_THAT(w->GetMinNbrs(), 15);
        EXPECT_THAT(w->GetMaxNbrs(), 15);
        EXPECT_DOUBLE_EQ(w->GetSparsity(), 0.0048622366288492708);
        EXPECT_DOUBLE_EQ(w->GetMeanNbrs(), 15);

        delete w;
    }
}