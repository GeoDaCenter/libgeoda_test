//
// Created by Xun Li on 2019-06-06.
//

#include <vector>
#include <limits.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/sa/LISA.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_sa.h>

using namespace testing;

namespace {

    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};

    TEST(LOCALSA_TEST, LISA_FDR) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* lisa = gda_localmoran(w, data);

        double fdr = gda_fdr(lisa, 0.05);
        double bo = gda_bo(lisa, 0.05);

        delete lisa;

        EXPECT_DOUBLE_EQ(fdr, 0.0041176470588235297);
        EXPECT_DOUBLE_EQ(bo, 0.00058823529411764712);
    }

    TEST(LOCALSA_TEST, JOINCOUNT_MULTI) {
        GeoDa gda("../../data/chicago_comm.shp");
        GeoDaWeight *w = gda_queen_weights(&gda);

        std::vector<std::vector<double> > data;
        data.push_back(gda.GetNumericCol("popneg"));
        data.push_back(gda.GetNumericCol("popplus"));

        LISA* jc = gda_multijoincount(w, data);

        std::vector<int> nnvals = jc->GetNumNeighbors();
        std::vector<double> pvals = jc->GetLocalSignificanceValues();
        std::vector<double> gvals = jc->GetLISAValues();
        delete jc;

        EXPECT_DOUBLE_EQ(gvals[0], 2);
        EXPECT_DOUBLE_EQ(gvals[1], 0);
        EXPECT_DOUBLE_EQ(gvals[2], 1);

        EXPECT_THAT(nnvals[0], 4);
        EXPECT_THAT(nnvals[1], 3);
        EXPECT_THAT(nnvals[2], 6);

        EXPECT_DOUBLE_EQ(pvals[0], 0.21299999999999999);
        EXPECT_DOUBLE_EQ(pvals[1], 0);
        EXPECT_DOUBLE_EQ(pvals[2], 0.20000000000000001);
        EXPECT_DOUBLE_EQ(pvals[3], 0.156);
    }

    TEST(LOCALSA_TEST, GEARY_MULTI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight *w = gda_queen_weights(&gda);
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        LISA* geary = gda_multigeary(w, data);

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;

        EXPECT_DOUBLE_EQ(gvals[0], 2.5045545811329406);
        EXPECT_DOUBLE_EQ(gvals[1], 0.3558770845279205);
        EXPECT_DOUBLE_EQ(gvals[2], 1.872894936446803);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 1);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.29399999999999998);
        EXPECT_DOUBLE_EQ(pvals[1], 0.001);
        EXPECT_DOUBLE_EQ(pvals[2], 0.014);
    }

    TEST(LOCALSA_TEST, LISA_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* lisa = gda_localmoran(w, data);

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;

        EXPECT_DOUBLE_EQ(mvals[0], 0.015431978309803657);
        EXPECT_DOUBLE_EQ(mvals[1], 0.32706332236560332);
        EXPECT_DOUBLE_EQ(mvals[2], 0.021295296214118884);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.41399999999999998);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, GEARY_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* geary = gda_geary(w, data);

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;

        EXPECT_DOUBLE_EQ(gvals[0], 7.3980833011783602);
        EXPECT_DOUBLE_EQ(gvals[1], 0.28361195650519017);
        EXPECT_DOUBLE_EQ(gvals[2], 3.6988922226329906);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 2);
        EXPECT_THAT(cvals[2], 4);

        EXPECT_DOUBLE_EQ(pvals[0], 0.39800000000000002);
        EXPECT_DOUBLE_EQ(pvals[1], 0.027);
        EXPECT_DOUBLE_EQ(pvals[2], 0.025000000000000001);
    }

    TEST(LOCALSA_TEST, JOINCOUNT_UNI) {
        GeoDa gda("../../data/columbus.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("nsa");

        LISA* jc = gda_joincount(w, data);

        std::vector<int> nnvals = jc->GetNumNeighbors();
        std::vector<double> pvals = jc->GetLocalSignificanceValues();
        std::vector<double> jvals = jc->GetLISAValues();
        delete jc;

        EXPECT_DOUBLE_EQ(jvals[0], 2);
        EXPECT_DOUBLE_EQ(jvals[1], 3);
        EXPECT_DOUBLE_EQ(jvals[2], 4);

        EXPECT_THAT(nnvals[0], 2);
        EXPECT_THAT(nnvals[1], 3);
        EXPECT_THAT(nnvals[2], 4);

        EXPECT_DOUBLE_EQ(pvals[0], 0.21299999999999999);
        EXPECT_DOUBLE_EQ(pvals[1], 0.070000000000000007);
        EXPECT_DOUBLE_EQ(pvals[2], 0.017000000000000001);
    }

    TEST(LOCALSA_TEST, LOCALG_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localg = gda_localg(w, data);

        std::vector<int> cvals = localg->GetClusterIndicators();
        std::vector<double> pvals = localg->GetLocalSignificanceValues();
        std::vector<double> gvals = localg->GetLISAValues();
        delete localg;

        EXPECT_DOUBLE_EQ(gvals[0], 0.012077920687925825);
        EXPECT_DOUBLE_EQ(gvals[1], 0.0099240961298508561);
        EXPECT_DOUBLE_EQ(gvals[2], 0.018753584525825453);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.414);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, LOCALGstar_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localgstar = gda_localgstar(w, data);

        std::vector<int> cvals = localgstar->GetClusterIndicators();
        std::vector<double> pvals = localgstar->GetLocalSignificanceValues();
        std::vector<double> gvals = localgstar->GetLISAValues();
        delete localgstar;

        EXPECT_DOUBLE_EQ(gvals[0], 0.014177043620524426);
        EXPECT_DOUBLE_EQ(gvals[1], 0.0096136007223101994);
        EXPECT_DOUBLE_EQ(gvals[2], 0.017574324039034434);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.414);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

}