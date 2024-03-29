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
#include <libgeoda/sa/BatchLISA.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_sa.h>

using namespace testing;

namespace {

    const double PRECISION_THRESHOLD = 1e-6;
    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};
    double significance_cutoff = 0.05;
    int nCPUs = 6;
    int permutations = 999;
    const std::string permutation_method = "complete";
    int last_seed_used = 123456789;

    TEST(LOCALSA_TEST, LOCALMORAN) {
        GeoDa gda("../../data/natregimes.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> hr = gda.GetNumericCol("HR60");
        std::vector<bool> undefs;

        LISA* lisa = gda_localmoran(w, hr, undefs, significance_cutoff, nCPUs, permutations, permutation_method,
                                    last_seed_used);
        LISA* lisa1 = gda_localmoran(w, hr, undefs, significance_cutoff, nCPUs, permutations, "lookup-table",
                                     last_seed_used);
        delete lisa1;

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;
        delete w;
        EXPECT_NEAR(mvals[0], 0.50169822326995339, PRECISION_THRESHOLD);
        EXPECT_NEAR(mvals[1], 0.28142804448894865, PRECISION_THRESHOLD);

        EXPECT_DOUBLE_EQ(pvals[0], 0.084);
        EXPECT_DOUBLE_EQ(pvals[1], 0.304);

        EXPECT_DOUBLE_EQ(cvals[0], 0);
        EXPECT_DOUBLE_EQ(cvals[1], 0);
    }

    TEST(LOCALSA_TEST, LOCALMORAN_BI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data1 = gda.GetNumericCol("Crm_prs");
        std::vector<double> data2 = gda.GetNumericCol("Litercy");

        LISA* lisa = gda_bi_localmoran(w, data1, data2, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);


        std::vector<int> cvals = lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> lvals = lisa->GetLISAValues();
        delete lisa;
        delete w;

        EXPECT_NEAR(lvals[0], 0.39266344763810573, PRECISION_THRESHOLD);
        EXPECT_NEAR(lvals[1], 0.75613610603433934, PRECISION_THRESHOLD);
        EXPECT_NEAR(lvals[2], -0.87851057571266755, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 1);
        EXPECT_THAT(cvals[2], 4);

        EXPECT_DOUBLE_EQ(pvals[0], 0.269);
        EXPECT_DOUBLE_EQ(pvals[1], 0.021);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, LOCALMORAN_EB) {
        GeoDa gda("../../data/natregimes.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> hr = gda.GetNumericCol("HR60");
        std::vector<double> pop = gda.GetNumericCol("PO60");

        LISA* lisa = gda_localmoran_eb(w, hr, pop, significance_cutoff, nCPUs, permutations, permutation_method,
                last_seed_used);

        // test lookup-table
        LISA* lisa_1 = gda_localmoran_eb(w, hr, pop, significance_cutoff, nCPUs, permutations, "lookup-table",
                last_seed_used);
        delete lisa_1;

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;
        delete w;
        EXPECT_NEAR(mvals[0], 0.03556859723358851, PRECISION_THRESHOLD);
        EXPECT_NEAR(mvals[1], 0.023622877901122327, PRECISION_THRESHOLD);

        EXPECT_NEAR(pvals[0], 0.155, PRECISION_THRESHOLD);
        EXPECT_NEAR(pvals[1], 0.48599999999999999, PRECISION_THRESHOLD);

        EXPECT_DOUBLE_EQ(cvals[0], 0);
        EXPECT_DOUBLE_EQ(cvals[1], 0);
    }

    TEST(LOCALSA_TEST, NEIGHBOR_MATCH_TEST) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        std::vector<std::vector<double> > result = gda_neighbor_match_test(&gda, 6, 1.0, false, false, false, data, "standardize", "euclidean");

        delete w;
    }

    TEST(LOCALSA_TEST, LOCALG_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localg = gda_localg(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        LISA* localg1 = gda_localg(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations,
                "lookup-table", last_seed_used);
        delete localg1;

        std::vector<int> cvals = localg->GetClusterIndicators();
        std::vector<double> pvals = localg->GetLocalSignificanceValues();
        std::vector<double> gvals = localg->GetLISAValues();
        delete localg;
        delete w;

        EXPECT_NEAR(gvals[0], 0.012077920687925825, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[1], 0.0099240961298508561, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[2], 0.018753584525825453, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.414);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, LISA_FDR) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Donatns");
        LISA* lisa = gda_localmoran(w, data,  std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);
        double fdr = gda_fdr(lisa, 0.01);

        EXPECT_NEAR(fdr, 0.00011764705882352942, PRECISION_THRESHOLD);

        delete lisa;

        GeoDa gda1("../../data/columbus.shp");
        GeoDaWeight* w1 = gda_queen_weights(&gda1, 1, false, 0);
        std::vector<double> data1 = gda1.GetNumericCol("nsa");

        LISA* lisa1 = gda_localmoran(w1, data1, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        double fdr005 = gda_fdr(lisa1, 0.05);
        double fdr001 = gda_fdr(lisa1, 0.01);
        double bo = gda_bo(lisa1, 0.05);

        delete lisa1;

        delete w;
        delete w1;

        EXPECT_NEAR(fdr005, 0.012244897959183675, PRECISION_THRESHOLD);
        EXPECT_NEAR(fdr001, 0.00020408163265306123, PRECISION_THRESHOLD);
        EXPECT_NEAR(bo, 0.0010204081632653062, PRECISION_THRESHOLD);
    }

    TEST(LOCALSA_TEST, JOINCOUNT_MULTI) {
        GeoDa gda("../../data/chicago_comm.shp");
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);

        std::vector<std::vector<double> > data;
        data.push_back(gda.GetNumericCol("popneg"));
        data.push_back(gda.GetNumericCol("popplus"));

        LISA* jc = gda_localmultijoincount(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        LISA* jc1 = gda_localmultijoincount(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs,
                permutations, "lookup-table", last_seed_used);
        delete jc1;

        std::vector<int> nnvals = jc->GetNumNeighbors();
        std::vector<double> pvals = jc->GetLocalSignificanceValues();
        std::vector<double> gvals = jc->GetLISAValues();
        delete jc;
        delete w;

        EXPECT_DOUBLE_EQ(gvals[0], 2);
        EXPECT_DOUBLE_EQ(gvals[1], 0);
        EXPECT_DOUBLE_EQ(gvals[2], 1);

        EXPECT_THAT(nnvals[0], 4);
        EXPECT_THAT(nnvals[1], 3);
        EXPECT_THAT(nnvals[2], 6);

        EXPECT_NEAR(pvals[0], 0.21299999999999999, PRECISION_THRESHOLD);
        EXPECT_DOUBLE_EQ(pvals[1], -1);
        EXPECT_NEAR(pvals[2], 0.20000000000000001, PRECISION_THRESHOLD);
        EXPECT_DOUBLE_EQ(pvals[3], 0.156);
    }

    TEST(LOCALSA_TEST, GEARY_MULTI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        LISA* geary = gda_localmultigeary(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        LISA* geary1 = gda_localmultigeary(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs,
                permutations, "lookup-table", last_seed_used);
        delete geary1;

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;
        delete w;

        EXPECT_NEAR(gvals[0], 2.5045545811329406, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[1], 0.3558770845279205, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[2], 1.872894936446803, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 1);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_NEAR(pvals[0], 0.29399999999999998, PRECISION_THRESHOLD);
        EXPECT_DOUBLE_EQ(pvals[1], 0.001);
        EXPECT_DOUBLE_EQ(pvals[2], 0.014);
    }

    TEST(LOCALSA_TEST, LISA_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* lisa = gda_localmoran(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;
        delete w;

        EXPECT_NEAR(mvals[0], 0.015431978309803657, PRECISION_THRESHOLD);
        EXPECT_NEAR(mvals[1], 0.32706332236560332, PRECISION_THRESHOLD);
        EXPECT_NEAR(mvals[2], 0.021295296214118884, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_NEAR(pvals[0], 0.41399999999999998, PRECISION_THRESHOLD);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, GEARY_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* geary = gda_localgeary(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        LISA* geary1 = gda_localgeary(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations,
                "lookup-table", last_seed_used);
        delete geary1;

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;
        delete w;

        EXPECT_NEAR(gvals[0], 7.3980833011783602, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[1], 0.28361195650519017, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[2], 3.6988922226329906, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 2);
        EXPECT_THAT(cvals[2], 4);

        EXPECT_NEAR(pvals[0], 0.39800000000000002, PRECISION_THRESHOLD);
        EXPECT_DOUBLE_EQ(pvals[1], 0.027);
        EXPECT_NEAR(pvals[2], 0.025000000000000001, PRECISION_THRESHOLD);
    }

    TEST(LOCALSA_TEST, QUANTILE_LISA) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");
        int k=7;
        int quantile = 7;
        LISA* ql = gda_quantilelisa(w, k, quantile, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);
        LISA* ql1 = gda_quantilelisa(w, k, quantile, data, std::vector<bool>(), significance_cutoff, nCPUs,
                permutations, "lookup-table", last_seed_used);
        delete ql1;

        std::vector<int> cvals = ql->GetClusterIndicators();
        std::vector<double> pvals = ql->GetLocalSignificanceValues();
        std::vector<double> jc = ql->GetLISAValues();
        delete ql;
        delete w;

        EXPECT_DOUBLE_EQ(jc[0], 1);
        EXPECT_DOUBLE_EQ(jc[1], 0);
        EXPECT_DOUBLE_EQ(jc[2], 0);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 0);

        EXPECT_DOUBLE_EQ(pvals[0], 0.434);
        EXPECT_DOUBLE_EQ(pvals[1], -1);
        EXPECT_DOUBLE_EQ(pvals[2], -1);
    }



    TEST(LOCALSA_TEST, LOCALGstar_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localgstar = gda_localgstar(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);

        LISA* localgstar1 = gda_localgstar(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations,
                "lookup-table", last_seed_used);
        delete localgstar1;

        std::vector<int> cvals = localgstar->GetClusterIndicators();
        std::vector<double> pvals = localgstar->GetLocalSignificanceValues();
        std::vector<double> gvals = localgstar->GetLISAValues();
        delete localgstar;
        delete w;

        EXPECT_NEAR(gvals[0], 0.014177043620524426, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[1], 0.0096136007223101994, PRECISION_THRESHOLD);
        EXPECT_NEAR(gvals[2], 0.017574324039034434, PRECISION_THRESHOLD);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.414);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
    }

    TEST(LOCALSA_TEST, JOINCOUNT_UNI) {
        GeoDa gda("../../data/deaths_nd_by_house.shp");
        GeoDaWeight* w = gda_distance_weights(&gda, 20, "", 1.0, false, false, false, "", false);
        std::vector<double> data = gda.GetNumericCol("death_dum");

        significance_cutoff = 0.01;
        permutations = 99999;

        LISA* jc = gda_localjoincount(w, data, std::vector<bool>(),significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);
        LISA* jc1 = gda_localjoincount(w, data, std::vector<bool>(),significance_cutoff, nCPUs, permutations,
                "lookup-table", last_seed_used);
        delete jc1;

        std::vector<int> nnvals = jc->GetNumNeighbors();
        std::vector<double> pvals = jc->GetLocalSignificanceValues();
        std::vector<double> jvals = jc->GetLISAValues();
        delete jc;
        delete w;

        EXPECT_DOUBLE_EQ(jvals[0], 0);
        EXPECT_DOUBLE_EQ(jvals[1], 1);
        EXPECT_DOUBLE_EQ(jvals[2], 0);

        EXPECT_THAT(nnvals[0], 4);
        EXPECT_THAT(nnvals[1], 5);
        EXPECT_THAT(nnvals[2], 5);

        EXPECT_DOUBLE_EQ(pvals[0], -1);
        EXPECT_DOUBLE_EQ(pvals[1], 0.325560);
        EXPECT_DOUBLE_EQ(pvals[2], -1);
    }



    TEST(LOCALSA_TEST, JOINCOUNT_BI) {
        GeoDa gda("../../data/deaths_nd_by_house.shp");
        GeoDaWeight* w = gda_distance_weights(&gda, 20, "", 1.0, false, false, false, "", false);
        std::vector<double> death = gda.GetNumericCol("death_dum");
        std::vector<double> death_iv = death;
        for (size_t i=0; i<death.size(); ++i) death_iv[i] = 1-death_iv[i];
        std::vector<std::vector<bool> > undefs;
        std::vector<std::vector<double> > data = {death, death_iv};
        significance_cutoff = 0.01;
        permutations = 99999;
        LISA* jc = gda_localmultijoincount(w, data, undefs,significance_cutoff, nCPUs, permutations, permutation_method, last_seed_used);
        LISA* jc1 = gda_localmultijoincount(w, data, undefs,significance_cutoff, nCPUs, permutations,
                "lookup-table", last_seed_used);
        delete jc1;

        std::vector<int> nnvals = jc->GetNumNeighbors();
        std::vector<double> pvals = jc->GetLocalSignificanceValues();
        std::vector<double> jvals = jc->GetLISAValues();
        delete jc;
        delete w;

        EXPECT_DOUBLE_EQ(jvals[0], 0);
        EXPECT_DOUBLE_EQ(jvals[1], 4);
        EXPECT_DOUBLE_EQ(jvals[2], 0);

        EXPECT_THAT(nnvals[0], 4);
        EXPECT_THAT(nnvals[1], 5);
        EXPECT_THAT(nnvals[2], 5);

        EXPECT_DOUBLE_EQ(pvals[0], -1);
        EXPECT_DOUBLE_EQ(pvals[1], 0.25645);
        EXPECT_DOUBLE_EQ(pvals[2], -1);
    }
}