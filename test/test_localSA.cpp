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

    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};
    double significance_cutoff = 0.05;
    int nCPUs = 6;
    int permutations = 999;
    int last_seed_used = 123456789;

    TEST(LOCALSA_TEST, LOCALMORAN_EB) {
        GeoDa gda("../../data/natregimes.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> hr = gda.GetNumericCol("HR60");
        std::vector<double> pop = gda.GetNumericCol("PO60");

        LISA* lisa = gda_localmoran_eb(w, hr, pop, significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;
        delete w;
        EXPECT_DOUBLE_EQ(mvals[0], 0.03556859723358851);
        EXPECT_DOUBLE_EQ(mvals[1], 0.023622877901122327);

        EXPECT_DOUBLE_EQ(pvals[0], 0.155);
        EXPECT_DOUBLE_EQ(pvals[1], 0.48599999999999999);

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

    TEST(LOCALSA_TEST, BATCH_LOCALMORAN) {
        GeoDa gda("../../data/natregimes.shp");
        const char *cols[24] = {
                "HR60", "HR70", "HR80", "HR90",
                "HC60", "HC70", "HC80", "HC90",
                "PO60", "PO70", "PO80", "PO90",
                "RD60", "RD70", "RD80", "RD90",
                "PS60", "PS70", "PS80", "PS90",
                "UE60", "UE70", "UE80", "UE90"
        };
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 24; ++i) {
            data.push_back(gda.GetNumericCol(cols[i]));
        }
        clock_t t = clock();
        BatchLISA* bm = gda_batchlocalmoran(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs, permutations, last_seed_used);
        const double work_time = (clock() - t) / double(CLOCKS_PER_SEC);
        std::vector<int> cvals = bm->GetClusterIndicators(0);
        std::vector<double> pvals = bm->GetLocalSignificanceValues(0);
        std::vector<double> mvals = bm->GetLISAValues(0);
        delete bm;
        delete w;
        /*
        EXPECT_DOUBLE_EQ(mvals[0], 0.015431978309803657);
        EXPECT_DOUBLE_EQ(mvals[1], 0.32706332236560332);
        EXPECT_DOUBLE_EQ(mvals[2], 0.021295296214118884);

        EXPECT_THAT(cvals[0], 0);
        EXPECT_THAT(cvals[1], 0);
        EXPECT_THAT(cvals[2], 1);

        EXPECT_DOUBLE_EQ(pvals[0], 0.41399999999999998);
        EXPECT_DOUBLE_EQ(pvals[1], 0.123);
        EXPECT_DOUBLE_EQ(pvals[2], 0.001);
         */
    }

    TEST(LOCALSA_TEST, LISA_FDR) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Donatns");
        LISA* lisa = gda_localmoran(w, data,  std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);
        double fdr = gda_fdr(lisa, 0.01);

        EXPECT_DOUBLE_EQ(fdr, 0.00011764705882352942);

        delete lisa;

        GeoDa gda1("../../data/columbus.shp");
        GeoDaWeight* w1 = gda_queen_weights(&gda1, 1, false, 0);
        std::vector<double> data1 = gda1.GetNumericCol("nsa");

        LISA* lisa1 = gda_localmoran(w1, data1, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

        double fdr005 = gda_fdr(lisa1, 0.05);
        double fdr001 = gda_fdr(lisa1, 0.01);
        double bo = gda_bo(lisa1, 0.05);

        delete lisa1;

        delete w;
        delete w1;

        EXPECT_DOUBLE_EQ(fdr005, 0.012244897959183675);
        EXPECT_DOUBLE_EQ(fdr001, 0.00020408163265306123);
        EXPECT_DOUBLE_EQ(bo, 0.0010204081632653062);
    }

    TEST(LOCALSA_TEST, JOINCOUNT_MULTI) {
        GeoDa gda("../../data/chicago_comm.shp");
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);

        std::vector<std::vector<double> > data;
        data.push_back(gda.GetNumericCol("popneg"));
        data.push_back(gda.GetNumericCol("popplus"));

        LISA* jc = gda_localmultijoincount(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs, permutations, last_seed_used);

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

        EXPECT_DOUBLE_EQ(pvals[0], 0.21299999999999999);
        EXPECT_DOUBLE_EQ(pvals[1], 0);
        EXPECT_DOUBLE_EQ(pvals[2], 0.20000000000000001);
        EXPECT_DOUBLE_EQ(pvals[3], 0.156);
    }

    TEST(LOCALSA_TEST, GEARY_MULTI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight *w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        LISA* geary = gda_localmultigeary(w, data, std::vector<std::vector<bool> >(), significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;
        delete w;

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
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* lisa = gda_localmoran(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals= lisa->GetClusterIndicators();
        std::vector<double> pvals = lisa->GetLocalSignificanceValues();
        std::vector<double> mvals = lisa->GetLISAValues();
        delete lisa;
        delete w;

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
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* geary = gda_localgeary(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals = geary->GetClusterIndicators();
        std::vector<double> pvals = geary->GetLocalSignificanceValues();
        std::vector<double> gvals = geary->GetLISAValues();
        delete geary;
        delete w;

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

    TEST(LOCALSA_TEST, QUANTILE_LISA) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");
        int k=7;
        int quantile = 7;
        LISA* ql = gda_quantilelisa(w, k, quantile, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

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
        EXPECT_DOUBLE_EQ(pvals[1], 0.0);
        EXPECT_DOUBLE_EQ(pvals[2], 0.0);
    }



    TEST(LOCALSA_TEST, LOCALGstar_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localgstar = gda_localgstar(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals = localgstar->GetClusterIndicators();
        std::vector<double> pvals = localgstar->GetLocalSignificanceValues();
        std::vector<double> gvals = localgstar->GetLISAValues();
        delete localgstar;
        delete w;

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

    TEST(LOCALSA_TEST, JOINCOUNT_UNI) {
        GeoDa gda("../../data/deaths_nd_by_house.shp");
        GeoDaWeight* w = gda_distance_weights(&gda, 20, "", 1.0, false, false, false, "", false);
        std::vector<double> data = gda.GetNumericCol("death_dum");

        significance_cutoff = 0.01;
        permutations = 99999;
        LISA* jc = gda_localjoincount(w, data, std::vector<bool>(),significance_cutoff, nCPUs, permutations, last_seed_used);

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

        EXPECT_DOUBLE_EQ(pvals[0], 0.0);
        EXPECT_DOUBLE_EQ(pvals[1], 0.325560);
        EXPECT_DOUBLE_EQ(pvals[2], 0.0);
    }

    TEST(LOCALSA_TEST, LOCALG_UNI) {
        GeoDa gda("../../data/Guerry.shp");
        GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
        std::vector<double> data = gda.GetNumericCol("Crm_prp");

        LISA* localg = gda_localg(w, data, std::vector<bool>(), significance_cutoff, nCPUs, permutations, last_seed_used);

        std::vector<int> cvals = localg->GetClusterIndicators();
        std::vector<double> pvals = localg->GetLocalSignificanceValues();
        std::vector<double> gvals = localg->GetLISAValues();
        delete localg;
        delete w;

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
        LISA* jc = gda_localmultijoincount(w, data, undefs,significance_cutoff, nCPUs, permutations, last_seed_used);

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

        EXPECT_DOUBLE_EQ(pvals[0], 0.0);
        EXPECT_DOUBLE_EQ(pvals[1], 0.25645);
        EXPECT_DOUBLE_EQ(pvals[2], 0.0);
    }
}