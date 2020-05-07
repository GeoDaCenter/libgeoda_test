//
// Created by Xun Li on 2019-11-30.
//


#include <vector>
#include <limits.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <libgeoda/libgeoda.h>
#include <libgeoda/gda_algorithms.h>
#include <libgeoda/gda_data.h>

using namespace testing;

namespace {

    const char *col_names[6] = {"Crm_prs", "Crm_prp", "Litercy", "Donatns", "Infants", "Suicids"};

    TEST(ALGORITHMS_TEST, PCA) {
        GeoDa gda("../../data/Guerry.shp");
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        data = gda_standardize(data);
        PCAResult* result = gda_pca(data, "svd");

        EXPECT_STRCASEEQ(result->getMethod().c_str(), "svd");
        EXPECT_THAT(result->getStandardDev(),
                ElementsAre(1.46303403, 1.09581947, 1.04978454, 0.816680014, 0.740725815, 0.583970726));
        EXPECT_THAT(result->getPropOfVar(),
                    ElementsAre(0.356744826, 0.200136751, 0.183674619, 0.111161061, 0.0914457887, 0.0568369664));
        EXPECT_THAT(result->getCumProp(),
                    ElementsAre(0.356744826, 0.556881547, 0.74055618, 0.851717234, 0.943163037, 1.000000));
        EXPECT_FLOAT_EQ(result->getKaiser(), 3.0);
        EXPECT_FLOAT_EQ(result->getThresh95(), 5.0);
        EXPECT_THAT(result->getEigenValues(),
                    ElementsAre(2.1404686, 1.20082033, 1.10204756, 0.666966259, 0.548674762, 0.341021806));

        std::vector<std::vector<float> > loadings = result->getLoadings();
        EXPECT_THAT(loadings[0],
                    ElementsAre(-0.0658684447,  -0.512325525, 0.511752903, -0.106195144, -0.451337427, -0.506270468));

        std::vector<std::vector<float> > sqcorr = result->getSqCorrelations();
        EXPECT_THAT(sqcorr[0],
                    ElementsAre(0.00928681157, 0.418853581, 0.499427617, 0.0130218817, 0.0000568266842, 0.0593530759));

        std::vector<std::vector<float> > pcs = result->getPriComponents();
        EXPECT_FLOAT_EQ(pcs[0][0], -2.15079784);
        EXPECT_FLOAT_EQ(pcs[0][1], 1.24630284);
        EXPECT_FLOAT_EQ(pcs[0][2], -2.07714891);

        delete result;
    }

    TEST(ALGORITHMS_TEST, MDS) {
        GeoDa gda("../../data/Guerry.shp");
        std::vector<std::vector<double> > data;
        for (size_t i = 0; i < 6; ++i) {
            data.push_back(gda.GetNumericCol(col_names[i]));
        }
        data = gda_standardize(data);
        std::vector<std::vector<double>> result = gda_mds(data, 2, "euclidean");

        //Classical (Torgerson) MDS on Euclidean distances is equivalent to PCA
        EXPECT_EQ(result.size(), 2);
        EXPECT_FLOAT_EQ(result[0][0], -2.1507991383689053);
        EXPECT_FLOAT_EQ(result[0][1], 1.2463026830737149);
        EXPECT_FLOAT_EQ(result[0][2], -2.0771487788479344);
    }
}
