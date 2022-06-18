/**
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unistd.h>
#include <iostream>
#include <istream>
#include <sstream>
#include <cstring>
#include <fastText/fasttext.h>
#include <fastText/args.h>
#include <fastText/autotune.h>
#include <fasttext-wrapper.hpp>
#include <fstream>
using namespace fasttext;

extern "C" {

    fasttext::FastText ft_model;
    bool ft_initialized = false;

    bool ft_has_newline(std::string str) {
        return (0 == str.compare(str.length() - 1, 1, "\n"));
    };

    void printTestUsage() {
    std::cerr
        << "usage: fasttext test <model> <test-data> [<k>] [<th>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename (if -, read from stdin)\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << "  <th>         (optional; 0.0 by default) probability threshold\n"
        << std::endl;
    }

    void printTestLabelUsage() {
    std::cerr
        << "usage: fasttext test-label <model> <test-data> [<k>] [<th>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << "  <th>         (optional; 0.0 by default) probability threshold\n"
        << std::endl;
    }

   void test(const std::vector<std::string>& args) {
    bool perLabel = args[1] == "test-label";

    if (args.size() < 4 || args.size() > 6) {
        perLabel ? printTestLabelUsage() : printTestUsage();
        exit(EXIT_FAILURE);
    }

    const auto& model = args[2];
    const auto& input = args[3];
    int32_t k = args.size() > 4 ? std::stoi(args[4]) : 1;
    real threshold = args.size() > 5 ? std::stof(args[5]) : 0.0;

    FastText fasttext;
    fasttext.loadModel(model);

    Meter meter(false);

    if (input == "-") {
        fasttext.test(std::cin, k, threshold, meter);
    } else {
        std::ifstream ifs(input);
        if (!ifs.is_open()) {
        std::cerr << "Test file cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
        }
        fasttext.test(ifs, k, threshold, meter);
    }

    if (perLabel) {
        std::cout << std::fixed << std::setprecision(6);
        auto writeMetric = [](const std::string& name, double value) {
        std::cout << name << " : ";
        if (std::isfinite(value)) {
            std::cout << value;
        } else {
            std::cout << "--------";
        }
        std::cout << "  ";
        };

        std::shared_ptr<const Dictionary> dict = fasttext.getDictionary();
        for (int32_t labelId = 0; labelId < dict->nlabels(); labelId++) {
        writeMetric("F1-Score", meter.f1Score(labelId));
        writeMetric("Precision", meter.precision(labelId));
        writeMetric("Recall", meter.recall(labelId));
        std::cout << " " << dict->getLabel(labelId) << std::endl;
        }
    }
    meter.writeGeneralMetrics(std::cout, k);
    }
    
    void ft_train(fasttext::Args a) {
       std::shared_ptr<fasttext::FastText> fasttext = std::make_shared<fasttext::FastText>();
       std::string outputFileName;

        if (a.hasAutotune() &&
            a.getAutotuneModelSize() != fasttext::Args::kUnlimitedModelSize) {
            outputFileName = a.output + ".ftz";
        } else {
            outputFileName = a.output + ".bin";
        }
        std::ofstream ofs(outputFileName);
        if (!ofs.is_open()) {
            throw std::invalid_argument(
                outputFileName + " cannot be opened for saving.");
        }
        ofs.close();
        if (a.hasAutotune()) {
            fasttext::Autotune autotune(fasttext);
            autotune.train(a);
        } else {
            fasttext->train(a);
        }
        fasttext->saveModel(outputFileName);
        fasttext->saveVectors(a.output + ".vec");
        if (a.saveOutput) {
            fasttext->saveOutput(a.output + ".output");
        }
    }

    void ft_run(int argc, char** argv){
        std::vector<std::string> args(argv, argv + argc);
        fasttext::Args a = fasttext::Args();
        a.parseArgs(args);
        for (int ai = 2; ai < args.size(); ai += 2) {
            if (args[ai][0] != '-') {
            std::cerr << "Provided argument without a dash! Usage:" << args[ai][0] << std::endl;
            exit(EXIT_FAILURE);
            }
        }
        std::string command(args[1]);
        if (command == "skipgram" || command == "cbow" || command == "supervised") {
            ft_train(a);
        } else if (command == "test" || command == "test-label") {
            test(args);
        } 
    }

    int ft_load_model(const char *path) {
        if (!ft_initialized) {
            if(access(path, F_OK) != 0) {
                return -1;
            }
            ft_model.loadModel(std::string(path));
            ft_initialized = true;
        }
        return 0;
    }

    int ft_predict(const char *query_in, float *prob, char *out, int out_size) {

        int32_t k = 1;
        fasttext::real threshold = 0.0;

        std::string query(query_in);

        if(!ft_has_newline(query)) {
            query.append("\n");
        }

        std::istringstream inquery(query);
        std::istream &in = inquery;

        std::vector<std::pair<fasttext::real, std::string>> predictions;

        if(!ft_model.predictLine(in, predictions, k, threshold)) {
            *prob = -1;
            strncpy(out, "", out_size);
            return -1;
        }

        for(const auto &prediction : predictions) {
            *prob = prediction.first;
            strncpy(out, prediction.second.c_str(), out_size);
        }

        return 0;
    }

    int ft_get_vector_dimension()
    {
        if(!ft_initialized) {
            return -1;
        }
        return ft_model.getDimension();
    }

    int ft_get_sentence_vector(const char* query_in, float* vector_out, int vector_size)
    {
        std::string query(query_in);

        if(!ft_has_newline(query)) {
            query.append("\n");
        }

        std::istringstream inquery(query);
        std::istream &in = inquery;
        fasttext::Vector svec(ft_model.getDimension());

        ft_model.getSentenceVector(in, svec);
        if(svec.size() != vector_size) {
            return -1;
        }
        memcpy(vector_out, svec.data(), vector_size*sizeof(float));
        return 0;
    }
}
